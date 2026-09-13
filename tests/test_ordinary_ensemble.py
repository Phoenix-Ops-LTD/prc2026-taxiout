# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from carrier_contest import specialist_scope
from ensemble_contest import categorical
from nm_neighbor_features import completed_nm_neighbors
from ordinary_ensemble import OrdinaryConfig, blend, matrix, predict, train
from pipeline import ID, TARGET, TIME, sha256, write_json
from test_duration_contest import sample
from traffic_features import nm_schedule_fallback_baseline


def test_neighbor_context_excludes_self_future_arrivals_and_forbidden_fields() -> None:
    raw = pd.DataFrame({"PHASE_mvt": ["DEP", "DEP", "DEP", "DEP", "ARR", "DEP"],
        "ADEP_mvt": "EDDF", "ADEP_flt": "EDDF", "RUNWAY_mvt": "09", "STAND_mvt": "A1",
        "MVT_TIME_UTC_mvt": ["2025-01-01T11:50Z", "2025-01-01T11:50Z", "2025-01-01T12:00Z",
            "2025-01-01T12:10Z", "2025-01-01T11:55Z", "2024-12-31T23:55Z"],
        "AOBT_3_flt": ["2025-01-01T11:40Z", "2025-01-01T11:30Z", "2025-01-01T11:30Z",
            "2025-01-01T11:40Z", "2025-01-01T11:40Z", "2024-12-31T23:00Z"],
        TARGET: "FORBIDDEN", "BLOCK_TIME_UTC_mvt": "FORBIDDEN"}, index=[8, 6, 3, 4, 1, 9])
    dep = raw.loc[[3]]
    actual = completed_nm_neighbors(dep, raw)
    assert actual.shape == (1, 30)
    for group in ["airport", "runway", "stand"]:
        assert actual[f"nm_neighbor_{group}_15m_count"].iloc[0] == 2
        assert actual[f"nm_neighbor_{group}_15m_mean"].iloc[0] == 900
        assert actual[f"nm_neighbor_{group}_15m_std"].iloc[0] == 300
        assert actual[f"nm_neighbor_{group}_15m_own_minus_mean"].iloc[0] == 900
        assert actual[f"nm_neighbor_{group}_last_proxy"].iloc[0] == 900
    altered = raw.assign(TAXITIME_SEC_mvt=-999999, BLOCK_TIME_UTC_mvt="2000-01-01T00:00Z")
    altered.loc[4, "AOBT_3_flt"] = "2025-01-01T12:00Z"
    pd.testing.assert_frame_equal(actual, completed_nm_neighbors(dep, altered))
    pd.testing.assert_frame_equal(actual, completed_nm_neighbors(dep, raw.iloc[::-1]))
    assert completed_nm_neighbors(dep, raw.iloc[:0]).nm_neighbor_airport_15m_count.eq(0).all()
    assert completed_nm_neighbors(dep.iloc[:0], raw).shape == (0, 30)
    midnight = dep.copy()
    midnight["MVT_TIME_UTC_mvt"] = "2025-01-01T00:00Z"
    assert completed_nm_neighbors(midnight, raw).nm_neighbor_airport_15m_count.eq(0).all()
    with pytest.raises(ValueError, match="complete"):
        completed_nm_neighbors(dep.assign(MVT_TIME_UTC_mvt=pd.NaT), raw)


def test_fixed_ordinary_blend_and_configuration_guards() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG"], "missing_IOBT_flt": [1, 0]}, index=[4, 1])
    base = pd.Series([123.456789, 100.], index=x.index)
    neighbor = pd.Series([500., 200.], index=x.index)
    airport = pd.Series([1000., 400.], index=x.index)
    assert blend(x, base, neighbor, airport).tolist() == [123.456789, 150.]
    with pytest.raises(ValueError, match="alignment"):
        blend(x, base, neighbor.iloc[::-1], airport)
    for invalid in [-neighbor, neighbor * np.nan]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            blend(x, base, invalid, airport)
    for bad in [{"trees": 0}, {"trees": 680}, {"data_permission_ref": " "}]:
        with pytest.raises(ValidationError):
            OrdinaryConfig.model_validate({"data_permission_ref": "synthetic-test", **bad})


def test_ordinary_full_fit_inference_alignment_and_tampering(tmp_path: Path) -> None:
    raw = sample().assign(BLOCK_TIME_UTC_mvt=pd.NaT)
    raw.loc[:3, "ADEP_mvt"] = "LIRF"
    raw.loc[:3, "IOBT_flt"] = pd.NaT
    raw.loc[12, TARGET] = -7
    raw.loc[20, TARGET] = 20000
    data = tmp_path / "data"
    data.mkdir()
    for month, frame in raw.groupby(raw[TIME].dt.month):
        frame.to_parquet(data / f"training_{month:02d}.parquet", index=False)
    run = tmp_path / "model"
    config = OrdinaryConfig(trees=1, data_permission_ref="synthetic-test")
    train(data, run, config)
    dep, x, columns = matrix(raw)
    scope = specialist_scope(x)
    report = json.loads((run / "report.json").read_text())
    assert report["fit_rows"] == int((raw[TARGET].ge(0) & ~scope).sum())
    with pytest.raises(ValueError, match="fresh"):
        train(data, run, config)
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template = tmp_path / "template.parquet"
    baseline = tmp_path / "baseline.parquet"
    original = dep[[ID, TARGET]].copy().iloc[::-1].reset_index(drop=True)
    original[TARGET] = np.arange(len(original), dtype=float) + 123.456789
    original.to_parquet(template, index=False)
    original.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {"model_version": "prc2026-arrival-boost-blend/8.0.0",
        "submission_sha256": sha256(baseline), "ranking_sha256": sha256(ranking), "template_sha256": sha256(template)})
    output = tmp_path / "result.parquet"
    predict(run, baseline, ranking, template, output, "synthetic-test")
    actual = pd.read_parquet(output)
    encoded = categorical(x)
    offset = nm_schedule_fallback_baseline(x)
    neighbor = np.maximum(0, lgb.Booster(model_file=str(run / "neighbor-model.txt")).predict(encoded, num_threads=2) + offset)
    airport_values = pd.Series(0., index=x.index)
    for airport in report["airport_fit_rows"]:
        index = dep["ADEP_mvt"].eq(airport)
        model = lgb.Booster(model_file=str(run / f"airport-{airport}.txt"))
        airport_values.loc[index] = np.maximum(0, model.predict(encoded.loc[index, columns], num_threads=2) + offset.loc[index])
    combined = pd.Series((.5 * airport_values + .5 * neighbor).to_numpy(), index=dep[ID])
    expected = .75 * original[TARGET] + .25 * original[ID].map(combined)
    special = original[ID].isin(dep.loc[scope, ID])
    expected.loc[special] = original.loc[special, TARGET]
    np.testing.assert_array_equal(actual[TARGET], expected)
    pd.testing.assert_frame_equal(actual.loc[special], original.loc[special])
    report["candidate_weight"] = .5
    write_json(run / "report.json", report)
    with pytest.raises(ValueError, match="fixed ensemble"):
        predict(run, baseline, ranking, template, tmp_path / "bad-config.parquet", "synthetic-test")
    report["candidate_weight"] = .25
    write_json(run / "report.json", report)
    with (run / "neighbor-model.txt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        predict(run, baseline, ranking, template, tmp_path / "bad-model.parquet", "synthetic-test")

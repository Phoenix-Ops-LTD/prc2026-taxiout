# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from arrival_boost_contest import BoostConfig, blend, predict, train
from arrival_contest import matrix
from carrier_contest import specialist_scope
from pipeline import ID, TARGET, TIME, sha256, write_json
from test_duration_contest import sample
from traffic_features import nm_schedule_fallback_baseline


def test_boost_scope_and_configuration_guards() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG"], "missing_IOBT_flt": [1, 1]}, index=[4, 2])
    base = pd.Series([123.456789, 200.], index=x.index)
    candidate = pd.Series([999., 400.], index=x.index)
    assert blend(x, base, candidate).tolist() == [123.456789, 300.]
    for invalid, message in [(candidate.iloc[::-1], "alignment"), (candidate * np.nan, "finite"), (-candidate, "nonnegative")]:
        with pytest.raises(ValueError, match=message):
            blend(x, base, invalid)
    for values in [{"trees": 0}, {"trees": 5001}, {"device": "invalid"}, {"data_permission_ref": " "}]:
        with pytest.raises(ValidationError):
            BoostConfig.model_validate({"data_permission_ref": "synthetic-test", **values})


def test_full_training_and_inference_round_trip_keeps_v7_specialist(tmp_path: Path) -> None:
    raw = sample().assign(BLOCK_TIME_UTC_mvt=pd.NaT)
    raw.loc[:7, "ADEP_mvt"] = "LIRF"
    raw.loc[:7, "IOBT_flt"] = pd.NaT
    raw.loc[12, TARGET] = -7
    raw.loc[20, TARGET] = 20000
    data = tmp_path / "data"
    data.mkdir()
    for month, g in raw.groupby(raw[TIME].dt.month):
        g.to_parquet(data / f"training_{month:02d}.parquet", index=False)
    run = tmp_path / "models"
    train(data, run, BoostConfig(trees=3, device="CPU", data_permission_ref="synthetic-test"))
    report = json.loads((run / "report.json").read_text())
    dep, x = matrix(raw)
    scope = specialist_scope(x)
    assert report["fit_rows"] == int((raw[TARGET].ge(0) & ~scope).sum())
    assert report["residual_cap"] == 7200
    with pytest.raises(ValueError, match="fresh"):
        train(data, run, BoostConfig(trees=3, device="CPU", data_permission_ref="synthetic-test"))
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template = tmp_path / "template.parquet"
    baseline = tmp_path / "baseline.parquet"
    original = dep[[ID, TARGET]].copy().iloc[::-1].reset_index(drop=True)
    original[TARGET] = np.arange(len(original), dtype=float) + 123.456789
    original.to_parquet(template, index=False)
    original.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {"model_version": "prc2026-arrival-blend/7.0.0",
        "submission_sha256": sha256(baseline), "ranking_sha256": sha256(ranking), "template_sha256": sha256(template)})
    output = tmp_path / "candidate.parquet"
    predict(run, baseline, ranking, template, output, "synthetic-test")
    actual = pd.read_parquet(output)
    model = CatBoostRegressor()
    model.load_model(str(run / "model.cbm"))
    expected_model = pd.Series(np.maximum(0, model.predict(x, thread_count=2)
        + nm_schedule_fallback_baseline(x)).to_numpy(), index=dep[ID])
    expected = .5 * original[TARGET] + .5 * original[ID].map(expected_model)
    special = original[ID].isin(dep.loc[scope, ID])
    expected.loc[special] = original.loc[special, TARGET]
    np.testing.assert_array_equal(actual[TARGET], expected)
    pd.testing.assert_frame_equal(actual.loc[special], original.loc[special])
    report["parameters"]["depth"] = 8
    write_json(run / "report.json", report)
    with pytest.raises(ValueError, match="parameters"):
        predict(run, baseline, ranking, template, tmp_path / "bad-parameters.parquet", "synthetic-test")
    report["parameters"]["depth"] = 9
    write_json(run / "report.json", report)
    with (run / "model.cbm").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        predict(run, baseline, ranking, template, tmp_path / "bad-model.parquet", "synthetic-test")

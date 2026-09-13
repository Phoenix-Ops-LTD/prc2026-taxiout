# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from arrival_boost_contest import BoostConfig, train as train_old
from carrier_contest import specialist_scope
from neighbor_boost_contest import (
    NeighborBoostConfig, V8_VERSION, V9_VERSION, predict, replace_component, train,
)
from ordinary_ensemble import matrix
from pipeline import ID, TARGET, TIME, sha256, write_json
from test_duration_contest import sample
from traffic_features import nm_schedule_fallback_baseline


def test_replacement_algebra_preserves_specialist_and_does_not_clip() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG", "LIRF"], "missing_IOBT_flt": [1, 1, 0]}, index=[9, 2, 7])
    baseline = pd.Series([123.456789, 200., 400.], index=x.index)
    new = pd.Series([999., 300., 100.], index=x.index)
    old = pd.Series([10., 100., 500.], index=x.index)
    np.testing.assert_array_equal(replace_component(x, baseline, new, old), [123.456789, 275., 250.])
    for invalid in [new * np.nan, -new, new * np.inf]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace_component(x, baseline, invalid, old)
    with pytest.raises(ValueError, match="alignment"):
        replace_component(x, baseline, new.iloc[::-1], old)
    with pytest.raises(ValueError, match="do not clip"):
        replace_component(x, baseline, new, old * 100.)
    with pytest.raises(ValueError, match="do not clip"):
        replace_component(x, pd.Series(1.7e308, index=x.index), pd.Series(1.7e308, index=x.index), old)


def test_competition_configuration_is_frozen_and_synthetic_runs_are_explicit() -> None:
    assert NeighborBoostConfig(data_permission_ref="synthetic-test").trees == 4999
    for invalid in [{"trees": 0}, {"trees": 5000}, {"device": "other"},
            {"data_permission_ref": " "}, {"trees": 3}, {"device": "CPU"}]:
        with pytest.raises(ValidationError):
            NeighborBoostConfig.model_validate({"data_permission_ref": "synthetic-test", **invalid})
    assert NeighborBoostConfig(trees=3, device="CPU", data_class="synthetic", data_permission_ref="synthetic-test").trees == 3


def test_full_synthetic_round_trip_binds_old_component_and_preserves_all_rows(tmp_path: Path) -> None:
    raw = sample().assign(BLOCK_TIME_UTC_mvt=pd.NaT)
    # Closely spaced departures provide actual strict-past neighbor context.
    for month, group in raw.groupby(raw[TIME].dt.month):
        raw.loc[group.index, TIME] = pd.Timestamp(year=2025, month=month, day=1, tz="UTC") + pd.to_timedelta(np.arange(len(group)) * 5, unit="m")
    raw["MVT_TIME_UTC_mvt"] = raw[TIME] + pd.Timedelta(minutes=20)
    for key, minutes in {"IOBT_flt": 0, "EOBT_1_flt": 0, "AOBT_3_flt": 5,
            "LOBT_flt": 5, "ARVT_1_flt": 60, "ARVT_3_flt": 75}.items():
        raw[key] = raw[TIME] + pd.Timedelta(minutes=minutes)
    raw["AOBT_3_flt"] -= pd.to_timedelta(np.arange(len(raw)) % 11, unit="m")
    raw.loc[:3, "ADEP_mvt"] = "LIRF"
    raw.loc[:3, "IOBT_flt"] = pd.NaT
    raw.loc[12, TARGET] = -7
    raw.loc[20, TARGET] = 20000
    original_labels = raw[TARGET].copy()
    data = tmp_path / "data"
    data.mkdir()
    for month, frame in raw.groupby(raw[TIME].dt.month):
        frame.to_parquet(data / f"training_{month:02d}.parquet", index=False)
    run, old_run = tmp_path / "new-model", tmp_path / "old-model"
    config = NeighborBoostConfig(trees=3, device="CPU", data_class="synthetic", data_permission_ref="synthetic-test")
    train(data, run, config)
    train_old(data, old_run, BoostConfig(trees=3, device="CPU", data_permission_ref="synthetic-test"))
    report = json.loads((run / "report.json").read_text())
    dep, x, old_columns = matrix(raw)
    scope = specialist_scope(x)
    assert report["status"] == "SYNTHETIC_TEST_ONLY"
    assert report["fit_rows"] == int((raw[TARGET].ge(0) & ~scope).sum())
    assert report["negative_label_rows"] == 1
    assert report["departure_rows"] == len(raw)
    assert report["residual_cap"] == 7200
    assert len(report["features"]) == 153
    pd.testing.assert_series_equal(raw[TARGET], original_labels)
    with pytest.raises(ValueError, match="fresh"):
        train(data, run, config)
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template, baseline, v8_baseline = (tmp_path / name for name in ["template.parquet", "v9.parquet", "v8.parquet"])
    original = dep[[ID, TARGET]].copy().iloc[::-1].reset_index(drop=True)
    original[TARGET] = np.arange(len(original), dtype=float) + 1234.56789
    original.to_parquet(template, index=False)
    original.to_parquet(baseline, index=False)
    v8_values = original[TARGET] + 100
    special = original[ID].isin(dep.loc[scope, ID])
    v8_values.loc[special] = original.loc[special, TARGET]
    original.assign(TAXITIME_SEC_mvt=v8_values).to_parquet(v8_baseline, index=False)
    common = {"data_class": "synthetic", "ranking_sha256": sha256(ranking), "template_sha256": sha256(template),
        "rows": len(original), "specialist_rows": int(scope.sum()), "residual_fit_cap": 7200}
    v8_manifest = {**common, "model_version": V8_VERSION, "submission_sha256": sha256(v8_baseline),
        "baseline_model_version": "prc2026-arrival-blend/7.0.0", "boost_weight": .5,
        "boost_model_sha256": sha256(old_run / "model.cbm")}
    v9_manifest = {**common, "model_version": V9_VERSION, "submission_sha256": sha256(baseline),
        "baseline_sha256": sha256(v8_baseline), "baseline_model_version": V8_VERSION,
        "candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5}
    write_json(v8_baseline.with_suffix(".manifest.json"), v8_manifest)
    write_json(baseline.with_suffix(".manifest.json"), v9_manifest)
    output = tmp_path / "candidate.parquet"
    predict(run, old_run, baseline, v8_baseline, ranking, template, output, "synthetic-test")
    actual = pd.read_parquet(output)
    new_model, old_model = CatBoostRegressor(), CatBoostRegressor()
    new_model.load_model(str(run / "model.cbm"))
    old_model.load_model(str(old_run / "model.cbm"))
    offset = nm_schedule_fallback_baseline(x)
    new_values = np.maximum(0, new_model.predict(x, thread_count=2) + offset)
    old_values = np.maximum(0, old_model.predict(x.loc[:, old_columns], thread_count=2) + offset)
    delta = pd.Series((new_values - old_values).to_numpy(), index=dep[ID])
    expected = original[TARGET] + .375 * original[ID].map(delta)
    special = original[ID].isin(dep.loc[scope, ID])
    expected.loc[special] = original.loc[special, TARGET]
    assert (expected.loc[~special] != original.loc[~special, TARGET]).any()
    np.testing.assert_array_equal(actual[TARGET], expected)
    pd.testing.assert_frame_equal(actual.loc[special], original.loc[special])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["data_class"] == "synthetic"
    assert manifest["old_model_sha256"] == v8_manifest["boost_model_sha256"]
    assert manifest["baseline_manifest_sha256"] == sha256(baseline.with_suffix(".manifest.json"))
    assert manifest["new_report_sha256"] == sha256(run / "report.json")
    with pytest.raises(ValueError, match="fresh"):
        predict(run, old_run, baseline, v8_baseline, ranking, template, output, "synthetic-test")

    # Each altered link must fail before producing a submission.
    counter = 0
    for path, source, field, value, message in [
        (baseline.with_suffix(".manifest.json"), v9_manifest, "baseline_sha256", "0" * 64, "not bound"),
        (v8_baseline.with_suffix(".manifest.json"), v8_manifest, "boost_model_sha256", sha256(run / "model.cbm"), "Old model digest"),
        (baseline.with_suffix(".manifest.json"), v9_manifest, "submission_sha256", "0" * 64, "Baseline digest"),
        (v8_baseline.with_suffix(".manifest.json"), v8_manifest, "ranking_sha256", "0" * 64, "ranking or template"),
        (baseline.with_suffix(".manifest.json"), v9_manifest, "template_sha256", "0" * 64, "ranking or template"),
        (baseline.with_suffix(".manifest.json"), v9_manifest, "candidate_weight", .5, "weights"),
        (baseline.with_suffix(".manifest.json"), v9_manifest, "specialist_rows", 0, "specialist"),
        (run / "report.json", report, "old_features", old_columns[::-1], "Feature schema"),
        (run / "report.json", report, "tree_count", 2, "tree count"),
        (run / "report.json", report, "replacement_weight", .5, "replacement contract"),
        (run / "report.json", report, "input_hashes", {}, "same twelve"),
        (run / "report.json", report, "parameters", {**report["parameters"], "depth": 8}, "parameters"),
        (run / "report.json", report, "categories", [], "categorical"),
    ]:
        counter += 1
        write_json(path, {**source, field: value})
        rejected_output = tmp_path / f"rejected-{counter}.parquet"
        with pytest.raises(ValueError, match=message):
            predict(run, old_run, baseline, v8_baseline, ranking, template, rejected_output, "synthetic-test")
        assert not rejected_output.exists()
        write_json(path, source)
    changed_report = {**report, "tree_count": 2, "config": {**report["config"], "trees": 2},
        "parameters": {**report["parameters"], "iterations": 2}}
    write_json(run / "report.json", changed_report)
    with pytest.raises(ValueError, match="Stored tree count"):
        predict(run, old_run, baseline, v8_baseline, ranking, template, tmp_path / "false-trees.parquet", "synthetic-test")
    write_json(run / "report.json", report)
    changed_v9 = original.copy()
    changed_v9.loc[special, TARGET] += 1
    changed_v9.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {**v9_manifest, "submission_sha256": sha256(baseline)})
    with pytest.raises(ValueError, match="changed the v8 specialist"):
        predict(run, old_run, baseline, v8_baseline, ranking, template, tmp_path / "changed-specialist.parquet", "synthetic-test")
    original.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), v9_manifest)
    with (old_run / "model.cbm").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Old model digest"):
        predict(run, old_run, baseline, v8_baseline, ranking, template, tmp_path / "tampered-model.parquet", "synthetic-test")

# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from carrier_contest import specialist_scope
from clock_overlay_contest import predict as clock_predict
from following_boost_contest import (
    FollowingBoostConfig, V10_VERSION, V11_VERSION, clock_scope, matrix, predict, replace_component, train,
)
from neighbor_boost_contest import NeighborBoostConfig, train as train_old
from pipeline import ID, TARGET, TIME, sha256, write_json
from test_duration_contest import sample
from traffic_features import nm_schedule_fallback_baseline


def test_replacement_preserves_clock_and_specialist_before_arithmetic() -> None:
    time = pd.Timestamp("2025-01-01T12:00Z")
    dep = pd.DataFrame({ID: [11., 12., 13.], "PHASE_mvt": "DEP", "ADEP_mvt": ["LIRF", "LFPG", "LFPG"],
        "ADEP_flt": ["LIRF", "LFPG", "LFPG"], "MVT_TIME_UTC_mvt": time,
        "IOBT_flt": [pd.NaT, time, time], "AOBT_3_flt": time - pd.Timedelta(minutes=10),
        "LOBT_flt": [time, time-pd.Timedelta(minutes=150), time]}, index=[9, 3, 7])
    x = pd.DataFrame({"ADEP_mvt": dep.ADEP_mvt, "missing_IOBT_flt": [1, 0, 0]}, index=dep.index)
    baseline = pd.Series([123.456789, 9000., 200.], index=dep.index)
    new = pd.Series([1., 1., 300.], index=dep.index)
    old = pd.Series([1e300, 1e300, 100.], index=dep.index)
    np.testing.assert_array_equal(clock_scope(dep), [False, True, False])
    np.testing.assert_array_equal(replace_component(dep, x, baseline, new, old), [123.456789, 9000., 275.])
    for invalid in [new * np.nan, -new, new * np.inf]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace_component(dep, x, baseline, invalid, old)
    with pytest.raises(ValueError, match="alignment"):
        replace_component(dep, x, baseline, new.iloc[::-1], old)
    with pytest.raises(ValueError, match="do not clip"):
        replace_component(dep, x, baseline, new, old.replace(100., 10000.))


def test_clock_scope_handles_the_strict_positive_replacement_boundary() -> None:
    time = pd.Timestamp("2025-01-01T12:00:00Z")
    dep = pd.DataFrame({ID: [1., 2.], "PHASE_mvt": "DEP", "ADEP_mvt": "EDDF", "ADEP_flt": "EDDF",
        "MVT_TIME_UTC_mvt": time, "IOBT_flt": time, "AOBT_3_flt": time + pd.Timedelta(seconds=7200),
        "LOBT_flt": [time - pd.Timedelta(microseconds=1), time]})
    np.testing.assert_array_equal(clock_scope(dep), [True, False])


def test_competition_configuration_preserves_the_frozen_protocol() -> None:
    assert FollowingBoostConfig(data_permission_ref="fixture").trees == 4999
    invalid_configurations: list[dict[str, object]] = [{"trees": 0}, {"trees": 5000}, {"trees": 3}, {"device": "CPU"},
        {"data_permission_ref": " "}, {"model_version": V10_VERSION}, {"unexpected": 1}]
    for invalid in invalid_configurations:
        with pytest.raises(ValidationError):
            FollowingBoostConfig.model_validate({"data_permission_ref": "fixture", **invalid})
    assert FollowingBoostConfig(trees=3, device="CPU", data_class="synthetic", data_permission_ref="fixture").trees == 3


def test_synthetic_round_trip_preserves_v11_and_checks_model_lineage(tmp_path: Path) -> None:
    raw = sample()
    raw["BLOCK_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    for month, group in raw.groupby(raw[TIME].dt.month):
        raw.loc[group.index, TIME] = pd.Timestamp(year=2025, month=month, day=1, tz="UTC") + pd.to_timedelta(np.arange(len(group))*5, unit="m")
    raw["MVT_TIME_UTC_mvt"] = raw[TIME] + pd.Timedelta(minutes=20)
    for key, minutes in {"IOBT_flt": 0, "EOBT_1_flt": 0, "AOBT_3_flt": 5,
            "LOBT_flt": 5, "ARVT_1_flt": 60, "ARVT_3_flt": 75}.items():
        raw[key] = raw[TIME] + pd.Timedelta(minutes=minutes)
    raw["AOBT_3_flt"] -= pd.to_timedelta(np.arange(len(raw)) % 11, unit="m")
    raw.loc[:3, "ADEP_mvt"] = "LIRF"
    raw.loc[:3, "IOBT_flt"] = pd.NaT
    raw.loc[12, TARGET] = -7
    raw.loc[20, TARGET] = 20000
    raw.loc[5, "ADEP_flt"] = raw.loc[5, "ADEP_mvt"]
    clock = raw.loc[5, "MVT_TIME_UTC_mvt"]
    assert isinstance(clock, pd.Timestamp)
    raw.loc[5, "LOBT_flt"] = clock - pd.Timedelta(seconds=9000)
    original_labels = raw[TARGET].copy()
    data = tmp_path / "data"
    data.mkdir()
    for month, frame in raw.groupby(raw[TIME].dt.month):
        frame.to_parquet(data / f"training_{month:02d}.parquet", index=False)
    run, old_run = tmp_path / "new-model", tmp_path / "old-model"
    config = FollowingBoostConfig(trees=3, device="CPU", data_class="synthetic", data_permission_ref="fixture")
    train(data, run, config)
    train_old(data, old_run, NeighborBoostConfig(trees=3, device="CPU", data_class="synthetic", data_permission_ref="fixture"))
    report = json.loads((run / "report.json").read_text())
    old_report = json.loads((old_run / "report.json").read_text())
    dep, x, old_columns = matrix(raw)
    scope = specialist_scope(x)
    assert report["status"] == "SYNTHETIC_TEST_ONLY" and report["tree_count"] == 3
    assert report["fit_rows"] == int((raw[TARGET].ge(0) & ~scope).sum())
    assert report["negative_label_rows"] == 1 and report["capped_fit_rows"] >= 1
    assert len(report["features"]) == 183 and len(old_columns) == 153
    pd.testing.assert_series_equal(raw[TARGET], original_labels)
    for name, digest in report["source_sha256"].items():
        assert sha256(run / "source-snapshot" / name) == digest
    with pytest.raises(ValueError, match="fresh"):
        train(data, run, config)
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template, v10_path, baseline = (tmp_path / name for name in ["template.parquet", "v10.parquet", "v11.parquet"])
    original = dep[[ID, TARGET]].iloc[::-1].reset_index(drop=True).copy()
    original[TARGET] = np.arange(len(original), dtype=float) + 1234.56789
    original.to_parquet(template, index=False)
    original.to_parquet(v10_path, index=False)
    v10 = {"data_class": "synthetic", "model_version": V10_VERSION, "submission_sha256": sha256(v10_path),
        "ranking_sha256": sha256(ranking), "template_sha256": sha256(template), "rows": len(original),
        "specialist_rows": int(scope.sum()), "residual_fit_cap": 7200, "replacement_weight": .375,
        "baseline_model_version": "prc2026-ordinary-ensemble/9.0.0", "new_model_sha256": sha256(old_run / "model.cbm"),
        "new_report_sha256": sha256(old_run / "report.json")}
    write_json(v10_path.with_suffix(".manifest.json"), v10)
    clock_predict(v10_path, ranking, template, baseline, "fixture")
    v11 = json.loads(baseline.with_suffix(".manifest.json").read_text())
    assert v11["model_version"] == V11_VERSION and v11["changed_prediction_rows"] == 1
    output = tmp_path / "candidate.parquet"
    predict(run, old_run, baseline, v10_path, ranking, template, output, "fixture")
    actual, previous = pd.read_parquet(output), pd.read_parquet(baseline)
    new_model, old_model = CatBoostRegressor(), CatBoostRegressor()
    new_model.load_model(str(run / "model.cbm"))
    old_model.load_model(str(old_run / "model.cbm"))
    offset = nm_schedule_fallback_baseline(x)
    new_values = np.maximum(0, new_model.predict(x, thread_count=2) + offset)
    old_values = np.maximum(0, old_model.predict(x.loc[:, old_columns], thread_count=2) + offset)
    delta = pd.Series(np.asarray(new_values-old_values), index=dep[ID])
    expected = previous[TARGET] + .375 * previous[ID].map(delta)
    protected_ids = set(dep.loc[scope, ID]) | {dep.loc[5, ID]}
    protected = previous[ID].isin(protected_ids)
    expected.loc[protected] = previous.loc[protected, TARGET]
    assert (expected.loc[~protected] != previous.loc[~protected, TARGET]).any()
    np.testing.assert_array_equal(actual[TARGET], expected)
    pd.testing.assert_frame_equal(actual.loc[protected], previous.loc[protected])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["clock_rows"] == 1 and manifest["specialist_rows"] == 4
    assert manifest["new_model_sha256"] == sha256(run / "model.cbm")
    assert manifest["old_model_sha256"] == v10["new_model_sha256"]
    with pytest.raises(ValueError, match="fresh"):
        predict(run, old_run, baseline, v10_path, ranking, template, output, "fixture")
    tampering = [
        (baseline.with_suffix(".manifest.json"), v11, "baseline_sha256", "0"*64, "not bound"),
        (baseline.with_suffix(".manifest.json"), v11, "proxy_raw_clip", [0, 7200], "clock-rule contract"),
        (v10_path.with_suffix(".manifest.json"), v10, "new_model_sha256", "0"*64, "not bound"),
        (run / "report.json", report, "old_features", old_columns[::-1], "Feature schema"),
        (run / "report.json", report, "tree_count", 2, "tree count"),
        (run / "report.json", report, "replacement_weight", .5, "replacement contract"),
        (run / "report.json", report, "source_sha256", {}, "source digests"),
        (run / "report.json", report, "source_sha256", {k: v for k, v in report["source_sha256"].items() if k != "pipeline.py"}, "source digests"),
        (run / "report.json", report, "input_hashes", {}, "same twelve"),
        (run / "report.json", report, "parameters", {**report["parameters"], "depth": 8}, "parameters"),
        (run / "report.json", report, "categories", [], "categorical"),
    ]
    for count, (path, source, field, value, message) in enumerate(tampering):
        write_json(path, {**source, field: value})
        rejected = tmp_path / f"rejected-{count}.parquet"
        with pytest.raises(ValueError, match=message):
            predict(run, old_run, baseline, v10_path, ranking, template, rejected, "fixture")
        assert not rejected.exists()
        write_json(path, source)
    changed = previous.copy()
    changed.loc[~protected, TARGET] += 1
    changed.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {**v11, "submission_sha256": sha256(baseline)})
    with pytest.raises(ValueError, match="differs from the fixed clock overlay"):
        predict(run, old_run, baseline, v10_path, ranking, template, tmp_path / "changed-baseline.parquet", "fixture")
    previous.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), v11)
    with (old_run / "model.cbm").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Old model and report"):
        predict(run, old_run, baseline, v10_path, ranking, template, tmp_path / "changed-model.parquet", "fixture")
    assert old_report["input_hashes"] == report["input_hashes"]

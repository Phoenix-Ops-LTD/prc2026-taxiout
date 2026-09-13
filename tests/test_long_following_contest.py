# SPDX-License-Identifier: GPL-3.0-only
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from carrier_contest import specialist_scope
from clock_overlay_contest import predict as clock_predict
from ensemble_contest import categorical
from long_following_contest import (
    BASELINE_VERSIONS, LongFollowingConfig, clock_scope, matrix, parameters, predict, replace_component, train,
)
from ordinary_ensemble import OrdinaryConfig, parameters as old_parameters, train as train_old
from pipeline import ID, TARGET, TIME, sha256, write_json
from test_duration_contest import sample
from traffic_features import nm_schedule_fallback_baseline


def test_configuration_fixes_the_original_parameters_except_tree_count() -> None:
    config = LongFollowingConfig(data_permission_ref="fixture")
    assert config.trees == 2716
    assert {**parameters(config), "n_estimators": 679} == old_parameters(OrdinaryConfig(data_permission_ref="fixture"))
    invalid_configurations: list[dict[str, object]] = [{"trees": 0}, {"trees": 2717}, {"trees": 3},
        {"data_permission_ref": " "}, {"model_version": BASELINE_VERSIONS[0]}, {"device": "GPU"}, {"unexpected": 1}]
    for invalid in invalid_configurations:
        with pytest.raises(ValidationError):
            LongFollowingConfig.model_validate({"data_permission_ref": "fixture", **invalid})
    assert LongFollowingConfig(trees=5, data_class="synthetic", data_permission_ref="fixture").trees == 5


def test_replacement_protects_clock_and_specialist_before_arithmetic() -> None:
    time = pd.Timestamp("2025-01-01T12:00Z")
    dep = pd.DataFrame({ID: [11., 12., 13.], "PHASE_mvt": "DEP", "ADEP_mvt": ["LIRF", "LFPG", "LFPG"],
        "ADEP_flt": ["LIRF", "LFPG", "LFPG"], "MVT_TIME_UTC_mvt": time,
        "IOBT_flt": [pd.NaT, time, time], "AOBT_3_flt": time,
        "LOBT_flt": [time, time-pd.Timedelta(minutes=150), time]}, index=[9, 3, 7])
    x = pd.DataFrame({"ADEP_mvt": dep.ADEP_mvt, "missing_IOBT_flt": [1, 0, 0]}, index=dep.index)
    baseline = pd.Series([123.456789, 9000., 200.], index=dep.index)
    new = pd.Series([1., 1., 300.], index=dep.index)
    old = pd.Series([1e300, 1e300, 100.], index=dep.index)
    np.testing.assert_array_equal(replace_component(dep, x, baseline, new, old), [123.456789, 9000., 225.])
    for invalid in [new*np.nan, -new, new*np.inf]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace_component(dep, x, baseline, invalid, old)
    with pytest.raises(ValueError, match="alignment"):
        replace_component(dep, x, baseline, new.iloc[::-1], old)
    with pytest.raises(ValueError, match="do not clip"):
        replace_component(dep, x, baseline, new, old.replace(100., 10000.))


def test_actual_synthetic_models_round_trip_and_reject_broken_lineage(tmp_path: Path) -> None:
    raw = sample()
    raw["BLOCK_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    for month, group in raw.groupby(raw[TIME].dt.month):
        raw.loc[group.index, TIME] = pd.Timestamp(year=2025, month=month, day=1, tz="UTC")+pd.to_timedelta(np.arange(len(group))*5, unit="m")
    raw["MVT_TIME_UTC_mvt"] = raw[TIME]+pd.Timedelta(minutes=20)
    for key, minutes in {"IOBT_flt": 0, "EOBT_1_flt": 0, "AOBT_3_flt": 5, "LOBT_flt": 5, "ARVT_1_flt": 60, "ARVT_3_flt": 75}.items():
        raw[key] = raw[TIME]+pd.Timedelta(minutes=minutes)
    raw["AOBT_3_flt"] -= pd.to_timedelta(np.arange(len(raw)) % 11, unit="m")
    raw.loc[:3, "ADEP_mvt"] = "LIRF"
    raw.loc[:3, "IOBT_flt"] = pd.NaT
    raw.loc[12, TARGET] = -7
    raw.loc[20, TARGET] = 20000
    raw.loc[5, "ADEP_flt"] = raw.loc[5, "ADEP_mvt"]
    clock = raw.loc[5, "MVT_TIME_UTC_mvt"]
    assert isinstance(clock, pd.Timestamp)
    raw.loc[5, "LOBT_flt"] = clock-pd.Timedelta(seconds=9000)
    original_labels = raw[TARGET].copy()
    data = tmp_path / "data"
    data.mkdir()
    for month, frame in raw.groupby(raw[TIME].dt.month):
        frame.to_parquet(data / f"training_{month:02d}.parquet", index=False)
    run, old_run = tmp_path / "new-model", tmp_path / "old-model"
    train(data, run, LongFollowingConfig(trees=5, data_class="synthetic", data_permission_ref="fixture"))
    train_old(data, old_run, OrdinaryConfig(trees=3, data_permission_ref="fixture"))
    report = json.loads((run / "report.json").read_text())
    old_report = json.loads((old_run / "report.json").read_text())
    dep, x, old_columns = matrix(raw)
    special, clock_mask = specialist_scope(x), clock_scope(dep)
    assert report["status"] == "SYNTHETIC_TEST_ONLY" and report["tree_count"] == 5
    assert report["fit_rows"] == int((raw[TARGET].ge(0) & ~special).sum())
    assert report["negative_label_rows"] == 1 and report["capped_fit_rows"] >= 1
    assert report["input_hashes"] == old_report["input_hashes"]
    pd.testing.assert_series_equal(raw[TARGET], original_labels)
    for name, digest in report["source_sha256"].items():
        assert sha256(run / "source-snapshot" / name) == digest
    with pytest.raises(ValueError, match="fresh"):
        train(data, run, LongFollowingConfig(trees=5, data_class="synthetic", data_permission_ref="fixture"))
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template, v9_path, v10_path, v11_path, baseline = (tmp_path / name for name in ["template.parquet", "v9.parquet", "v10.parquet", "v11.parquet", "v12.parquet"])
    original = dep[[ID, TARGET]].iloc[::-1].reset_index(drop=True).copy()
    original[TARGET] = np.arange(len(original), dtype=float)+1234.56789
    for path in [template, v9_path, v10_path]:
        original.to_parquet(path, index=False)
    common = {"data_class": "synthetic", "ranking_sha256": sha256(ranking), "template_sha256": sha256(template),
        "rows": len(original), "specialist_rows": int(special.sum()), "residual_fit_cap": 7200}
    v9 = {**common, "model_version": BASELINE_VERSIONS[3], "submission_sha256": sha256(v9_path),
        "candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5, "model_sha256": old_report["model_sha256"]}
    write_json(v9_path.with_suffix(".manifest.json"), v9)
    v10 = {**common, "model_version": BASELINE_VERSIONS[2], "submission_sha256": sha256(v10_path),
        "baseline_sha256": sha256(v9_path), "baseline_manifest_sha256": sha256(v9_path.with_suffix(".manifest.json")),
        "baseline_model_version": BASELINE_VERSIONS[3], "replacement_weight": .375,
        "new_model_sha256": "1"*64, "new_report_sha256": "2"*64}
    write_json(v10_path.with_suffix(".manifest.json"), v10)
    clock_predict(v10_path, ranking, template, v11_path, "fixture")
    v11 = json.loads(v11_path.with_suffix(".manifest.json").read_text())
    previous = pd.read_parquet(v11_path)
    previous.to_parquet(baseline, index=False)
    v12 = {**common, "model_version": BASELINE_VERSIONS[0], "submission_sha256": sha256(baseline),
        "baseline_sha256": sha256(v11_path), "baseline_manifest_sha256": sha256(v11_path.with_suffix(".manifest.json")),
        "baseline_model_version": BASELINE_VERSIONS[1], "v10_baseline_sha256": sha256(v10_path),
        "v10_manifest_sha256": sha256(v10_path.with_suffix(".manifest.json")), "replacement_weight": .375,
        "old_model_sha256": v10["new_model_sha256"], "old_report_sha256": v10["new_report_sha256"],
        "source_sha256": v11["source_sha256"], "clock_rule_version": "prc2026-nm-clock-rule/1.0.0", "clock_rows": int(clock_mask.sum())}
    write_json(baseline.with_suffix(".manifest.json"), v12)
    output = tmp_path / "candidate.parquet"
    predict(run, old_run, baseline, v11_path, v10_path, v9_path, ranking, template, output, "fixture")
    actual = pd.read_parquet(output)
    new_model = lgb.Booster(model_file=str(run / "model.txt"))
    old_model = lgb.Booster(model_file=str(old_run / "neighbor-model.txt"))
    offset = nm_schedule_fallback_baseline(x)
    cats = categorical(x)
    new_values = np.maximum(0, new_model.predict(cats, num_threads=2)+offset)
    old_values = np.maximum(0, old_model.predict(cats.loc[:, old_columns], num_threads=2)+offset)
    delta = pd.Series(np.asarray(new_values-old_values), index=dep[ID])
    expected = previous[TARGET]+.125*previous[ID].map(delta)
    protected = previous[ID].isin(dep.loc[special | clock_mask, ID])
    expected.loc[protected] = previous.loc[protected, TARGET]
    assert (expected.loc[~protected] != previous.loc[~protected, TARGET]).any()
    np.testing.assert_array_equal(actual[TARGET], expected)
    pd.testing.assert_frame_equal(actual.loc[protected], previous.loc[protected])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["clock_rows"] == 1 and manifest["specialist_rows"] == 4
    assert manifest["new_model_sha256"] == sha256(run / "model.txt")
    assert manifest["old_model_sha256"] == old_report["model_sha256"]["neighbor"]
    with pytest.raises(ValueError, match="fresh"):
        predict(run, old_run, baseline, v11_path, v10_path, v9_path, ranking, template, output, "fixture")
    tampering = [
        (baseline.with_suffix(".manifest.json"), v12, "baseline_sha256", "0"*64, "lineage"),
        (baseline.with_suffix(".manifest.json"), v12, "clock_rows", 0, "protected-row"),
        (baseline.with_suffix(".manifest.json"), v12, "old_model_sha256", "0"*64, "CatBoost model lineage"),
        (run / "report.json", report, "old_features", old_columns[::-1], "Feature schema"),
        (run / "report.json", report, "tree_count", 2, "tree count"),
        (run / "report.json", report, "replacement_weight", .5, "replacement contract"),
        (run / "report.json", report, "source_sha256", {}, "source digests"),
        (run / "report.json", report, "source_sha256", {k: v for k, v in report["source_sha256"].items() if k != "pipeline.py"}, "source digests"),
        (run / "report.json", report, "input_hashes", {}, "same twelve"),
        (run / "report.json", report, "parameters", {**report["parameters"], "num_leaves": 31}, "parameters"),
        (run / "report.json", report, "categories", [], "categorical"),
        (old_run / "report.json", old_report, "parameters", {**old_report["parameters"], "num_leaves": 31}, "parameters"),
    ]
    for count, (path, source, field, value, message) in enumerate(tampering):
        write_json(path, {**source, field: value})
        rejected = tmp_path / f"rejected-{count}.parquet"
        with pytest.raises(ValueError, match=message):
            predict(run, old_run, baseline, v11_path, v10_path, v9_path, ranking, template, rejected, "fixture")
        assert not rejected.exists()
        write_json(path, source)
    changed = previous.copy()
    changed.loc[protected, TARGET] += 1
    changed.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {**v12, "submission_sha256": sha256(baseline)})
    with pytest.raises(ValueError, match="protected v11 values"):
        predict(run, old_run, baseline, v11_path, v10_path, v9_path, ranking, template, tmp_path / "changed-baseline.parquet", "fixture")
    previous.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), v12)
    with (old_run / "neighbor-model.txt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Old model digest"):
        predict(run, old_run, baseline, v11_path, v10_path, v9_path, ranking, template, tmp_path / "changed-model.parquet", "fixture")

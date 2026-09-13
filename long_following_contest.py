# SPDX-License-Identifier: GPL-3.0-only
"""Fixed longer LightGBM replacement of v12 for retrospective PRC research.

Fitting and prediction neither select a candidate nor upload a submission.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

import nm_clock_overlay
from carrier_contest import specialist_scope
from ensemble_contest import categorical
from following_boost_contest import clock_scope as clock_scope, matrix as matrix
from neighbor_boost_contest import checked_baseline, read_object
from nm_clock_overlay import apply_clock_overlay
from ordinary_ensemble import OrdinaryConfig, parameters as ordinary_parameters
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline

VERSION = "prc2026-long-following-replacement/13.0.0"
BASELINE_VERSIONS = ["prc2026-following-boost-replacement/12.0.0", "prc2026-nm-clock-overlay/11.0.0",
    "prc2026-neighbor-boost-replacement/10.0.0", "prc2026-ordinary-ensemble/9.0.0"]
REPLACEMENT_WEIGHT = .125
RESIDUAL_CAP = 7200


class LongFollowingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-long-following-replacement/13.0.0"] = "prc2026-long-following-replacement/13.0.0"
    trees: int = Field(default=2716, ge=1, le=2716)
    data_class: Literal["competition", "synthetic"] = "competition"
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def frozen_competition_fit(self) -> Self:
        if self.data_class == "competition" and self.trees != 2716:
            raise ValueError("Competition fitting requires the frozen 2716 CPU trees")
        return self


def parameters(config: LongFollowingConfig) -> dict[str, Any]:
    params = ordinary_parameters(OrdinaryConfig(data_permission_ref=config.data_permission_ref))
    return {**params, "n_estimators": config.trees}


def replace_component(dep: pd.DataFrame, x: pd.DataFrame, baseline: pd.Series,
        new: pd.Series, old: pd.Series) -> pd.Series:
    if not dep.index.equals(x.index):
        raise ValueError("Movement and feature row alignment differs")
    for values in [baseline, new, old]:
        if not values.index.equals(x.index):
            raise ValueError("Prediction row alignment differs")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("Component predictions must be finite and nonnegative")
    protected = specialist_scope(x) | clock_scope(dep)
    result = baseline.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result.loc[~protected] = baseline.loc[~protected] + REPLACEMENT_WEIGHT * (new.loc[~protected] - old.loc[~protected])
    if not np.isfinite(result).all() or result.lt(0).any():
        raise ValueError("Replacement must be finite and nonnegative; do not clip")
    if not np.array_equal(result.loc[protected], baseline.loc[protected]):
        raise ValueError("Specialist and clock predictions must remain exact")
    return result


def train(data: Path, output: Path, config: LongFollowingConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {path.name: sha256(path) for path in paths}
    source = Path(__file__).resolve().parent
    sources = {path.name: sha256(path) for path in source.glob("*.py")}
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
    if clock.isna().any() or not clock.dt.year.eq(2025).all() or set(clock.dt.month) != set(range(1, 13)):
        raise ValueError("Complete movement clocks covering all twelve 2025 months are required")
    print("BUILDING_LONG_FOLLOWING_FEATURES", len(raw), flush=True)
    dep, x, old_columns = matrix(raw)
    del raw
    if len(x.columns) != 183 or len(old_columns) != 153:
        raise ValueError("Expected the documented 183 and 153 feature schemas")
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    fit = y.ge(0) & ~specialist_scope(x)
    if not fit.any():
        raise ValueError("Ordinary-scope fitting rows are required")
    offset = nm_schedule_fallback_baseline(x)
    cats = [str(column) for column in x.select_dtypes(include=["str", "object"]).columns]
    params = parameters(config)
    if {path.name: sha256(path) for path in paths} != hashes or any(sha256(source / name) != digest for name, digest in sources.items()):
        raise ValueError("Training input or source changed while building features")
    output.mkdir(parents=True)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sources:
        shutil.copyfile(source / name, snapshot / name)
    metadata = {"config": config.model_dump(), "parameters": params, "features": list(x),
        "categories": cats, "old_features": old_columns, "input_hashes": hashes, "source_sha256": sources,
        "departure_rows": len(dep), "fit_rows": int(fit.sum()), "negative_label_rows": int(y.lt(0).sum()),
        "capped_fit_rows": int((y-offset).loc[fit].abs().gt(RESIDUAL_CAP).sum()),
        "residual_cap": RESIDUAL_CAP, "replacement_weight": REPLACEMENT_WEIGHT,
        "clock_rule_version": nm_clock_overlay.VERSION}
    write_json(output / "inputs.json", metadata)
    x = categorical(x)
    model = lgb.LGBMRegressor(**params)
    print("FITTING_LONG_FOLLOWING", int(fit.sum()), config.trees, flush=True)
    model.fit(x.loc[fit], (y-offset).clip(-RESIDUAL_CAP, RESIDUAL_CAP).loc[fit], categorical_feature=cats)
    if model.booster_.current_iteration() != config.trees:
        raise ValueError("Fitted tree count differs from the frozen configuration")
    if {path.name: sha256(path) for path in paths} != hashes or any(
            sha256(source / name) != digest or sha256(snapshot / name) != digest for name, digest in sources.items()):
        raise ValueError("Training input or source changed during fitting")
    model.booster_.save_model(str(output / "model.txt"))
    write_json(output / "report.json", {**metadata, "status": ("FULL_FIT_NOT_OFFICIAL_SCORE"
        if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"),
        "model_sha256": sha256(output / "model.txt"), "tree_count": model.booster_.current_iteration()})
    print("LONG_FOLLOWING_READY", sha256(output / "model.txt"), flush=True)


def checked_lightgb(run: Path, report: dict[str, Any], columns: list[str], cats: list[str],
        trees: int, old: bool) -> tuple[Any, str]:
    expected = parameters(LongFollowingConfig(trees=trees, data_class="synthetic", data_permission_ref="model-check"))
    feature_key, cap_key = ("neighbor_features", "residual_fit_cap") if old else ("features", "residual_cap")
    if report.get("parameters") != expected or report.get(cap_key) != RESIDUAL_CAP:
        raise ValueError("Stored model parameters or residual cap differ")
    hashes = report.get("input_hashes")
    if not isinstance(hashes, dict) or len(hashes) != 12 or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()):
        raise ValueError("Twelve training input digests are required")
    if report.get(feature_key) != columns or report.get("categories") != cats:
        raise ValueError("Stored feature or categorical schema differs")
    if not isinstance(report.get("fit_rows"), int) or report["fit_rows"] < 1:
        raise ValueError("A completed ordinary fit is required")
    path = run / ("neighbor-model.txt" if old else "model.txt")
    digest = sha256(path)
    recorded = report.get("model_sha256")
    if old:
        recorded = recorded.get("neighbor") if isinstance(recorded, dict) else None
    if digest != recorded:
        raise ValueError("Model digest mismatch")
    model = lgb.Booster(model_file=str(path))
    if model.num_trees() != trees or model.current_iteration() != trees or model.feature_name() != columns:
        raise ValueError("Native model trees or features differ")
    if model.pandas_categorical is None or len(model.pandas_categorical) != len(cats):
        raise ValueError("Native categorical schema differs")
    required = dict(num_iterations=trees, objective="regression", num_leaves=63, learning_rate=.035,
        min_data_in_leaf=80, lambda_l2=10, feature_fraction=.9, num_threads=2,
        seed=20260907, deterministic=True, force_col_wise=True)
    if any(model.params.get(name) != value for name, value in required.items()):
        raise ValueError("Native model parameters differ")
    return model, digest


def predict(run: Path, old_run: Path, baseline_path: Path, v11_path: Path, v10_path: Path, v9_path: Path,
        ranking: Path, template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = read_object(run / "report.json")
    config = LongFollowingConfig.model_validate(report.get("config"))
    status = "FULL_FIT_NOT_OFFICIAL_SCORE" if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"
    if report.get("status") != status or report.get("replacement_weight") != REPLACEMENT_WEIGHT:
        raise ValueError("A completed fit with the fixed replacement contract is required")
    if report.get("tree_count") != config.trees or report.get("clock_rule_version") != nm_clock_overlay.VERSION:
        raise ValueError("Report tree count or clock-rule version differs")
    source, snapshot = Path(__file__).resolve().parent, run / "source-snapshot"
    sources = report.get("source_sha256")
    if not isinstance(sources, dict) or not {Path(__file__).name, "following_boost_contest.py", "nm_following_features.py", "nm_clock_overlay.py"}.issubset(sources):
        raise ValueError("Training source digests are required")
    if set(sources) != {path.name for path in snapshot.glob("*.py")}:
        raise ValueError("Training source digests and snapshot coverage differ")
    for name, digest in sources.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py") or not (source / name).is_file() or sha256(source / name) != digest or sha256(snapshot / name) != digest:
            raise ValueError("Training and inference source digests differ")
    baselines = [baseline_path, v11_path, v10_path, v9_path]
    paths = [run / "report.json", run / "model.txt", old_run / "report.json", old_run / "neighbor-model.txt",
        ranking, template_path, *baselines, *[path.with_suffix(".manifest.json") for path in baselines],
        *[source / name for name in sources], *[snapshot / name for name in sources]]
    before = {path: sha256(path) for path in paths}
    template = pd.read_parquet(template_path)
    frames, manifests = [], []
    for path, version in zip(baselines, BASELINE_VERSIONS, strict=True):
        frame, manifest = checked_baseline(path, version, before[ranking], before[template_path], template, config.data_class)
        frames.append(frame)
        manifests.append(manifest)
    v12, v11, v10, v9 = manifests
    for position, manifest in enumerate(manifests[:-1]):
        parent = baselines[position+1]
        if manifest.get("baseline_sha256") != before[parent] or manifest.get("baseline_manifest_sha256") != before[parent.with_suffix(".manifest.json")] or manifest.get("baseline_model_version") != BASELINE_VERSIONS[position+1]:
            raise ValueError("The v12/v11/v10/v9 baseline lineage is not bound to the supplied artifacts")
    if v12.get("v10_baseline_sha256") != before[v10_path] or v12.get("v10_manifest_sha256") != before[v10_path.with_suffix(".manifest.json")]:
        raise ValueError("The v12 extra v10 lineage differs")
    if any(m.get("replacement_weight") != .375 or m.get("residual_fit_cap") != RESIDUAL_CAP for m in [v12, v10]):
        raise ValueError("The existing CatBoost replacement contracts differ")
    if v12.get("old_model_sha256") != v10.get("new_model_sha256") or v12.get("old_report_sha256") != v10.get("new_report_sha256"):
        raise ValueError("The intervening CatBoost model lineage differs")
    for manifest in [v12, v11]:
        clock_sources = manifest.get("source_sha256")
        if not isinstance(clock_sources, dict) or clock_sources.get("nm_clock_overlay.py") != sources["nm_clock_overlay.py"]:
            raise ValueError("The baseline clock source differs")
    if v11.get("proxy_raw_clip") != [-604800, 604800] or v11.get("proxy_valid_inclusive_range") != [-7200, 172800] or v12.get("clock_rule_version") != nm_clock_overlay.VERSION:
        raise ValueError("The baseline clock-rule contract differs")
    old_report = read_object(old_run / "report.json")
    old_config = OrdinaryConfig.model_validate(old_report.get("config"))
    if old_report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE" or (config.data_class == "competition" and old_config.trees != 679):
        raise ValueError("The original 679-tree ordinary component is required")
    for contract in [v9, old_report]:
        if any(contract.get(key) != value for key, value in {"candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5, "residual_fit_cap": RESIDUAL_CAP}.items()):
            raise ValueError("The original ordinary component weights or cap differ")
    old_digests = v9.get("model_sha256")
    if not isinstance(old_digests, dict) or old_report.get("model_sha256") != old_digests or old_digests.get("neighbor") != before[old_run / "neighbor-model.txt"]:
        raise ValueError("Old model digest is not bound through the submitted v9 manifest")
    if old_report.get("input_hashes") != report.get("input_hashes"):
        raise ValueError("Old and new components must use the same twelve training inputs")
    dep, x, old_columns = matrix(pd.read_parquet(ranking))
    if len(x.columns) != 183 or len(old_columns) != 153 or report.get("old_features") != old_columns:
        raise ValueError("Feature schema differs from the documented replacement")
    if len(dep) != len(template) or not pd.Index(dep[ID]).is_unique or not pd.Index(dep[ID]).isin(template[ID]).all():
        raise ValueError("Ranking departures and template IDs must match exactly")
    scope, clock = specialist_scope(x), clock_scope(dep)
    if any(manifest.get("specialist_rows") != int(scope.sum()) for manifest in manifests) or v12.get("clock_rows") != int(clock.sum()):
        raise ValueError("Baseline protected-row counts differ")
    expected_v11 = apply_clock_overlay(dep.set_index(ID), frames[2].set_index(ID)[TARGET])
    if not np.array_equal(frames[1][ID].map(expected_v11), frames[1][TARGET]):
        raise ValueError("The v11 baseline differs from its original clock overlay")
    protected_ids = dep.loc[scope | clock, ID]
    protected = frames[0][ID].isin(protected_ids)
    if not np.array_equal(frames[0].loc[protected, TARGET], frames[1].loc[protected, TARGET]):
        raise ValueError("The v12 baseline changed its protected v11 values")
    cats = [str(c) for c in x.select_dtypes(include=["str", "object"]).columns]
    new_model, new_digest = checked_lightgb(run, report, [str(c) for c in x.columns], cats, config.trees, False)
    old_model, old_digest = checked_lightgb(old_run, old_report, old_columns, cats, old_config.trees, True)
    offset = nm_schedule_fallback_baseline(x)
    x = categorical(x)
    new = pd.Series(np.maximum(0, new_model.predict(x, num_threads=2)+offset.to_numpy()), index=x.index)
    old = pd.Series(np.maximum(0, old_model.predict(x.loc[:, old_columns], num_threads=2)+offset.to_numpy()), index=x.index)
    baseline = pd.Series(dep[ID].map(frames[0].set_index(ID)[TARGET]).to_numpy(), index=x.index)
    values = replace_component(dep, x, baseline, new, old)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    if {path: sha256(path) for path in paths} != before:
        raise ValueError("Source, model, baseline, ranking or template changed during prediction")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    saved = pd.read_parquet(output)
    validate_submission(template, saved)
    if not np.array_equal(saved[TARGET], result[TARGET]) or {path: sha256(path) for path in paths} != before:
        raise ValueError("Serialization or bound inputs changed")
    write_json(output.with_suffix(".manifest.json"), {"data_class": config.data_class, "data_permission_ref": permission,
        "model_version": VERSION, "submission_sha256": sha256(output), "ranking_sha256": before[ranking],
        "template_sha256": before[template_path], "baseline_sha256": before[baseline_path], "baseline_model_version": BASELINE_VERSIONS[0],
        "baseline_manifest_sha256": before[baseline_path.with_suffix(".manifest.json")],
        "baseline_chain_sha256": {version: {"submission": before[path], "manifest": before[path.with_suffix(".manifest.json")]} for version, path in zip(BASELINE_VERSIONS, baselines, strict=True)},
        "new_model_sha256": new_digest, "old_model_sha256": old_digest, "new_report_sha256": before[run / "report.json"],
        "old_report_sha256": before[old_run / "report.json"], "source_sha256": sources,
        "replacement_weight": REPLACEMENT_WEIGHT, "residual_fit_cap": RESIDUAL_CAP, "clock_rule_version": nm_clock_overlay.VERSION,
        "formula": "v12 + 0.125 * (nonnegative LightGBM183/2716 - nonnegative LightGBM153/679), outside fixed specialist and clock scopes",
        "rows": len(result), "specialist_rows": int(scope.sum()), "clock_rows": int(clock.sum()),
        "status": "PREDICTIONS_NOT_AN_OFFICIAL_SCORE" if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    fit.add_argument("--trees", type=int, default=2716)
    fit.add_argument("--data-class", choices=["competition", "synthetic"], default="competition")
    pred = commands.add_parser("predict")
    for name in ["run", "old-run", "baseline", "v11-baseline", "v10-baseline", "v9-baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, LongFollowingConfig(trees=args.trees,
            data_class=args.data_class, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.old_run, args.baseline, args.v11_baseline, args.v10_baseline, args.v9_baseline,
            args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

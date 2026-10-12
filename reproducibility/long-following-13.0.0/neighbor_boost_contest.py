# SPDX-License-Identifier: GPL-3.0-only
"""Fixed replacement of the v9 ensemble's ordinary CatBoost component.

Retrospective PRC research only. Local validation and promotion are separate
steps: training or predicting here never establishes an official improvement.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carrier_contest import load_model, specialist_scope
from ordinary_ensemble import matrix
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline

VERSION = "prc2026-neighbor-boost-replacement/10.0.0"
V9_VERSION = "prc2026-ordinary-ensemble/9.0.0"
V8_VERSION = "prc2026-arrival-boost-blend/8.0.0"
REPLACEMENT_WEIGHT = .375
RESIDUAL_CAP = 7200
PARAMETER_KEYS = ("iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count",
    "loss_function", "random_seed", "task_type", "thread_count", "allow_writing_files", "verbose")


class NeighborBoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-neighbor-boost-replacement/10.0.0"] = "prc2026-neighbor-boost-replacement/10.0.0"
    trees: int = Field(default=4999, ge=1, le=4999)
    device: Literal["CPU", "GPU"] = "GPU"
    data_class: Literal["competition", "synthetic"] = "competition"
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def frozen_competition_fit(self) -> Self:
        if self.data_class == "competition" and (self.trees != 4999 or self.device != "GPU"):
            raise ValueError("Competition fitting requires the frozen 4999 GPU trees")
        return self


def parameters(config: NeighborBoostConfig, output: Path) -> dict[str, Any]:
    return dict(iterations=config.trees, depth=9, learning_rate=.05, l2_leaf_reg=7,
        border_count=254, loss_function="RMSE", random_seed=20260907,
        task_type=config.device, thread_count=2, allow_writing_files=True,
        train_dir=str((output / "training-log").resolve()), verbose=200)


def replace_component(x: pd.DataFrame, baseline: pd.Series, new: pd.Series, old: pd.Series) -> pd.Series:
    for values in [baseline, new, old]:
        if not values.index.equals(x.index):
            raise ValueError("Prediction row alignment differs")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("Component predictions must be finite and nonnegative")
    result = baseline.copy()
    ordinary = ~specialist_scope(x)
    # Each component has the same nonnegative prediction floor as its original
    # model. The replacement itself is never clipped or otherwise adjusted.
    with np.errstate(over="ignore", invalid="ignore"):
        result.loc[ordinary] = (baseline.loc[ordinary]
            + REPLACEMENT_WEIGHT * (new.loc[ordinary] - old.loc[ordinary]))
    if not np.isfinite(result).all() or result.lt(0).any():
        raise ValueError("Replacement must be finite and nonnegative; do not clip")
    return result


def train(data: Path, output: Path, config: NeighborBoostConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {path.name: sha256(path) for path in paths}
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
    if clock.isna().any() or not clock.dt.year.eq(2025).all() or set(clock.dt.month) != set(range(1, 13)):
        raise ValueError("Complete movement clocks covering all twelve 2025 months are required")
    print("BUILDING_NEIGHBOR_BOOST_FEATURES", len(raw), flush=True)
    dep, x, old_columns = matrix(raw)
    del raw
    if len(x.columns) != 153 or len(old_columns) != 123:
        raise ValueError("Expected the documented 153 and 123 feature schemas")
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    # This mask and cap apply only to fitting; original labels are not rewritten.
    fit = y.ge(0) & ~specialist_scope(x)
    if not fit.any():
        raise ValueError("Ordinary-scope fitting rows are required")
    offset = nm_schedule_fallback_baseline(x)
    cats = [str(c) for c in x.select_dtypes(include=["str", "object"]).columns]
    params = parameters(config, output)
    output.mkdir(parents=True)
    metadata = {"config": config.model_dump(), "parameters": params, "features": list(x),
        "categories": cats, "old_features": old_columns, "input_hashes": hashes,
        "departure_rows": len(dep), "fit_rows": int(fit.sum()),
        "negative_label_rows": int(y.lt(0).sum()), "residual_cap": RESIDUAL_CAP,
        "replacement_weight": REPLACEMENT_WEIGHT}
    write_json(output / "inputs.json", metadata)
    model = CatBoostRegressor(**params)
    print("FITTING_NEIGHBOR_BOOST", int(fit.sum()), config.trees, flush=True)
    model.fit(x.loc[fit], (y - offset).clip(-RESIDUAL_CAP, RESIDUAL_CAP).loc[fit], cat_features=cats,
        save_snapshot=True, snapshot_file=str((output / "training.snapshot").resolve()), snapshot_interval=60)
    if model.tree_count_ != config.trees:
        raise ValueError("Fitted tree count differs from the frozen configuration")
    model.save_model(str(output / "model.cbm"))
    write_json(output / "report.json", {**metadata, "status": ("FULL_FIT_NOT_OFFICIAL_SCORE"
        if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"),
        "model_sha256": sha256(output / "model.cbm"), "tree_count": model.tree_count_})
    print("NEIGHBOR_BOOST_READY", sha256(output / "model.cbm"), flush=True)


def read_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def check_parameters(report: dict[str, Any], config: NeighborBoostConfig, run: Path) -> None:
    expected = parameters(config, run)
    supplied = report.get("parameters", {})
    if not isinstance(supplied, dict) or any(supplied.get(key) != expected[key] for key in PARAMETER_KEYS):
        raise ValueError("Stored model parameters differ from the fixed configuration")
    if report.get("residual_cap") != RESIDUAL_CAP or not isinstance(report.get("fit_rows"), int) or report["fit_rows"] < 1:
        raise ValueError("A completed ordinary fit with the documented residual cap is required")
    hashes = report.get("input_hashes")
    if not isinstance(hashes, dict) or len(hashes) != 12 or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()):
        raise ValueError("Twelve training input digests are required")


def checked_boost(run: Path, report: dict[str, Any], columns: list[str], config: NeighborBoostConfig) -> tuple[Any, str]:
    check_parameters(report, config, run)
    model, digest = load_model(run, columns)
    if model.tree_count_ != config.trees:
        raise ValueError("Stored tree count differs from the model report")
    actual = model.get_all_params()
    expected = parameters(config, run)
    # CatBoost serializes some floats at float32 precision; report checks above
    # remain exact while the fitted model gets a tight serialization tolerance.
    for key in ["iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "loss_function", "random_seed", "task_type"]:
        if isinstance(expected[key], float):
            same = isinstance(actual.get(key), (float, int)) and math.isclose(actual[key], expected[key], rel_tol=1e-6, abs_tol=1e-9)
        else:
            same = actual.get(key) == expected[key]
        if not same:
            raise ValueError("Fitted model parameters differ from the report")
    expected_categories = [columns[index] for index in model.get_cat_feature_indices()]
    if report.get("categories") != expected_categories:
        raise ValueError("Stored categorical feature schema differs")
    return model, digest


def checked_baseline(path: Path, version: str, ranking_hash: str, template_hash: str,
        template: pd.DataFrame, data_class: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = read_object(path.with_suffix(".manifest.json"))
    if manifest.get("model_version") != version or manifest.get("data_class") != data_class:
        raise ValueError("The documented baseline version and data class are required")
    if manifest.get("submission_sha256") != sha256(path):
        raise ValueError("Baseline digest mismatch")
    if manifest.get("ranking_sha256") != ranking_hash or manifest.get("template_sha256") != template_hash:
        raise ValueError("Baseline ranking or template digest mismatch")
    frame = pd.read_parquet(path)
    validate_submission(template, frame)
    if manifest.get("rows") != len(frame):
        raise ValueError("Baseline manifest row count differs")
    return frame, manifest


def predict(run: Path, old_run: Path, baseline_path: Path, v8_baseline_path: Path,
        ranking: Path, template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = read_object(run / "report.json")
    config = NeighborBoostConfig.model_validate(report.get("config"))
    expected_status = "FULL_FIT_NOT_OFFICIAL_SCORE" if config.data_class == "competition" else "SYNTHETIC_TEST_ONLY"
    if report.get("status") != expected_status or report.get("replacement_weight") != REPLACEMENT_WEIGHT:
        raise ValueError("A completed fit with the fixed replacement contract is required")
    if report.get("tree_count") != config.trees:
        raise ValueError("Report tree count differs from the configuration")
    ranking_hash, template_hash = sha256(ranking), sha256(template_path)
    template = pd.read_parquet(template_path)
    baseline_frame, v9 = checked_baseline(baseline_path, V9_VERSION, ranking_hash, template_hash, template, config.data_class)
    v8_frame, v8 = checked_baseline(v8_baseline_path, V8_VERSION, ranking_hash, template_hash, template, config.data_class)
    if v9.get("baseline_sha256") != sha256(v8_baseline_path) or v9.get("baseline_model_version") != V8_VERSION:
        raise ValueError("The v9 baseline is not bound to the supplied v8 artifact")
    for manifest, weights in [(v9, {"candidate_weight": .25, "neighbor_weight": .5, "airport_weight": .5}),
            (v8, {"boost_weight": .5})]:
        if manifest.get("residual_fit_cap") != RESIDUAL_CAP or any(manifest.get(key) != value for key, value in weights.items()):
            raise ValueError("Baseline component weights or residual cap differ")
    if v8.get("baseline_model_version") != "prc2026-arrival-blend/7.0.0":
        raise ValueError("The supplied v8 must preserve the documented v7 specialist")
    old_report = read_object(old_run / "report.json")
    if old_report.get("status") not in (["FULL_FIT_NOT_OFFICIAL_SCORE"] if config.data_class == "competition"
            else ["SYNTHETIC_TEST_ONLY", "FULL_FIT_NOT_OFFICIAL_SCORE"]):
        raise ValueError("The original component must be a completed full fit")
    if old_report.get("model_sha256") != v8.get("boost_model_sha256") or sha256(old_run / "model.cbm") != v8.get("boost_model_sha256"):
        raise ValueError("Old model digest is not bound through the v9 to v8 manifest chain")
    if old_report.get("input_hashes") != report.get("input_hashes"):
        raise ValueError("Old and new components must use the same twelve training inputs")
    dep, x, old_columns = matrix(pd.read_parquet(ranking))
    if len(x.columns) != 153 or len(old_columns) != 123 or report.get("old_features") != old_columns:
        raise ValueError("Feature schema differs from the documented replacement")
    if len(dep) != len(template) or not pd.Index(dep[ID]).isin(template[ID]).all():
        raise ValueError("Ranking departures and template IDs must match exactly")
    model, digest = checked_boost(run, report, [str(c) for c in x.columns], config)
    old_model, old_digest = checked_boost(old_run, old_report, old_columns, config)
    offset = nm_schedule_fallback_baseline(x)
    new = pd.Series(np.maximum(0, model.predict(x, thread_count=2) + offset), index=x.index)
    old = pd.Series(np.maximum(0, old_model.predict(x.loc[:, old_columns], thread_count=2) + offset), index=x.index)
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    scope = specialist_scope(x)
    if any(manifest.get("specialist_rows") != int(scope.sum()) for manifest in [v9, v8]):
        raise ValueError("Baseline specialist row count differs")
    v8_values = pd.Series(dep[ID].map(v8_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    if not np.array_equal(baseline.loc[scope], v8_values.loc[scope]):
        raise ValueError("The v9 baseline changed the v8 specialist predictions")
    values = replace_component(x, baseline, new, old)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    validate_submission(template, pd.read_parquet(output))
    write_json(output.with_suffix(".manifest.json"), {"data_class": config.data_class,
        "data_permission_ref": permission, "model_version": VERSION, "submission_sha256": sha256(output),
        "ranking_sha256": ranking_hash, "template_sha256": template_hash, "baseline_sha256": sha256(baseline_path),
        "baseline_model_version": V9_VERSION, "baseline_manifest_sha256": sha256(baseline_path.with_suffix(".manifest.json")),
        "v8_baseline_sha256": sha256(v8_baseline_path), "v8_manifest_sha256": sha256(v8_baseline_path.with_suffix(".manifest.json")),
        "new_model_sha256": digest, "old_model_sha256": old_digest,
        "new_report_sha256": sha256(run / "report.json"), "old_report_sha256": sha256(old_run / "report.json"),
        "replacement_weight": REPLACEMENT_WEIGHT, "residual_fit_cap": RESIDUAL_CAP,
        "formula": "v9 + 0.375 * (nonnegative CatBoost153 - nonnegative CatBoost123), ordinary scope only",
        "rows": len(result), "specialist_rows": int(scope.sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    fit.add_argument("--trees", type=int, default=4999)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    fit.add_argument("--data-class", choices=["competition", "synthetic"], default="competition")
    pred = commands.add_parser("predict")
    for name in ["run", "old-run", "baseline", "v8-baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, NeighborBoostConfig(trees=args.trees, device=args.device,
            data_class=args.data_class, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.old_run, args.baseline, args.v8_baseline, args.ranking, args.template,
            args.output, args.permission_ref)


if __name__ == "__main__":
    main()

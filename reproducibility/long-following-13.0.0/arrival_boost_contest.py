# SPDX-License-Identifier: GPL-3.0-only
"""Deeper completed-arrival residual model with a fixed v7 blend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field

from arrival_contest import matrix
from carrier_contest import load_model, specialist_scope
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class BoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-arrival-boost-blend/8.0.0"] = "prc2026-arrival-boost-blend/8.0.0"
    trees: int = Field(default=5000, ge=1, le=5000)
    device: Literal["CPU", "GPU"] = "GPU"
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")


def parameters(config: BoostConfig, output: Path) -> dict[str, Any]:
    return dict(iterations=config.trees, depth=9, learning_rate=.05, l2_leaf_reg=7,
        border_count=254, loss_function="RMSE", random_seed=20260907,
        task_type=config.device, thread_count=2, allow_writing_files=True,
        train_dir=str((output / "training-log").resolve()), verbose=200)


def blend(x: pd.DataFrame, baseline: pd.Series, candidate: pd.Series) -> pd.Series:
    if not baseline.index.equals(x.index) or not candidate.index.equals(x.index):
        raise ValueError("Prediction row alignment differs")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Predictions must be finite")
    if baseline.lt(0).any() or candidate.lt(0).any():
        raise ValueError("Predictions must be nonnegative")
    result = .5 * baseline + .5 * candidate
    scope = specialist_scope(x)
    result.loc[scope] = baseline.loc[scope]
    return result


def train(data: Path, output: Path, config: BoostConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {p.name: sha256(p) for p in paths}
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    clock = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
    if clock.isna().any() or not clock.dt.year.eq(2025).all():
        raise ValueError("Only complete 2025 movement clocks may enter training")
    print("BUILDING_ARRIVAL_BOOST_FEATURES", len(raw), flush=True)
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    fit = y.ge(0) & ~specialist_scope(x)
    if not fit.any():
        raise ValueError("Ordinary-scope fitting rows are required")
    offset = nm_schedule_fallback_baseline(x)
    cats = list(x.select_dtypes(include=["str", "object"]).columns)
    params = parameters(config, output)
    output.mkdir(parents=True)
    write_json(output / "inputs.json", {"config": config.model_dump(), "parameters": params,
        "features": list(x), "input_hashes": hashes, "fit_rows": int(fit.sum()), "residual_cap": 7200})
    model = CatBoostRegressor(**params)
    print("FITTING_ARRIVAL_BOOST", int(fit.sum()), config.trees, flush=True)
    model.fit(x.loc[fit], (y - offset).clip(-7200, 7200).loc[fit], cat_features=cats,
        save_snapshot=True, snapshot_file=str((output / "training.snapshot").resolve()), snapshot_interval=60)
    model.save_model(str(output / "model.cbm"))
    write_json(output / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "parameters": params, "features": list(x), "categories": cats,
        "fit_rows": int(fit.sum()), "input_hashes": hashes, "residual_cap": 7200,
        "model_sha256": sha256(output / "model.cbm")})
    print("ARRIVAL_BOOST_READY", sha256(output / "model.cbm"), flush=True)


def predict(run: Path, baseline_path: Path, ranking: Path, template_path: Path,
            output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE" or report.get("residual_cap") != 7200:
        raise ValueError("A completed full-data fit with the documented residual cap is required")
    config = BoostConfig(trees=report["parameters"]["iterations"],
        device=report["parameters"]["task_type"], data_permission_ref=permission)
    expected = parameters(config, run)
    for key in ["iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "loss_function", "random_seed", "task_type"]:
        if report["parameters"].get(key) != expected[key]:
            raise ValueError("Stored model parameters differ from the documented configuration")
    baseline_manifest = json.loads(baseline_path.with_suffix(".manifest.json").read_text())
    if baseline_manifest.get("model_version") != "prc2026-arrival-blend/7.0.0":
        raise ValueError("The baseline must be the documented v7 arrival ensemble")
    if baseline_manifest["submission_sha256"] != sha256(baseline_path):
        raise ValueError("Baseline digest mismatch")
    if baseline_manifest["ranking_sha256"] != sha256(ranking) or baseline_manifest["template_sha256"] != sha256(template_path):
        raise ValueError("Baseline ranking or template digest mismatch")
    template = pd.read_parquet(template_path)
    baseline_frame = pd.read_parquet(baseline_path)
    validate_submission(template, baseline_frame)
    dep, x = matrix(pd.read_parquet(ranking))
    model, digest = load_model(run, [str(c) for c in x.columns])
    if model.tree_count_ != config.trees:
        raise ValueError("Stored tree count differs from the model report")
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    candidate = pd.Series(np.maximum(0, model.predict(x, thread_count=2)
        + nm_schedule_fallback_baseline(x)), index=x.index)
    values = blend(x, baseline, candidate)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition",
        "data_permission_ref": permission, "model_version": config.model_version,
        "submission_sha256": sha256(output), "ranking_sha256": sha256(ranking),
        "template_sha256": sha256(template_path), "baseline_sha256": sha256(baseline_path),
        "baseline_model_version": baseline_manifest["model_version"], "boost_model_sha256": digest,
        "boost_weight": .5, "residual_fit_cap": 7200, "rows": len(result),
        "specialist_rows": int(specialist_scope(x).sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    fit.add_argument("--trees", type=int, default=5000)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    pred = commands.add_parser("predict")
    for name in ["run", "baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, BoostConfig(trees=args.trees, device=args.device, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.baseline, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

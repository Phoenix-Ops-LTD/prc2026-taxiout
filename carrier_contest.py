# SPDX-License-Identifier: GPL-3.0-only
"""Carrier residual model and LIRF unmatched specialist; retrospective PRC only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, Field

from contest import matrix as nm_matrix
from pipeline import ID, TARGET, TIME, rmse, sha256, validate_submission, write_json
from traffic_features import add_carrier_features, nm_schedule_fallback_baseline


class CarrierConfig(BaseModel):
    model_version: str = "prc2026-carrier-specialist/3.0.0"
    seed: int = 20260907
    device: Literal["CPU", "GPU"] = "GPU"
    global_iterations: int = Field(default=2980, ge=1)
    specialist_iterations: int = Field(default=1014, ge=1)
    data_permission_ref: str = Field(min_length=1)


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep, base = nm_matrix(raw)
    return dep, add_carrier_features(base, dep)


def specialist_scope(x: pd.DataFrame) -> pd.Series:
    return x["ADEP_mvt"].eq("LIRF") & x["missing_IOBT_flt"].eq(1)


def combine(x: pd.DataFrame, global_residual: np.ndarray, specialist_residual: np.ndarray) -> pd.Series:
    scope = specialist_scope(x)
    if len(global_residual) != len(x) or len(specialist_residual) != int(scope.sum()):
        raise ValueError("Prediction counts do not match the feature rows and specialist scope")
    residual = pd.Series(global_residual, index=x.index, dtype=float)
    residual.loc[scope] = specialist_residual
    result = (residual + nm_schedule_fallback_baseline(x)).clip(lower=0)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite predictions")
    return result


def estimator(config: CarrierConfig, specialist: bool, directory: Path) -> Any:
    return CatBoostRegressor(
        iterations=config.specialist_iterations if specialist else config.global_iterations,
        depth=5 if specialist else 9, learning_rate=.04 if specialist else .045,
        l2_leaf_reg=10 if specialist else 5, loss_function="RMSE", random_seed=config.seed,
        thread_count=2 if specialist else 4, task_type="CPU" if specialist else config.device,
        allow_writing_files=True, train_dir=str(directory.resolve()), verbose=200,
    )


def train(data: Path, output: Path, config: CarrierConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if not pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.year.eq(2025).all():
        raise ValueError("Only 2025 movements may enter training")
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Training labels must be finite")
    target = y - nm_schedule_fallback_baseline(x)
    valid = pd.to_datetime(dep[TIME], utc=True).dt.month.isin([1, 7])
    cats = list(x.select_dtypes(include=["object", "str"]).columns)
    output.mkdir(parents=True)
    predictions: dict[str, np.ndarray] = {}
    for name, scoped in [("global", False), ("specialist", True)]:
        scope = specialist_scope(x) if scoped else pd.Series(True, index=x.index)
        run = output / name
        run.mkdir()
        model = estimator(config, scoped, run / "validation-log")
        fit = ~valid & y.ge(0) & scope
        model.fit(x.loc[fit], target.loc[fit], cat_features=cats,
            save_snapshot=True, snapshot_file=str((run / "validation.snapshot").resolve()), snapshot_interval=60)
        model.save_model(str(run / "validation-model.cbm"))
        predictions[name] = np.asarray(model.predict(x.loc[valid & scope], thread_count=2))
        final = estimator(config, scoped, run / "full-log")
        final.fit(x.loc[y.ge(0) & scope], target.loc[y.ge(0) & scope], cat_features=cats,
            save_snapshot=True, snapshot_file=str((run / "full.snapshot").resolve()), snapshot_interval=60)
        final.save_model(str(run / "model.cbm"))
        write_json(run / "report.json", {"features": list(x), "categories": cats,
            "model_sha256": sha256(run / "model.cbm"), "config": config.model_dump(),
            "input_hashes": {p.name: sha256(p) for p in paths}, "fit_rows": int(fit.sum()),
            "full_fit_rows": int((y.ge(0) & scope).sum())})
    values = combine(x.loc[valid], predictions["global"], predictions["specialist"])
    result = dep.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
    result["prediction"] = values
    result.to_parquet(output / "validation.parquet", index=False)
    write_json(output / "report.json", {"status": "LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "rmse_seconds": rmse(y.loc[valid], values),
        "validation_note": "January/July 2025 holdout; hyperparameters selected on this split, not an untouched test"})


def load_model(run: Path, columns: list[str]) -> tuple[Any, str]:
    report = json.loads((run / "report.json").read_text())
    digest = sha256(run / "model.cbm")
    if report.get("model_sha256") != digest or report["features"] != columns:
        raise ValueError("Model digest or feature schema mismatch")
    model = CatBoostRegressor()
    model.load_model(str(run / "model.cbm"))
    if list(model.feature_names_) != columns:
        raise ValueError("Stored model schema mismatch")
    return model, digest


def predict(global_run: Path, specialist_run: Path, ranking: Path, template_path: Path,
            output: Path, permission: str) -> None:
    if output.exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    dep, x = matrix(pd.read_parquet(ranking))
    columns = [str(column) for column in x.columns]
    global_model, global_hash = load_model(global_run, columns)
    specialist_model, specialist_hash = load_model(specialist_run, columns)
    scope = specialist_scope(x)
    values = combine(x, np.asarray(global_model.predict(x, thread_count=4)),
        np.asarray(specialist_model.predict(x.loc[scope], thread_count=2)) if scope.any() else np.array([]))
    template = pd.read_parquet(template_path)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition", "data_permission_ref": permission,
        "model_version": "prc2026-carrier-specialist/3.0.0", "submission_sha256": sha256(output),
        "template_sha256": sha256(template_path), "ranking_sha256": sha256(ranking),
        "global_model_sha256": global_hash, "specialist_model_sha256": specialist_hash,
        "rows": len(result), "specialist_rows": int(scope.sum()), "feature_columns": list(x)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    fit.add_argument("--permission-ref", required=True)
    pred = sub.add_parser("predict")
    for name in ["global-run", "specialist-run", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, CarrierConfig(device=args.device, data_permission_ref=args.permission_ref))
    else:
        predict(args.global_run, args.specialist_run, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

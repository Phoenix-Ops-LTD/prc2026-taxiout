# SPDX-License-Identifier: GPL-3.0-only
"""Standalone retrospective PRC model. Competition data only; no product imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, Field

from pipeline import ID, TARGET, TIME, departures, rmse, sha256, validate_submission, write_json
from traffic_features import add_nm_observed_features, add_planned_features, enriched_features, nm_schedule_fallback_baseline


class Config(BaseModel):
    schema_version: str = "1.0.0"
    model_version: str = "prc2026-nm-residual/2.0.0"
    iterations: int = Field(default=2200, ge=10, le=10000)
    depth: int = Field(default=8, ge=2, le=10)
    learning_rate: float = Field(default=.055, gt=0, le=1)
    seed: int = 20260907
    device: Literal["CPU", "GPU"] = "GPU"
    validation: Literal["july", "seasonal"] = "seasonal"
    data_permission_ref: str = Field(min_length=1)


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep = departures(raw)
    return dep, add_nm_observed_features(add_planned_features(enriched_features(dep, raw), dep), dep)


def estimator(config: Config, iterations: int) -> Any:
    return CatBoostRegressor(iterations=iterations, depth=config.depth, learning_rate=config.learning_rate,
        loss_function="RMSE", random_seed=config.seed, thread_count=6, allow_writing_files=False,
        verbose=200, task_type=config.device)


def train(data: Path, output: Path, config: Config) -> None:
    if output.exists():
        raise ValueError("Use a fresh run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly training files")
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if not pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.year.eq(2025).all():
        raise ValueError("Only 2025 movements may enter fitting/validation")
    dep, x = matrix(raw)
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Targets must be finite")
    clock = pd.to_datetime(dep[TIME], utc=True)
    valid = clock.ge("2025-07-01") if config.validation == "july" else clock.dt.month.isin([1, 7])
    fit = ~valid & y.ge(0)
    offset = nm_schedule_fallback_baseline(x)
    target = y - offset
    categories = list(x.select_dtypes(include=["object", "str"]).columns)
    output.mkdir(parents=True)
    model = estimator(config, config.iterations)
    model.fit(x.loc[fit], target.loc[fit], cat_features=categories,
        eval_set=(x.loc[valid], target.loc[valid]), early_stopping_rounds=120)
    predictions = np.maximum(0, model.predict(x.loc[valid], thread_count=4) + offset.loc[valid])
    result = dep.loc[valid, [ID, "ADEP_mvt", TIME, TARGET]].copy()
    result["prediction"] = predictions
    result.to_parquet(output / "validation.parquet", index=False)
    model.save_model(str(output / "validation-model.cbm"))
    report: dict[str, Any] = {"status": "LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE", "config": config.model_dump(),
        "features": list(x), "categories": categories, "selected_iterations": int(model.tree_count_),
        "rmse_seconds": rmse(y.loc[valid], predictions), "training_rows": int(fit.sum()),
        "validation_rows": int(valid.sum()), "input_hashes": {p.name: sha256(p) for p in paths},
        "per_airport": {str(a): {"n": len(g), "rmse": rmse(g[TARGET], g.prediction)} for a, g in result.groupby("ADEP_mvt")},
        "per_month": {str(a): {"n": len(g), "rmse": rmse(g[TARGET], g.prediction)} for a, g in result.groupby(pd.to_datetime(result[TIME], utc=True).dt.month)}}
    write_json(output / "report.json", report)
    final = estimator(config, model.tree_count_)
    final.fit(x.loc[y.ge(0)], target.loc[y.ge(0)], cat_features=categories)
    final.save_model(str(output / "model.cbm"))
    report["model_sha256"] = sha256(output / "model.cbm")
    write_json(output / "report.json", report)


def predict(run: Path, ranking: Path, template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or not permission.strip():
        raise ValueError("Use a fresh output and an authorized competition-data reference")
    report = json.loads((run / "report.json").read_text())
    if report["model_sha256"] != sha256(run / "model.cbm"):
        raise ValueError("Model digest changed")
    dep, x = matrix(pd.read_parquet(ranking))
    if list(x) != report["features"]:
        raise ValueError("Feature schema mismatch")
    model = CatBoostRegressor()
    model.load_model(str(run / "model.cbm"))
    values = np.maximum(0, model.predict(x, thread_count=6) + nm_schedule_fallback_baseline(x))
    template = pd.read_parquet(template_path)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition", "data_permission_ref": permission,
        "submission_sha256": sha256(output), "template_sha256": sha256(template_path),
        "ranking_sha256": sha256(ranking), "model_sha256": report["model_sha256"], "rows": len(result),
        "validation_rmse_seconds": report["rmse_seconds"], "feature_columns": list(x)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    fit.add_argument("--validation", choices=["july", "seasonal"], default="seasonal")
    fit.add_argument("--permission-ref", required=True)
    pred = sub.add_parser("predict")
    for name in ["run", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, Config(device=args.device, validation=args.validation, data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

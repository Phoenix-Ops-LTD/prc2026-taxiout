# SPDX-License-Identifier: GPL-3.0-only
"""Fixed 75/25 CatBoost/LightGBM blend with the unchanged LIRF specialist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from carrier_contest import combine, load_model, matrix, specialist_scope
from pipeline import ID, TARGET, TIME, rmse, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class BlendConfig(BaseModel):
    model_version: str = "prc2026-carrier-blend/4.0.0"
    seed: int = 20260907
    trees: int = Field(default=679, ge=1)
    data_permission_ref: str = Field(min_length=1)


def blended_residual(cat: np.ndarray, light: np.ndarray, offset: pd.Series) -> np.ndarray:
    if len(cat) != len(light) or len(cat) != len(offset):
        raise ValueError("Prediction lengths differ")
    # Each constituent was validated with its own nonnegative prediction floor.
    return np.asarray(.75 * np.maximum(0, cat + offset) + .25 * np.maximum(0, light + offset) - offset)


def categorical(x: pd.DataFrame) -> pd.DataFrame:
    result = x.copy()
    for column in result.select_dtypes(include=["object", "str"]).columns:
        result[column] = result[column].astype("category")
    return result


def train(data: Path, output: Path, config: BlendConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if not pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.year.eq(2025).all():
        raise ValueError("Training movements must be from 2025")
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    offset = nm_schedule_fallback_baseline(x)
    x = categorical(x)
    params: dict[str, Any] = {"objective": "regression", "metric": "rmse", "num_leaves": 63,
        "learning_rate": .035, "min_child_samples": 80, "reg_lambda": 10., "colsample_bytree": .9,
        "n_jobs": 2, "verbosity": -1, "random_state": config.seed, "deterministic": True, "force_col_wise": True}
    valid = pd.to_datetime(dep[TIME], utc=True).dt.month.isin([1, 7])
    model = lgb.LGBMRegressor(n_estimators=config.trees, **params)
    model.fit(x.loc[~valid & y.ge(0)], (y-offset).loc[~valid & y.ge(0)])
    predictions = np.maximum(0, model.predict(x.loc[valid]) + offset.loc[valid])
    output.mkdir(parents=True)
    model.booster_.save_model(str(output / "validation-model.txt"))
    result = dep.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
    result["prediction"] = predictions
    result.to_parquet(output / "validation.parquet", index=False)
    final = lgb.LGBMRegressor(n_estimators=config.trees, **params)
    final.fit(x.loc[y.ge(0)], (y-offset).loc[y.ge(0)])
    final.booster_.save_model(str(output / "model.txt"))
    write_json(output / "report.json", {"status": "LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE", "config": config.model_dump(),
        "parameters": params, "features": list(x), "model_sha256": sha256(output / "model.txt"),
        "rmse_seconds": rmse(y.loc[valid], predictions), "input_hashes": {p.name: sha256(p) for p in paths}})


def predict(global_run: Path, specialist_run: Path, light_run: Path, ranking: Path,
            template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    dep, x = matrix(pd.read_parquet(ranking))
    columns = [str(c) for c in x.columns]
    cat, cat_hash = load_model(global_run, columns)
    specialist, specialist_hash = load_model(specialist_run, columns)
    report = json.loads((light_run / "report.json").read_text())
    digest = sha256(light_run / "model.txt")
    if report.get("model_sha256") != digest or report["features"] != columns:
        raise ValueError("LightGBM digest or feature schema mismatch")
    light = lgb.Booster(model_file=str(light_run / "model.txt"))
    if light.feature_name() != columns:
        raise ValueError("LightGBM stored feature schema mismatch")
    scope = specialist_scope(x)
    residual = blended_residual(np.asarray(cat.predict(x, thread_count=4)),
        np.asarray(light.predict(categorical(x), num_threads=2)), nm_schedule_fallback_baseline(x))
    values = combine(x, residual, np.asarray(specialist.predict(x.loc[scope], thread_count=2)) if scope.any() else np.array([]))
    template = pd.read_parquet(template_path)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition", "data_permission_ref": permission,
        "model_version": "prc2026-carrier-blend/4.0.0", "submission_sha256": sha256(output),
        "template_sha256": sha256(template_path), "ranking_sha256": sha256(ranking),
        "global_model_sha256": cat_hash, "specialist_model_sha256": specialist_hash, "lightgbm_sha256": digest,
        "lightgbm_weight": .25, "rows": len(result), "specialist_rows": int(scope.sum()), "feature_columns": columns})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for key in ["data", "output"]:
        fit.add_argument(f"--{key}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    pred = commands.add_parser("predict")
    for key in ["global-run", "specialist-run", "light-run", "ranking", "template", "output"]:
        pred.add_argument(f"--{key}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, BlendConfig(data_permission_ref=args.permission_ref))
    else:
        predict(args.global_run, args.specialist_run, args.light_run, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

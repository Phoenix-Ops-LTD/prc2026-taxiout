# SPDX-License-Identifier: GPL-3.0-only
"""Equal-weight seed ensemble for the existing unmatched-LIRF specialist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field, field_validator

from carrier_contest import matrix, specialist_scope
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class BaggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: str = "prc2026-specialist-bagging/6.0.0"
    additional_seeds: tuple[int, ...] = (20260908, 20260909, 20260910, 20260911)
    iterations: int = Field(default=1014, ge=1)
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")

    @field_validator("additional_seeds")
    @classmethod
    def distinct_new_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if not seeds or len(seeds) != len(set(seeds)) or 20260907 in seeds:
            raise ValueError("Use distinct additional seeds, excluding the baseline seed")
        if any(seed < 0 for seed in seeds):
            raise ValueError("Seeds must be nonnegative")
        return seeds


def average_specialist(x: pd.DataFrame, baseline: pd.Series, experts: np.ndarray) -> pd.Series:
    """Keep all other predictions exact; average independently floored experts."""
    scope = specialist_scope(x)
    if not baseline.index.equals(x.index):
        raise ValueError("Baseline prediction row alignment differs")
    if experts.ndim != 2 or experts.shape[0] < 1 or experts.shape[1] != int(scope.sum()):
        raise ValueError("Expert prediction shape differs from the specialist scope")
    if not np.isfinite(baseline).all() or not np.isfinite(experts).all():
        raise ValueError("Predictions must be finite")
    if baseline.lt(0).any() or (experts < 0).any():
        raise ValueError("Each constituent prediction must be nonnegative")
    result = baseline.copy()
    result.loc[scope] = np.mean(np.vstack([baseline.loc[scope].to_numpy(), experts]), axis=0)
    return result


def train(data: Path, output: Path, config: BaggingConfig) -> None:
    if output.exists():
        raise ValueError("Use a fresh immutable run directory")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {path.name: sha256(path) for path in paths}
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if not pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.year.eq(2025).all():
        raise ValueError("Only 2025 movements may enter training")
    dep, all_features = matrix(raw)
    del raw
    scope = specialist_scope(all_features)
    x = all_features.loc[scope].copy()
    y = dep.loc[scope, TARGET].astype(float)
    del dep, all_features
    if x.empty or not np.isfinite(y).all():
        raise ValueError("Specialist rows and finite labels are required")
    fit = y.ge(0)
    offset = nm_schedule_fallback_baseline(x)
    categories = list(x.select_dtypes(include=["object", "str"]).columns)
    output.mkdir(parents=True)
    write_json(output / "inputs.json", {"config": config.model_dump(), "input_hashes": hashes,
        "features": list(x), "fit_rows": int(fit.sum())})
    experts: list[dict[str, Any]] = []
    for seed in config.additional_seeds:
        print("FITTING_SPECIALIST", seed, int(fit.sum()), flush=True)
        model = CatBoostRegressor(iterations=config.iterations, depth=5, learning_rate=.04,
            l2_leaf_reg=10, loss_function="RMSE", random_seed=seed, task_type="CPU",
            thread_count=1, allow_writing_files=False, verbose=False)
        model.fit(x.loc[fit], (y - offset).loc[fit], cat_features=categories)
        path = output / f"seed-{seed}.cbm"
        model.save_model(str(path))
        experts.append({"seed": seed, "file": path.name, "sha256": sha256(path)})
        print("SPECIALIST_READY", seed, flush=True)
    write_json(output / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "features": list(x), "categories": categories,
        "fit_rows": int(fit.sum()), "input_hashes": hashes, "experts": experts,
        "baseline_model_version": "prc2026-duration-blend/5.0.0",
        "baseline_specialist_seed": 20260907, "equal_weight": 1 / (len(experts) + 1)})


def predict(run: Path, baseline_path: Path, ranking: Path, template_path: Path,
            output: Path, permission: str) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE":
        raise ValueError("Ranking prediction requires completed full-data models")
    config = BaggingConfig.model_validate(report["config"])
    experts = report["experts"]
    if [item["seed"] for item in experts] != list(config.additional_seeds):
        raise ValueError("Expert seeds do not match the run configuration")
    baseline_manifest = json.loads(baseline_path.with_suffix(".manifest.json").read_text())
    if baseline_manifest.get("model_version") != "prc2026-duration-blend/5.0.0":
        raise ValueError("The baseline must be the documented v5 duration ensemble")
    if baseline_manifest["submission_sha256"] != sha256(baseline_path):
        raise ValueError("Baseline digest mismatch")
    if baseline_manifest["ranking_sha256"] != sha256(ranking) or baseline_manifest["template_sha256"] != sha256(template_path):
        raise ValueError("Baseline ranking or template digest mismatch")
    template = pd.read_parquet(template_path)
    baseline_frame = pd.read_parquet(baseline_path)
    validate_submission(template, baseline_frame)
    dep, x = matrix(pd.read_parquet(ranking))
    if report["features"] != list(x):
        raise ValueError("Expert feature schema mismatch")
    scope = specialist_scope(x)
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    offset = nm_schedule_fallback_baseline(x.loc[scope])
    predictions: list[np.ndarray] = []
    for item in experts:
        expected_name = f"seed-{item['seed']}.cbm"
        if item["file"] != expected_name:
            raise ValueError("Unexpected expert model filename")
        path = run / expected_name
        if sha256(path) != item["sha256"]:
            raise ValueError("Expert model digest mismatch")
        model = CatBoostRegressor()
        model.load_model(str(path))
        if list(model.feature_names_) != list(x):
            raise ValueError("Stored expert schema mismatch")
        values = np.maximum(0, model.predict(x.loc[scope], thread_count=1) + offset) if scope.any() else np.array([])
        predictions.append(np.asarray(values))
    values = average_specialist(x, baseline, np.vstack(predictions))
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition",
        "data_permission_ref": permission, "model_version": config.model_version,
        "submission_sha256": sha256(output), "template_sha256": sha256(template_path),
        "ranking_sha256": sha256(ranking), "baseline_sha256": sha256(baseline_path),
        "baseline_model_version": baseline_manifest["model_version"], "experts": experts,
        "equal_weight": 1 / (len(experts) + 1), "rows": len(result), "specialist_rows": int(scope.sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    pred = commands.add_parser("predict")
    for name in ["run", "baseline", "ranking", "template", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, BaggingConfig(data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.baseline, args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

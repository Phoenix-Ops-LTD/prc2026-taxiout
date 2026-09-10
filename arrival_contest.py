# SPDX-License-Identifier: GPL-3.0-only
"""Completed-arrival context blended with the verified v6 ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from arrival_features import completed_arrival_features, normalized
from arrival_specialist import predict as predict_specialist
from carrier_contest import specialist_scope
from duration_contest import matrix as duration_matrix
from ensemble_contest import categorical
from pipeline import ID, TARGET, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class ArrivalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: Literal["prc2026-arrival-blend/7.0.0"] = "prc2026-arrival-blend/7.0.0"
    seed: int = Field(default=20260907, ge=0)
    trees: int = Field(default=679, ge=1)
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep, x = duration_matrix(raw)
    for feature, column in [("movement_service", "FLIGHT_mvt"), ("nm_service", "CALLSIGN_flt")]:
        x[feature] = normalized(dep[column])
        x["airport_" + feature] = x.ADEP_mvt + ":" + x[feature]
    context = completed_arrival_features(dep, raw)
    if not context.index.equals(x.index) or set(context).intersection(x):
        raise ValueError("Arrival context row or column alignment differs")
    return dep, pd.concat([x, context], axis=1)


def blend(x: pd.DataFrame, baseline: pd.Series, arrival: pd.Series) -> pd.Series:
    if not baseline.index.equals(x.index) or not arrival.index.equals(x.index):
        raise ValueError("Prediction row alignment differs")
    if not np.isfinite(baseline).all() or not np.isfinite(arrival).all():
        raise ValueError("Predictions must be finite")
    if baseline.lt(0).any() or arrival.lt(0).any():
        raise ValueError("Constituent predictions must be nonnegative")
    result = .75 * baseline + .25 * arrival
    scope = specialist_scope(x)
    result.loc[scope] = baseline.loc[scope]
    return result


def train(data: Path, output: Path, config: ArrivalConfig) -> None:
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
    print("BUILDING_ARRIVAL_FEATURES", len(raw), flush=True)
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    fit = y.ge(0)
    offset = nm_schedule_fallback_baseline(x)
    categories = list(x.select_dtypes(include=["str", "object"]).columns)
    x = categorical(x)
    parameters: dict[str, Any] = dict(objective="regression", metric="rmse", num_leaves=63,
        learning_rate=.035, min_child_samples=80, reg_lambda=10., colsample_bytree=.9,
        n_jobs=2, verbosity=-1, random_state=config.seed, deterministic=True, force_col_wise=True)
    output.mkdir(parents=True)
    write_json(output / "inputs.json", {"config": config.model_dump(), "input_hashes": hashes,
        "parameters": parameters, "features": list(x), "fit_rows": int(fit.sum())})
    model = lgb.LGBMRegressor(n_estimators=config.trees, **parameters)
    print("FITTING_ARRIVAL", int(fit.sum()), len(x.columns), flush=True)

    def progress(env: Any) -> None:
        if env.iteration % 100 == 0:
            print("ARRIVAL_TREE", env.iteration, flush=True)

    model.fit(x.loc[fit], (y - offset).loc[fit], categorical_feature=categories, callbacks=[progress])
    model.booster_.save_model(str(output / "model.txt"))
    write_json(output / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "input_hashes": hashes, "parameters": parameters,
        "features": list(x), "categories": categories, "fit_rows": int(fit.sum()),
        "model_sha256": sha256(output / "model.txt"),
        "arrival_features_source": "https://prc-data-challenge-2026.netlify.app/data.html#the-ranking-dataset"})
    print("ARRIVAL_MODEL_READY", sha256(output / "model.txt"), flush=True)


def predict(run: Path, baseline_path: Path, ranking: Path, template_path: Path,
            output: Path, permission: str, specialist_run: Path) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE":
        raise ValueError("Ranking prediction requires a completed full-data model")
    config = ArrivalConfig.model_validate(report["config"])
    baseline_manifest = json.loads(baseline_path.with_suffix(".manifest.json").read_text())
    if baseline_manifest.get("model_version") != "prc2026-specialist-bagging/6.0.0":
        raise ValueError("The baseline must be the documented v6 specialist ensemble")
    if baseline_manifest["submission_sha256"] != sha256(baseline_path):
        raise ValueError("Baseline digest mismatch")
    if baseline_manifest["ranking_sha256"] != sha256(ranking) or baseline_manifest["template_sha256"] != sha256(template_path):
        raise ValueError("Baseline ranking or template digest mismatch")
    digest = sha256(run / "model.txt")
    if report["model_sha256"] != digest:
        raise ValueError("Arrival model digest mismatch")
    template = pd.read_parquet(template_path)
    baseline_frame = pd.read_parquet(baseline_path)
    validate_submission(template, baseline_frame)
    raw = pd.read_parquet(ranking)
    dep, x = matrix(raw)
    if report["features"] != list(x):
        raise ValueError("Arrival feature schema mismatch")
    model = lgb.Booster(model_file=str(run / "model.txt"))
    if model.feature_name() != list(x):
        raise ValueError("Stored arrival feature schema mismatch")
    baseline = pd.Series(dep[ID].map(baseline_frame.set_index(ID)[TARGET]).to_numpy(), index=x.index)
    arrival = pd.Series(np.maximum(0, model.predict(categorical(x), num_threads=2)
        + nm_schedule_fallback_baseline(x)), index=x.index)
    values = blend(x, baseline, arrival)
    expert = predict_specialist(specialist_run, raw)
    scope = specialist_scope(x)
    if set(expert.index) != set(dep.loc[scope, ID]) or not expert.index.is_unique:
        raise ValueError("Arrival specialist IDs differ from the declared scope")
    if not np.isfinite(expert).all() or expert.lt(0).any():
        raise ValueError("Arrival specialist predictions must be finite and nonnegative")
    values.loc[scope] = .5 * baseline.loc[scope] + .5 * dep.loc[scope, ID].map(expert)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition",
        "data_permission_ref": permission, "model_version": config.model_version,
        "submission_sha256": sha256(output), "ranking_sha256": sha256(ranking),
        "template_sha256": sha256(template_path), "baseline_sha256": sha256(baseline_path),
        "baseline_model_version": baseline_manifest["model_version"], "arrival_model_sha256": digest,
        "arrival_weight": .25, "arrival_specialist_weight": .5,
        "arrival_specialist_report_sha256": sha256(specialist_run / "report.json"),
        "arrival_experts": json.loads((specialist_run / "report.json").read_text())["experts"],
        "rows": len(result), "specialist_rows": int(scope.sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for name in ["data", "output"]:
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--permission-ref", required=True)
    pred = commands.add_parser("predict")
    for name in ["run", "baseline", "ranking", "template", "output", "arrival-specialist"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, ArrivalConfig(data_permission_ref=args.permission_ref))
    else:
        predict(args.run, args.baseline, args.ranking, args.template, args.output, args.permission_ref, args.arrival_specialist)


if __name__ == "__main__":
    main()

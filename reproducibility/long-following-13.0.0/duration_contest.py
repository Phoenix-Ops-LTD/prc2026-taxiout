# SPDX-License-Identifier: GPL-3.0-only
"""Retrospective NM duration context, blended with the reproducible v4 baseline."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, Field

from carrier_contest import load_model, matrix as carrier_matrix, specialist_scope
from ensemble_contest import predict as predict_baseline
from pipeline import ID, TARGET, TIME, rmse, sha256, validate_submission, write_json
from traffic_features import nm_schedule_fallback_baseline


class DurationConfig(BaseModel):
    model_version: str = "prc2026-duration-blend/5.0.0"
    seed: int = 20260907
    iterations: int = Field(default=4998, ge=1)
    device: Literal["CPU", "GPU"] = "GPU"
    data_permission_ref: str = Field(min_length=1, pattern=r"\S")


def add_duration_features(base: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    if not base.index.equals(dep.index):
        raise ValueError("Departure and feature row alignment differs")
    x = base.copy()
    for label, left, right in [
        ("callsign", "FLIGHT_mvt", "CALLSIGN_flt"),
        ("aircraft", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
        ("origin", "ADEP_mvt", "ADEP_flt"),
        ("destination", "ADES_mvt", "ADES_flt"),
        ("rule", "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"),
    ]:
        a = dep[left].astype("string").str.strip().str.upper()
        b = dep[right].astype("string").str.strip().str.upper()
        x[label + "_both_present"] = (a.notna() & b.notna()).astype(float)
        x[label + "_equal"] = a.eq(b).fillna(False).astype(float)
    x["airport_route"] = x.ADEP_mvt + ":" + x.ADES_mvt
    x["airport_aircraft"] = x.ADEP_mvt + ":" + x.AIRCRAFT_TYPE_mvt
    x["airport_carrier_runway"] = x.airport_movement_carrier + ":" + x.RUNWAY_mvt
    x["nm_planned_flight_duration"] = x.movement_minus_EOBT_1_flt - x.movement_minus_ARVT_1_flt
    x["nm_actual_flight_duration"] = x.movement_minus_AOBT_3_flt - x.movement_minus_ARVT_3_flt
    x["nm_flight_duration_change"] = x.nm_actual_flight_duration - x.nm_planned_flight_duration
    x["nm_duration_ratio"] = x.nm_actual_flight_duration / x.nm_planned_flight_duration.clip(lower=120)
    x["nm_actual_duration_plausible"] = x.nm_actual_flight_duration.between(120, 100000).astype(float)
    x["nm_planned_duration_plausible"] = x.nm_planned_flight_duration.between(120, 100000).astype(float)
    x["nm_actual_offblock_schedule_delay"] = x.movement_schedule_delta_sec - x.movement_minus_AOBT_3_flt
    x["nm_estimated_offblock_schedule_delay"] = x.movement_schedule_delta_sec - x.movement_minus_EOBT_1_flt
    x["nm_initial_offblock_schedule_delay"] = x.movement_schedule_delta_sec - x.movement_minus_IOBT_flt
    x["schedule_delay_days"] = np.floor(x.movement_schedule_delta_sec / 86400)
    return x


def matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep, base = carrier_matrix(raw)
    return dep, add_duration_features(base, dep)


def blend(x: pd.DataFrame, baseline: pd.Series, duration: pd.Series) -> pd.Series:
    if not baseline.index.equals(x.index) or not duration.index.equals(x.index):
        raise ValueError("Prediction row alignment differs")
    if not np.isfinite(baseline).all() or not np.isfinite(duration).all():
        raise ValueError("Predictions must be finite")
    if baseline.lt(0).any() or duration.lt(0).any():
        raise ValueError("Constituent predictions must be nonnegative")
    result = .25 * baseline + .75 * duration
    scope = specialist_scope(x)
    result.loc[scope] = baseline.loc[scope]
    return result


def train(data: Path, output: Path, config: DurationConfig, validation_only: bool = False,
          resume: bool = False) -> None:
    if (output.exists() and not resume) or (resume and not (output / "inputs.json").is_file()):
        raise ValueError("Use a fresh run directory, or resume an existing checkpoint")
    if (output / "report.json").exists():
        raise ValueError("Completed runs cannot be overwritten")
    paths = sorted(data.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("Expected twelve authorized 2025 monthly files")
    hashes = {p.name: sha256(p) for p in paths}
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if not pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.year.eq(2025).all():
        raise ValueError("Only 2025 movements may enter training")
    print("BUILDING_FEATURES", len(raw), flush=True)
    dep, x = matrix(raw)
    del raw
    y = dep[TARGET].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Training labels must be finite")
    offset = nm_schedule_fallback_baseline(x)
    valid = pd.to_datetime(dep[TIME], utc=True).dt.month.isin([1, 7])
    fit = y.ge(0) & (~valid if validation_only else True)
    cats = list(x.select_dtypes(include=["object", "str"]).columns)
    inputs = {"config": config.model_dump(), "input_hashes": hashes,
        "features": list(x), "fit_rows": int(fit.sum()), "validation_only": validation_only}
    if resume and json.loads((output / "inputs.json").read_text()) != inputs:
        raise ValueError("Resume configuration, source hashes or feature schema changed")
    output.mkdir(parents=True, exist_ok=resume)
    if not resume:
        write_json(output / "inputs.json", inputs)
    model = CatBoostRegressor(iterations=config.iterations, depth=9, learning_rate=.05,
        l2_leaf_reg=7, border_count=254, loss_function="RMSE", random_seed=config.seed,
        task_type=config.device, thread_count=4, allow_writing_files=True,
        train_dir=str((output / "training-log").resolve()), verbose=200)
    print("FITTING", int(fit.sum()), "rows", len(x.columns), "features", flush=True)
    model.fit(x.loc[fit], (y-offset).loc[fit], cat_features=cats, save_snapshot=True,
        snapshot_file=str((output / "training.snapshot").resolve()), snapshot_interval=60)
    model.save_model(str(output / "model.cbm"))
    report: dict[str, Any] = {"status": "LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE" if validation_only else "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": config.model_dump(), "features": list(x), "categories": cats,
        "input_hashes": hashes, "fit_rows": int(fit.sum()), "model_sha256": sha256(output / "model.cbm")}
    if validation_only:
        values = np.maximum(0, model.predict(x.loc[valid], thread_count=4) + offset.loc[valid])
        result = dep.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        result["prediction"] = values
        result.to_parquet(output / "validation.parquet", index=False)
        report["rmse_seconds"] = rmse(y.loc[valid], values)
    write_json(output / "report.json", report)
    print("MODEL_READY", report["model_sha256"], flush=True)


def predict(run: Path, global_run: Path, specialist_run: Path, light_run: Path,
            ranking: Path, template_path: Path, output: Path, permission: str) -> None:
    if output.exists() or not permission.strip():
        raise ValueError("Use a fresh output and competition-data authorization reference")
    report = json.loads((run / "report.json").read_text())
    if report.get("status") != "FULL_FIT_NOT_OFFICIAL_SCORE":
        raise ValueError("Ranking prediction requires a full-data model")
    dep, x = matrix(pd.read_parquet(ranking))
    model, digest = load_model(run, list(x.columns))
    duration = pd.Series(np.maximum(0, model.predict(x, thread_count=4)
        + nm_schedule_fallback_baseline(x)), index=x.index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prc-baseline-", dir=output.parent) as folder:
        base_path = Path(folder) / "baseline.parquet"
        predict_baseline(global_run, specialist_run, light_run, ranking, template_path, base_path, permission)
        base = pd.read_parquet(base_path).set_index(ID)[TARGET]
        baseline = pd.Series(dep[ID].map(base).to_numpy(), index=x.index)
        baseline_manifest = json.loads(base_path.with_suffix(".manifest.json").read_text())
    values = blend(x, baseline, duration)
    template = pd.read_parquet(template_path)
    result = template.copy()
    result[TARGET] = result[ID].map(pd.Series(values.to_numpy(), index=dep[ID]))
    validate_submission(template, result)
    result.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"data_class": "competition",
        "data_permission_ref": permission, "model_version": "prc2026-duration-blend/5.0.0",
        "submission_sha256": sha256(output), "ranking_sha256": sha256(ranking),
        "template_sha256": sha256(template_path), "duration_model_sha256": digest,
        "baseline": baseline_manifest, "duration_weight": .75,
        "rows": len(result), "specialist_rows": int(specialist_scope(x).sum())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for key in ["data", "output"]:
        fit.add_argument(f"--{key}", type=Path, required=True)
    fit.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    fit.add_argument("--validation-only", action="store_true")
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--permission-ref", required=True)
    pred = commands.add_parser("predict")
    for key in ["run", "global-run", "specialist-run", "light-run", "ranking", "template", "output"]:
        pred.add_argument(f"--{key}", type=Path, required=True)
    pred.add_argument("--permission-ref", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, DurationConfig(device=args.device,
            data_permission_ref=args.permission_ref), args.validation_only, args.resume)
    else:
        predict(args.run, args.global_run, args.specialist_run, args.light_run,
            args.ranking, args.template, args.output, args.permission_ref)


if __name__ == "__main__":
    main()

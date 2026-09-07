# SPDX-License-Identifier: GPL-3.0-only
"""Original competition pipeline. No dependency on proprietary RALE code or data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PHASE = "PHASE_mvt"
TIME = "SCHED_TIME_UTC_mvt"
VERSION = "prc2026-original-catboost/1.1.0"
# Explicit allowlist: no target, movement IDs, actual off-block times or arrivals.
# Runway and stand are deliberately excluded from predeparture mode until as-of availability is established.
CATEGORIES = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt"]


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0.0"
    model_version: str = VERSION
    seed: int = 20260906
    iterations: int = Field(default=1000, ge=10, le=10000)
    depth: int = Field(default=7, ge=2, le=10)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    data_class: Literal["synthetic", "competition"] = "synthetic"
    validation_start: str = "2025-10-01T00:00:00Z"
    data_permission_ref: str | None = None
    feature_mode: Literal["predeparture", "challenge_context"] = "predeparture"
    negative_target_policy: Literal["error", "drop"] = "error"
    thread_count: int = Field(default=1, ge=1, le=8)
    refit_full: bool = True


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def departures(frame: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    require_columns(frame, [ID, PHASE, TIME, "ADEP_mvt"] + ([TARGET] if training else []))
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Movement IDs must be unique and non-null before filtering")
    result = frame.loc[frame[PHASE].eq("DEP")].copy()
    result[TIME] = pd.to_datetime(result[TIME], utc=True, errors="raise")
    if result.empty or result[TIME].isna().any():
        raise ValueError("Departures and valid UTC scheduled timestamps are required")
    if training:
        target = pd.to_numeric(result[TARGET], errors="raise")
        if not np.isfinite(target.to_numpy(dtype=float)).all() or (target < 0).any():
            raise ValueError("Invalid training targets; resolve with a documented cleaning rule")
        result[TARGET] = target.astype(float)
    return result


def features(frame: pd.DataFrame, mode: str = "predeparture") -> pd.DataFrame:
    require_columns(frame, [TIME, "ADEP_mvt"])
    clock = pd.to_datetime(frame[TIME], utc=True, errors="raise")
    if clock.isna().any():
        raise ValueError("Missing scheduled timestamp")
    result = pd.DataFrame(index=frame.index)
    for column in CATEGORIES:
        result[column] = frame[column].fillna("UNKNOWN").astype(str) if column in frame else "UNKNOWN"
    result["hour_utc"] = clock.dt.hour.astype(float)
    result["weekday_utc"] = clock.dt.dayofweek.astype(float)
    result["month_utc"] = clock.dt.month.astype(float)
    result["hour_sin"] = np.sin(2 * np.pi * clock.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * clock.dt.hour / 24)
    # Calendar density counts scheduled departures only, never labels or actual movements.
    # This is batch schedule context, not a claim of real-time feed availability.
    schedule = pd.DataFrame({"airport": frame["ADEP_mvt"], "bin": clock.dt.floor("30min")}, index=frame.index)
    result["scheduled_departures_30m"] = schedule.groupby(["airport", "bin"], dropna=False)["airport"].transform("size").astype(float)
    if mode == "challenge_context":
        # Organizer-supplied movement context is available in the ranking file.
        # This retrospective challenge mode must never be called a predeparture forecast.
        for column in ["RUNWAY_mvt", "STAND_mvt"]:
            require_columns(frame, [column])
            result[column] = frame[column].fillna("UNKNOWN").astype(str)
            result["airport_" + column] = result["ADEP_mvt"] + ":" + result[column]
        require_columns(frame, ["MVT_TIME_UTC_mvt"])
        movement = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
        result["movement_hour_utc"] = movement.dt.hour.astype(float)
        result["movement_weekday_utc"] = movement.dt.dayofweek.astype(float)
        result["movement_month_utc"] = movement.dt.month.astype(float)
        result["movement_schedule_delta_sec"] = (movement - clock).dt.total_seconds().clip(-86400, 172800)
        bins = pd.DataFrame({"airport": frame["ADEP_mvt"], "bin": movement.dt.floor("30min")}, index=frame.index)
        result["movement_departures_30m"] = bins.groupby(["airport", "bin"], dropna=False)["airport"].transform("size").astype(float)
    elif mode != "predeparture":
        raise ValueError("Unknown feature mode")
    return result


def rmse(actual: Any, predicted: Any) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("RMSE requires matching, finite, nonempty values")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def validate_submission(template: pd.DataFrame, prediction: pd.DataFrame) -> None:
    if list(template.columns) != [ID, TARGET] or list(prediction.columns) != [ID, TARGET]:
        raise ValueError("Submission must have exactly the two template columns in order")
    for frame in [template, prediction]:
        if frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError("Duplicate or null movement IDs")
    if len(template) == 0 or not template[ID].reset_index(drop=True).equals(prediction[ID].reset_index(drop=True)):
        raise ValueError("Prediction IDs, dtypes, count and order must match template exactly")
    if not pd.api.types.is_numeric_dtype(prediction[TARGET]):
        raise ValueError("Predicted seconds must have a numeric dtype")
    values = prediction[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite non-negative seconds")


def train(paths: list[Path], output: Path, config: RunConfig) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Use a new immutable run directory")
    if config.data_class == "competition" and not config.data_permission_ref:
        raise ValueError("Competition runs require a data permission reference")
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    data = departures(raw)
    require_columns(data, [TARGET])
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(float)
    if not np.isfinite(data[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite training target")
    negative = data[TARGET] < 0
    if negative.any() and config.negative_target_policy == "error":
        raise ValueError("Negative taxi targets require an explicit cleaning policy")
    year_clock = pd.to_datetime(data["MVT_TIME_UTC_mvt"], utc=True, errors="raise") if "MVT_TIME_UTC_mvt" in data else data[TIME]
    if config.data_class == "competition" and not year_clock.dt.year.eq(2025).all():
        raise ValueError("Official training must contain 2025 departures only")
    boundary = pd.Timestamp(config.validation_start)
    if boundary.tzinfo is None:
        raise ValueError("Validation boundary must specify UTC")
    train_rows = data[TIME] < boundary
    if not train_rows.any() or train_rows.all():
        raise ValueError("Temporal validation requires data on both sides of the boundary")
    # An NM flight must not cross the temporal validation boundary.
    if "FLIGHT_ID_mvt" in data:
        left = set(data.loc[train_rows, "FLIGHT_ID_mvt"].dropna())
        right = set(data.loc[~train_rows, "FLIGHT_ID_mvt"].dropna())
        if left & right:
            raise ValueError("Flight groups cross the temporal boundary; purge them explicitly")
    fit_rows = train_rows & ~negative
    if not fit_rows.any():
        raise ValueError("No nonnegative fitting targets remain")
    x = features(data, config.feature_mode)
    categories = CATEGORIES + (["RUNWAY_mvt", "airport_RUNWAY_mvt", "STAND_mvt", "airport_STAND_mvt"] if config.feature_mode == "challenge_context" else [])
    model = CatBoostRegressor(iterations=config.iterations, depth=config.depth, learning_rate=config.learning_rate, loss_function="RMSE", random_seed=config.seed, thread_count=config.thread_count, allow_writing_files=False, verbose=False)
    model.fit(x.loc[fit_rows], data.loc[fit_rows, TARGET], cat_features=categories, eval_set=(x.loc[~train_rows], data.loc[~train_rows, TARGET]), early_stopping_rounds=80)
    predicted = np.maximum(0, model.predict(x.loc[~train_rows]))
    airport_means = data.loc[fit_rows].groupby("ADEP_mvt")[TARGET].mean()
    baseline = data.loc[~train_rows, "ADEP_mvt"].map(airport_means).fillna(float(data.loc[fit_rows, TARGET].mean()))
    validation = data.loc[~train_rows].copy()
    validation["prediction"] = predicted
    residual = np.abs(validation[TARGET].to_numpy(dtype=float) - predicted)
    report: dict[str, Any] = {
        "config": config.model_dump(), "status": "SYNTHETIC_TEST_ONLY" if config.data_class == "synthetic" else "LOCAL_HOLDOUT_NOT_OFFICIAL_SCORE",
        "training_departures": int(fit_rows.sum()), "validation_departures": int((~train_rows).sum()),
        "negative_targets_excluded_from_fit": int((train_rows & negative).sum()), "negative_targets_retained_in_validation": int((~train_rows & negative).sum()),
        "final_refit_departures": int((~negative).sum()), "feature_mode": config.feature_mode,
        "rmse_seconds": rmse(validation[TARGET], predicted), "airport_mean_rmse_seconds": rmse(validation[TARGET], baseline),
        "per_airport": {str(a): {"n": len(g), "rmse_seconds": rmse(g[TARGET], g["prediction"])} for a, g in validation.groupby("ADEP_mvt")},
        "absolute_error_p90_seconds": float(np.quantile(residual, .9)),
        "uncertainty_status": "Holdout residual statistic only; not calibrated deployment coverage",
        "selected_iterations": max(1, int(model.tree_count_)),
        "feature_names": list(x.columns), "feature_importance": dict(zip(x.columns, model.feature_importances_.tolist(), strict=True)),
        "input_sha256": {p.name: sha256(p) for p in paths},
    }
    output.mkdir(parents=True)
    if not config.refit_full:
        report["final_refit_departures"] = 0
        report["submission_ready"] = False
        model.save_model(str(output / "validation-model.cbm"))
        write_json(output / "report.json", report)
        return report
    # Refit on all training rows with the independently selected iteration count.
    final = CatBoostRegressor(iterations=report["selected_iterations"], depth=config.depth, learning_rate=config.learning_rate, loss_function="RMSE", random_seed=config.seed, thread_count=config.thread_count, allow_writing_files=False, verbose=False)
    final.fit(x.loc[~negative], data.loc[~negative, TARGET], cat_features=categories)
    final.save_model(str(output / "model.cbm"))
    report["model_sha256"] = sha256(output / "model.cbm")
    write_json(output / "report.json", report)
    return report


def predict(run: Path, ranking_path: Path, template_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise ValueError("Refusing to overwrite a submission or its evidence")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report.get("submission_ready") is False:
        raise ValueError("Validation-only runs cannot produce submissions")
    if sha256(run / "model.cbm") != report["model_sha256"]:
        raise ValueError("Model digest mismatch")
    ranking = departures(pd.read_parquet(ranking_path))
    template = pd.read_parquet(template_path)
    require_columns(template, [ID, TARGET])
    if set(ranking[ID]) != set(template[ID]):
        raise ValueError("Ranking departure IDs must match submitting template IDs")
    model = CatBoostRegressor()
    model.load_model(str(run / "model.cbm"))
    x = features(ranking, report["config"].get("feature_mode", "predeparture"))
    if list(x.columns) != report["feature_names"]:
        raise ValueError("Feature schema changed since training")
    values = np.maximum(0, model.predict(x))
    keyed = pd.Series(values, index=ranking[ID])
    prediction = template[[ID, TARGET]].copy()
    prediction[TARGET] = template[ID].map(keyed).astype(float)
    validate_submission(template, prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(output, index=False)
    validate_submission(template, pd.read_parquet(output))
    manifest = {"schema_version": "1.0.0", "data_class": report["config"]["data_class"], "model_version": VERSION, "model_sha256": report["model_sha256"], "template_sha256": sha256(template_path), "ranking_sha256": sha256(ranking_path), "submission_sha256": sha256(output), "rows": len(prediction), "data_permission_ref": report["config"]["data_permission_ref"], "official_score": None}
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--training", type=Path, nargs="+", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True)
    prediction = commands.add_parser("predict")
    for name in ["run", "ranking", "template", "output"]:
        prediction.add_argument(f"--{name}", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--template", type=Path, required=True)
    check.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        report = train(args.training, args.output, RunConfig.model_validate_json(args.config.read_text()))
        print(json.dumps({k: report[k] for k in ["status", "rmse_seconds", "airport_mean_rmse_seconds"]}))
    elif args.command == "predict":
        print(json.dumps(predict(args.run, args.ranking, args.template, args.output)))
    else:
        validate_submission(pd.read_parquet(args.template), pd.read_parquet(args.submission))
        print("Submission schema, ID alignment and finite values: PASS")


if __name__ == "__main__":
    main()

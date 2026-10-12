# SPDX-License-Identifier: GPL-3.0-only
"""Generalization-first audit, frozen forward validation, fit and exact submissions."""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from forecast_features import CATEGORIES, EXCLUDED, SCHEMA_VERSION, clean_departures, metrics, purged_split, schedule_features
from forecast_model import FittedForecast, ModelSpec
from pipeline import ID, PHASE, TARGET, TIME, sha256, validate_submission, write_json


class ForecastPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = SCHEMA_VERSION
    seed: int = 20261009
    selection_months: tuple[int, ...] = (9, 10, 11)
    locked_month: int = 12
    diagnostic_months: tuple[int, ...] = (2, 6, 7)
    threads: int = Field(default=6, ge=1, le=16)
    data_class: str = "competition"
    permission_ref: str
    schedule_assumption: str = "Supplied published schedule fields assumed known; dataset lacks historical as-of snapshots"
    negative_policy: str = "Exclude negative labels from fit only; retain all validation labels and positive outliers"
    selection_rule: str = "Lowest pooled Sep/Oct/Nov RMSE; fixed equal pair blends; December and proxies never select"
    # No early stopping on scored months, no random CV, no leaderboard input.
    models: dict[str, ModelSpec] = Field(default_factory=lambda: {
        "mean": ModelSpec(kind="mean"), "median": ModelSpec(kind="median"),
        "airport_time": ModelSpec(kind="airport_time"),
        "lightgbm": ModelSpec(kind="lightgbm", iterations=600),
        "lightgbm_no_demand": ModelSpec(kind="lightgbm", iterations=600, demand=False),
        "catboost": ModelSpec(kind="catboost", iterations=400),
        "hist": ModelSpec(kind="hist", iterations=120),
        "airport_lightgbm": ModelSpec(kind="airport_lightgbm", iterations=400)})


def log(message: str) -> None:
    print(datetime.now(timezone.utc).isoformat() + " " + message, flush=True)


def distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame[column].fillna("UNKNOWN").astype(str).value_counts().items()}


def drift(training: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ["ADEP_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt"]:
        if column not in other or column not in training:
            result[column] = {"status": "COLUMN_ABSENT"}
            continue
        a = training[column].fillna("UNKNOWN").astype(str)
        b = other[column].fillna("UNKNOWN").astype(str)
        keys = sorted(set(a) | set(b))
        p = a.value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
        q = b.value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
        # Airport-scoped runway/stand unseen rates avoid collisions such as stand "1".
        if column in ["RUNWAY_mvt", "STAND_mvt"]:
            a = training["ADEP_mvt"].astype(str) + ":" + a
            b = other["ADEP_mvt"].astype(str) + ":" + b
        result[column] = {"unseen_rows": int((~b.isin(set(a))).sum()), "unseen_fraction": float((~b.isin(set(a))).mean()),
                          "total_variation": float(np.abs(p - q).sum() / 2),
                          "missing_fraction": float(other[column].isna().mean())}
    return result


def audit(paths: list[Path], evaluation: dict[str, tuple[Path, Path]], output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    inputs: dict[str, Any] = {}
    reference: list[tuple[str, str]] | None = None
    for path in paths:
        schema = [(f.name, str(f.type)) for f in pq.read_schema(path)]
        if reference is not None and schema != reference:
            raise ValueError(f"Training schema differs: {path.name}")
        reference = schema
        raw = pd.read_parquet(path)
        movement = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
        if not movement.dt.year.eq(2025).all():
            raise ValueError("Training includes movement outside 2025")
        month = int(path.name.split("_")[1][5:7])
        start = pd.Timestamp(year=2025, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        # Actual source files include six movements 2-4 seconds before a file boundary.
        # Preserve these records; use timestamps rather than filenames for forward splits.
        if not ((movement >= start - pd.Timedelta(seconds=5)) & (movement < end)).all():
            raise ValueError("Training movement lies outside audited five-second file-boundary tolerance")
        inputs[path.name] = {"sha256": sha256(path), "rows": len(raw), "phase": distribution(raw, PHASE),
                             "missing": {c: int(raw[c].isna().sum()) for c in raw}, "schema": schema,
                             "movement_file_boundary_rows_retained": int((movement < start).sum())}
        # NM snapshots are audited above but never retained for forecast training.
        retained = [ID, PHASE, TIME, TARGET, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
                    "FLIGHT_RULE_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "RUNWAY_mvt", "STAND_mvt"]
        frames.append(raw[[c for c in retained if c in raw]])
        log(f"AUDIT {path.name}: {len(raw)} movements")
    if len(paths) != 12:
        raise ValueError("Real final-phase runs require all twelve monthly training files")
    raw = pd.concat(frames, ignore_index=True)
    del frames
    log("AUDIT concatenate complete; validate departure identity")
    data = clean_departures(raw, training=True)
    reports: dict[str, Any] = {}
    for role, (ranking, template) in evaluation.items():
        other_schema = pq.read_schema(ranking)
        selected_columns = [str(c) for c in raw.columns if c in other_schema.names] + ["MVT_TIME_UTC_mvt"]
        other_raw = pd.read_parquet(ranking, columns=selected_columns)
        other = clean_departures(other_raw)
        submitting = pd.read_parquet(template)
        validate_submission(submitting, submitting.assign(**{TARGET: 0.0}))
        if set(submitting[ID]) != set(other[ID]):
            raise ValueError(f"{role} template IDs differ from departures")
        if set(data[ID]) & set(other[ID]):
            raise ValueError("Training and evaluation share movement IDs")
        if TARGET not in other or not other[TARGET].isna().all():
            raise ValueError("Evaluation departure targets are not completely hidden")
        expected = {1, 7} if role == "ranking" else {1, 2, 6, 7}
        movement = pd.to_datetime(other["MVT_TIME_UTC_mvt"], utc=True, errors="raise")
        if not movement.dt.year.eq(2026).all() or set(movement.dt.month) != expected:
            raise ValueError(f"Unexpected {role} year/months")
        reports[role] = {"ranking_sha256": sha256(ranking), "template_sha256": sha256(template),
                         "movements": len(other_raw), "departures": len(other), "months": sorted(expected),
                         "schema": [(f.name, str(f.type)) for f in pq.read_schema(ranking)],
                         "missing": {c: int(other[c].isna().sum()) for c in other},
                         "airports": distribution(other, "ADEP_mvt"), "aircraft": distribution(other, "AIRCRAFT_TYPE_mvt"),
                         "drift": drift(data, other),
                         "documentation_note": "Actual final covers all four months; public data page describes two additional months" if role == "final" else None}
    month_clock = pd.to_datetime(data[TIME], utc=True)
    log("AUDIT input templates complete; compute monthly drift")
    report = {"schema_version": SCHEMA_VERSION, "training": inputs, "evaluation": reports,
              "training_departures": len(data), "training_airports": distribution(data, "ADEP_mvt"),
              "training_aircraft": distribution(data, "AIRCRAFT_TYPE_mvt"),
              "target_quantiles": np.quantile(data[TARGET], [0, .01, .1, .5, .9, .99, 1]).tolist(),
              "negative_training_labels": int(data[TARGET].lt(0).sum()),
              "scheduled_years": distribution(data.assign(year=month_clock.dt.year), "year"),
              "monthly": {str(m): {"n": int(mask.sum()), "mean_target": float(data.loc[mask, TARGET].mean()),
                                    "drift_vs_other_months": drift(data.loc[~mask], data.loc[mask])}
                          for m in range(1, 13) if (mask := month_clock.dt.month.eq(m)).any()},
              "leakage": {"excluded": EXCLUDED, "rule": "All *_flt fields excluded, exact predictor allowlist",
                          "labels_in_schedule_features": False, "actual_times_in_features": False,
                          "schedule_snapshot_provenance": "UNVERIFIED_AS_OF; schedule proxy assumption documented",
                          "runway_stand_lobt": "EXCLUDED_NO_PREDEPARTURE_PROVENANCE"}}
    write_json(output / "audit.json", report)
    return raw, data, report


def benchmark(data: pd.DataFrame, x: pd.DataFrame, plan: ForecastPlan, root: Path) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    oof: list[pd.DataFrame] = []
    for month in plan.selection_months:
        fit, valid, purged = purged_split(data, month)
        evidence = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        for name, spec in plan.models.items():
            log(f"FIT selection-{month:02} {name} rows={int(fit.sum())}")
            started = time.monotonic()
            model = FittedForecast(spec.model_copy(update={"threads": plan.threads})).fit(x.loc[fit], data.loc[fit, TARGET])
            evidence[name] = model.predict(x.loc[valid])
            report = metrics(evidence[TARGET], evidence[name], evidence["ADEP_mvt"])
            ledger.append({"experiment": f"forward-{month:02}-{name}", "features": model.columns,
                           "model": spec.kind, "parameters": model.spec.model_dump(),
                           "training_seconds": time.monotonic() - started, "training_rows": int(fit.sum()),
                           "purged_flight_rows": purged, "validation_month": month, **report})
            evidence.to_parquet(root / f"selection-{month:02}.parquet", index=False)
            write_json(root / "experiment-ledger.json", ledger)
            log(f"RESULT selection-{month:02} {name} RMSE={report['rmse']:.6f}")
        oof.append(evidence)
    pooled = pd.concat(oof, ignore_index=True)
    weights: dict[str, dict[str, float]] = {n: {n: 1.0} for n in plan.models}
    learned = [n for n in ["lightgbm", "catboost", "hist", "airport_lightgbm"] if n in plan.models]
    for a, b in itertools.combinations(learned, 2):
        weights[a + "+" + b] = {a: .5, b: .5}
    scores: dict[str, Any] = {}
    for name, mixture in weights.items():
        p = sum(w * pooled[n].to_numpy() for n, w in mixture.items())
        scores[name] = {"weights": mixture, **metrics(pooled[TARGET], p, pooled["ADEP_mvt"]),
                        "per_month": {str(m): metrics(g[TARGET], sum(w * g[n].to_numpy() for n, w in mixture.items()), g["ADEP_mvt"])
                                      for m, g in pooled.groupby(pd.to_datetime(pooled[TIME], utc=True).dt.month)}}
    selected = min(scores, key=lambda n: (scores[n]["rmse"], len(weights[n]), n))
    selection = {"status": "NEW_FORWARD_PROTOCOL_ON_PREVIOUSLY_USED_DATA_NOT_UNTOUCHED_DATASET",
                 "selected": selected, "weights": weights[selected], "scores": scores,
                 "locked_month_not_read_for_selection": plan.locked_month,
                 "leaderboard_used": False}
    write_json(root / "selection.json", selection)
    pooled.to_parquet(root / "selection-oof.parquet", index=False)
    log(f"SELECTED {selected} pooled RMSE={scores[selected]['rmse']:.6f}")
    return weights[selected], selection, ledger


def selected_fold(data: pd.DataFrame, x: pd.DataFrame, plan: ForecastPlan, weights: dict[str, float], month: int, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit, valid, purged = purged_split(data, month)
    result = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
    prediction = np.zeros(int(valid.sum()))
    for name, weight in weights.items():
        log(f"FIT fixed-evaluation-{month:02} {name}")
        model = FittedForecast(plan.models[name].model_copy(update={"threads": plan.threads})).fit(x.loc[fit], data.loc[fit, TARGET])
        prediction += weight * model.predict(x.loc[valid])
    result["prediction"] = prediction
    result.to_parquet(root / f"fixed-evaluation-{month:02}.parquet", index=False)
    report = {"month": month, "training_rows": int(fit.sum()), "purged_flight_rows": purged,
              **metrics(result[TARGET], prediction, result["ADEP_mvt"])}
    log(f"RESULT fixed-evaluation-{month:02} RMSE={report['rmse']:.6f}")
    return result, report


def make_submission(run: Path, ranking: Path, template: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Submission output must be new")
    manifest = json.loads((run / "model-manifest.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(ranking)
    data = clean_departures(raw)
    x = schedule_features(data, raw)
    values = np.zeros(len(data))
    for name, weight in manifest["weights"].items():
        values += weight * FittedForecast.load(run / name).predict(x)
    submitting = pd.read_parquet(template)
    if set(submitting[ID]) != set(data[ID]):
        raise ValueError("Ranking departure IDs and template differ")
    submitting[TARGET] = submitting[ID].map(pd.Series(values, index=data[ID])).astype(float)
    validate_submission(pd.read_parquet(template), submitting)
    output.parent.mkdir(parents=True, exist_ok=True)
    submitting.to_parquet(output, index=False)
    validate_submission(pd.read_parquet(template), pd.read_parquet(output))
    receipt = {"schema_version": SCHEMA_VERSION, "rows": len(submitting), "submission_sha256": sha256(output),
               "ranking_sha256": sha256(ranking), "template_sha256": sha256(template),
               "model_manifest_sha256": sha256(run / "model-manifest.json"),
               "data_class": manifest["data_class"], "data_permission_ref": manifest["permission_ref"],
               "official_score": None, "status": "LOCAL_VALIDATED_NOT_UPLOADED"}
    write_json(output.with_suffix(".manifest.json"), receipt)
    return receipt


def run(data_dir: Path, root: Path, plan: ForecastPlan, final_data_dir: Path | None = None) -> dict[str, Any]:
    if root.exists():
        raise ValueError("Use a new immutable run directory")
    if not plan.permission_ref or plan.data_class != "competition":
        raise ValueError("Real data requires competition permission reference")
    if plan.selection_months != (9, 10, 11) or plan.locked_month != 12 or plan.diagnostic_months != (2, 6, 7):
        raise ValueError("Changing the frozen validation protocol requires a new version")
    root.mkdir(parents=True)
    write_json(root / "plan.json", plan.model_dump())
    write_json(root / "environment.json", {"python": platform.python_version(), "packages": {
        n: importlib.metadata.version(n) for n in ["numpy", "pandas", "pyarrow", "catboost", "lightgbm", "scikit-learn", "pydantic"]},
        "source_hashes": {n: sha256(Path(__file__).parent / n) for n in ["forecast_run.py", "forecast_model.py", "forecast_features.py", "pipeline.py"]}})
    paths = sorted(data_dir.glob("training_*.parquet"))
    evaluation = {"ranking": (data_dir / "ranking.parquet", data_dir / "submitting.parquet")}
    final_root = final_data_dir or data_dir
    if (final_root / "final_ranking.parquet").exists() and (final_root / "final_submitting.parquet").exists():
        evaluation["final"] = (final_root / "final_ranking.parquet", final_root / "final_submitting.parquet")
    raw, data, audit_report = audit(paths, evaluation, root)
    log("BUILD schedule-only features")
    x = schedule_features(data, raw)
    x.to_parquet(root / "features.parquet", index=False)
    write_json(root / "feature-schema.json", {"schema_version": SCHEMA_VERSION, "features": list(x), "categories": CATEGORIES,
                                             "excluded": EXCLUDED, "schedule_assumption": plan.schedule_assumption})
    del raw
    weights, selection, ledger = benchmark(data, x, plan, root)
    # Selection is frozen before the final holdout and seasonal diagnostics are fitted/scored.
    locked, locked_report = selected_fold(data, x, plan, weights, plan.locked_month, root)
    diagnostics = {}
    for month in plan.diagnostic_months:
        _, report = selected_fold(data, x, plan, weights, month, root)
        diagnostics[str(month)] = report
        write_json(root / "diagnostics.json", diagnostics)
    # Complete-airport/time holdouts: busiest and least represented airport, based on fit counts only.
    fit, valid, _ = purged_split(data, 11)
    counts = data.loc[fit, "ADEP_mvt"].value_counts()
    stress = {}
    for airport in sorted({str(counts.index[0]), str(counts.index[-1])}):
        held = valid & data["ADEP_mvt"].eq(airport).to_numpy()
        fit_other = fit & ~data["ADEP_mvt"].eq(airport).to_numpy()
        if not held.any():
            continue
        predicted = np.zeros(int(held.sum()))
        for name, weight in weights.items():
            log(f"FIT held-airport-{airport} {name}")
            m = FittedForecast(plan.models[name].model_copy(update={"threads": plan.threads})).fit(x.loc[fit_other], data.loc[fit_other, TARGET])
            predicted += weight * m.predict(x.loc[held])
        stress[airport] = {"status": "COMPLETE_AIRPORT_NOVEMBER_HOLDOUT_GLOBAL_FALLBACK", **metrics(data.loc[held, TARGET], predicted, data.loc[held, "ADEP_mvt"])}
    write_json(root / "airport-stress.json", stress)
    final_fit = data[TARGET].ge(0)
    model_root = root / "model"
    model_root.mkdir()
    for name in weights:
        log(f"FINAL FIT {name} rows={int(final_fit.sum())}")
        started = time.monotonic()
        model = FittedForecast(plan.models[name].model_copy(update={"threads": plan.threads})).fit(x.loc[final_fit], data.loc[final_fit, TARGET])
        model.save(model_root / name)
        # Actual saved/native predictions must agree on all locked-month feature rows.
        original = model.predict(x.loc[locked.index])
        loaded = FittedForecast.load(model_root / name).predict(x.loc[locked.index])
        if not np.array_equal(original, loaded):
            raise ValueError("Saved/native inference parity failed")
        ledger.append({"experiment": "full-fit-" + name, "parameters": model.spec.model_dump(),
                       "features": model.columns, "training_seconds": time.monotonic() - started,
                       "training_rows": int(final_fit.sum()), "native_reload_exact_rows": len(locked)})
        write_json(root / "experiment-ledger.json", ledger)
    radius = float(np.quantile(np.abs(locked[TARGET] - locked["prediction"]), .9))
    write_json(model_root / "model-manifest.json", {"model_version": "prc2026-forecast/1.0.0", "schema_version": SCHEMA_VERSION,
        "weights": weights, "data_class": plan.data_class, "permission_ref": plan.permission_ref,
        "interval_radius_seconds": radius, "interval_status": "EMPIRICAL_DECEMBER_RESIDUAL_90_PERCENT_NOT_GUARANTEED_FUTURE_COVERAGE",
        "product_use": "REQUIRES_SEPARATELY_LICENSED_TRAINING_DATA_OR_EXPLICIT_MODEL_USE_RIGHTS"})
    submissions = {role: make_submission(model_root, ranking, template, root / "submissions" / ("final_submission.parquet" if role == "final" else "ranking_submission.parquet"))
                   for role, (ranking, template) in evaluation.items()}
    best = selection["scores"][selection["selected"]]
    worst = max(locked_report["per_airport"], key=lambda a: locked_report["per_airport"][a]["rmse"])
    final = {"BEST_CV_RMSE": best["rmse"], "LOCKED_DECEMBER_RMSE": locked_report["rmse"],
             "JAN_PROXY_RMSE": locked_report["rmse"], "JAN_PROXY_DEFINITION": "December forward winter proxy, not January calendar validation",
             "FEB_PROXY_RMSE": diagnostics["2"]["rmse"], "JUN_PROXY_RMSE": diagnostics["6"]["rmse"],
             "JUL_PROXY_RMSE": diagnostics["7"]["rmse"], "WORST_AIRPORT": worst,
             "BEST_MODEL": selection["selected"], "ENSEMBLE": weights,
             "LEAKAGE_CHECK": "ALLOWLIST_AND_PURGED_FORWARD_PASS; HISTORICAL_SCHEDULE_ASOF_UNVERIFIED",
             "FINAL_SUBMISSION": str(root / "submissions" / "final_submission.parquet") if "final" in submissions else "MISSING_FINAL_INPUTS",
             "REPRODUCIBLE": "FROZEN_PLAN_INPUT_SOURCE_ENVIRONMENT_MODEL_HASHES_AND_NATIVE_PARITY",
             "PUBLIC_REPO_READY": "SOURCE_EXPORT_PENDING_SECRET_SCAN_AND_CHECKS",
             "PRODUCT_MODEL_READY": "INTERFACE_LOCAL_ONLY; COMPETITION_WEIGHTS_RESTRICTED",
             "training_departures": audit_report["training_departures"], "selection": best,
             "locked": locked_report, "diagnostics": diagnostics, "submissions": submissions}
    write_json(root / "final-report.json", final)
    log("COMPLETE " + json.dumps({k: v for k, v in final.items() if k.isupper()}))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--final-data", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--permission-ref", required=True)
    train.add_argument("--threads", type=int, default=6)
    predict = commands.add_parser("predict")
    for name in ["run", "ranking", "template", "output"]:
        predict.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.data, args.output, ForecastPlan(permission_ref=args.permission_ref, threads=args.threads), args.final_data)
    else:
        print(json.dumps(make_submission(args.run, args.ranking, args.template, args.output)))


if __name__ == "__main__":
    main()

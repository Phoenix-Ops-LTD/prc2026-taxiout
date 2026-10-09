# SPDX-License-Identifier: GPL-3.0-only
"""Continuous CV board plus completed locked diagnostics and refit receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from forecast_features import clean_departures, purged_split
from forward_leaderboard import atomic_write, format_json, load_json, render, score, snapshot
from pipeline import ID, PHASE, TARGET, TIME, sha256


def diagnostic_cohorts(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    aircraft = data["AIRCRAFT_TYPE_mvt"].fillna("UNKNOWN").astype(str)
    airport = data["ADEP_mvt"].fillna("UNKNOWN").astype(str)
    stand = airport + ":" + data["STAND_mvt"].fillna("UNKNOWN").astype(str)
    for month in [2, 6, 7, 12]:
        fit, valid, _ = purged_split(data, month)
        frame = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        frame["unseen_aircraft"] = ~aircraft.loc[valid].isin(set(aircraft.loc[fit]))
        frame["unseen_airport_stand"] = (~airport.loc[valid].isin(set(airport.loc[fit]))
            | (data.loc[valid, "STAND_mvt"].notna() & ~stand.loc[valid].isin(set(stand.loc[fit & data["STAND_mvt"].notna().to_numpy()]))))
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def diagnostics(run: Path, context: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    selection_path = run / "selection.json"
    if not selection_path.exists():
        return results
    selection = load_json(selection_path)
    provenance = load_json(run / "leaderboard-provenance.json")
    bindings = load_json(run / "environment.json")["source_hashes"]
    common = {"model": selection["selected"], "device": "CPU", "rank": None, "champion": False,
        **provenance, "source_hashes": bindings, "training_time_seconds": None,
        "inference_time_seconds": None, "combined_time_seconds": None,
        "timing_status": "SEPARATE_PHASE_TIMERS_NOT_RECORDED",
        "status": "COMPLETED_DIAGNOSTIC_EXCLUDED_FROM_SELECTION", "used_for_model_selection": False}
    for month in [2, 6, 7, 12]:
        path = run / f"fixed-evaluation-{month:02}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        expected = context.loc[pd.to_datetime(context[TIME], utc=True).dt.month.eq(month)]
        keys = [ID, TARGET, TIME, "ADEP_mvt"]
        if not frame[keys].reset_index(drop=True).equals(expected[keys].reset_index(drop=True)):
            raise ValueError("Diagnostic prediction identities or labels changed")
        frame = frame.merge(expected[[ID, "unseen_aircraft", "unseen_airport_stand"]], on=ID, validate="one_to_one")
        results.append({**common, **score(frame, "prediction"), "experiment_id": run.name + f"/locked-diagnostic-{month:02}",
            "validation_folds": [f"2025-{month:02}"], "timestamp": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "timestamp_provenance": "PREDICTION_FILE_COMPLETION_MTIME", "prediction_sha256": sha256(path)})
    stress_path = run / "airport-stress.json"
    if stress_path.exists():
        for airport, metrics in load_json(stress_path).items():
            results.append({**common, "experiment_id": run.name + "/airport-stress-" + airport,
                "validation_folds": ["2025-11/held-airport-" + airport], "mean_mae": metrics["mae"],
                "median_mae": metrics["mae"], "worst_fold_mae": metrics["mae"], "worst_decile_mae": metrics.get("worst_decile_mae"),
                "unseen_aircraft_mae": metrics.get("unseen_aircraft_mae"), "unseen_airport_stand_mae": metrics["mae"],
                "catastrophic_error_rate": metrics.get("catastrophic_error_rate"), "validation_rows": metrics["n"], "rmse": metrics["rmse"],
                "missing_metric_reason": None if "worst_decile_mae" in metrics else "Stress phase retained aggregate RMSE/MAE; row-level predictions and requested tail/cohort metrics were not saved",
                "stress_status": metrics.get("status"),
                "timestamp": datetime.fromtimestamp(stress_path.stat().st_mtime, timezone.utc).isoformat()})
    return results


def cv_signature(runs: list[Path], selections: list[Path]) -> str:
    parts: list[Any] = []
    for run in runs:
        ledger = run / "experiment-ledger.json"
        if ledger.exists():
            parts.append([str(run), [e for e in load_json(ledger) if e.get("experiment", "").startswith("forward-")]])
        parts += [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in run.glob("selection-*.parquet")]
    for run in selections:
        parts += [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in [run / "selection.json", run / "selection-oof.parquet"] if p.exists()]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def discover_runs(root: Path, allowed: set[str]) -> list[Path]:
    """Discover comparable forecast runs; incompatible/official protocols stay out."""
    found = []
    for plan_path in sorted(root.glob("*/plan.json")):
        run = plan_path.parent
        ledger_path = run / "experiment-ledger.json"
        if not ledger_path.exists():
            continue
        try:
            plan = load_json(plan_path)
            if plan.get("selection_months", plan.get("months")) != [9, 10, 11]:
                continue
            models = plan.get("models", {})
            if not models or not all(isinstance(v, dict) and "kind" in v for v in models.values()):
                continue
            records = [e for e in load_json(ledger_path) if e.get("experiment", "").startswith("forward-")]
            if records and all(e.get("features") and set(e["features"]).issubset(allowed) for e in records):
                found.append(run.resolve())
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return found


def measured_execution(report: dict[str, Any], runs: list[Path]) -> None:
    """Carry real GPU identity and separate clocks without rewriting legacy evidence."""
    for run in runs:
        plan = load_json(run / "plan.json")
        if plan.get("execution_unit") != "GPU":
            continue
        ledger_path = run / "experiment-ledger.json"
        ledger = load_json(ledger_path) if ledger_path.exists() else []
        for entry in report["entries"] + report["completed_fold_experiments"]:
            if not entry["experiment_id"].startswith(run.name + "/"):
                continue
            model = entry["model"]
            records = [r for r in ledger if r.get("experiment", "").startswith("forward-")
                       and r.get("model") == model and f"2025-{r['validation_month']:02}" in entry["validation_folds"]]
            entry["device"] = "GPU RTX4090 / CPU inference"
            entry["parameters"] = {**entry.get("parameters", {}), **plan.get("training_overrides", {}).get(model, {})}
            if records:
                entry["training_time_seconds"] = sum(r["training_time_seconds"] for r in records)
                entry["inference_time_seconds"] = sum(r["inference_time_seconds"] for r in records)
                entry["timing_status"] = "MEASURED_SEPARATELY_SUM_OF_COMPLETED_FOLDS"
                entry["timestamp"] = max(r["completed_at_utc"] for r in records)
    singles = {e["model"]: e for e in report["entries"] if e.get("status") == "COMPLETED_VALIDATED" and e.get("model_family") != "fixed_equal_ensemble"}
    for entry in report["entries"]:
        weights = entry.get("parameters", {}).get("weights", {})
        if len(weights) < 2:
            continue
        components = [singles[n] for n in weights if n in singles]
        if len(components) != len(weights) or not any("GPU" in c["device"] for c in components):
            continue
        mixed = any(c["device"] == "CPU" for c in components)
        entry["device"] = "GPU RTX4090 + CPU training / CPU inference" if mixed else "GPU RTX4090 / CPU inference"
        for field in ["training_time_seconds", "inference_time_seconds"]:
            entry[field] = sum(c[field] for c in components) if all(c[field] is not None for c in components) else None
        entry["timing_status"] = "SUM_OF_MEASURED_COMPONENT_TIMES; blend composition unmeasured" if not mixed else "SEPARATE_TIMERS_UNAVAILABLE_FOR_LEGACY_CPU_COMPONENT; no retraining for fixed blend"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--selections", type=Path, nargs="*", default=[])
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--diagnostic-sources", type=Path, nargs="*", default=[])
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-discover", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("Polling interval must be positive")
    source = Path(__file__).parent
    args.runs = [p.resolve() for p in args.runs]
    args.selections = [p.resolve() for p in args.selections]
    args.final = args.final.resolve()
    diagnostic_sources = [p.resolve() for p in args.diagnostic_sources]
    if args.final not in diagnostic_sources:
        diagnostic_sources.append(args.final)
    allowed_features = set(pq.read_schema(args.runs[0] / "features.parquet").names)
    cv_context = pd.read_parquet(source / "runs" / "leaderboard-cohorts.parquet")
    audit = load_json(args.runs[0] / "audit.json")
    paths = sorted(args.data.glob("training_*.parquet"))
    if len(paths) != 12 or any(sha256(p) != audit["training"][p.name]["sha256"] for p in paths):
        raise ValueError("Diagnostic inputs differ from audit")
    cache = source / "runs" / "leaderboard-diagnostic-cohorts.parquet"
    if not cache.exists():
        columns = [ID, PHASE, TARGET, TIME, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt", "STAND_mvt"]
        data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
        diagnostic_cohorts(data).to_parquet(cache, index=False)
        atomic_write(cache.with_suffix(".json"), json.dumps({"sha256": sha256(cache), "audit_sha256": sha256(args.runs[0] / "audit.json")}))
        del data
    cache_binding = load_json(cache.with_suffix(".json"))
    if cache_binding["sha256"] != sha256(cache) or cache_binding["audit_sha256"] != sha256(args.runs[0] / "audit.json"):
        raise ValueError("Diagnostic cohort cache changed")
    context = pd.read_parquet(cache)
    commit = load_json(args.final / "leaderboard-provenance.json")["git_commit"]
    previous_cv = ""
    previous_diagnostics: list[Any] | None = None
    report: dict[str, Any] = {}
    while True:
        try:
            if not args.no_discover:
                for run in discover_runs(source / "runs", allowed_features):
                    if run not in args.runs:
                        args.runs.append(run)
                    if (run / "selection.json").exists() and (run / "selection-oof.parquet").exists() and run not in args.selections:
                        args.selections.append(run)
            current_cv = cv_signature(args.runs, args.selections)
            files = [p for r in diagnostic_sources for p in [*r.glob("fixed-evaluation-*.parquet"), r / "airport-stress.json", r / "experiment-ledger.json"]]
            current_diagnostics = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()]
            if current_cv != previous_cv or current_diagnostics != previous_diagnostics:
                if current_cv != previous_cv:
                    report = snapshot(args.runs, cv_context, source, commit, args.selections)
                    measured_execution(report, args.runs)
                diagnostic_rows: list[dict[str, Any]] = []
                seen_predictions: set[str] = set()
                for diagnostic_source in diagnostic_sources:
                    for entry in diagnostics(diagnostic_source, context):
                        digest = entry.get("prediction_sha256")
                        if digest is not None and digest in seen_predictions:
                            continue
                        if digest is not None:
                            seen_predictions.add(digest)
                        diagnostic_rows.append(entry)
                report["diagnostic_experiments"] = diagnostic_rows
                ledger_path = args.final / "experiment-ledger.json"
                ledger = load_json(ledger_path) if ledger_path.exists() else []
                report["completed_refits"] = [{**e, "rank": None, "champion": False,
                    "experiment_id": args.final.name + "/" + e["experiment"],
                    "model": e["experiment"].removeprefix("selected-final-full-fit-"),
                    "status": "COMPLETED_REFIT_NO_NEW_VALIDATION", "used_for_model_selection": False,
                    "mean_mae": None, "median_mae": None, "worst_fold_mae": None, "worst_decile_mae": None,
                    "unseen_aircraft_mae": None, "unseen_airport_stand_mae": None, "catastrophic_error_rate": None,
                    "training_time_seconds": e.get("training_time_seconds"), "inference_time_seconds": e.get("inference_time_seconds"),
                    "timing_status": "MEASURED_SEPARATELY" if "training_time_seconds" in e else "COMBINED_FIT_SAVE_RELOAD_PARITY_TIMER", "combined_time_seconds": e["training_seconds"],
                    "validation_folds": [], "device": "CPU", "git_commit": commit,
                    "missing_metric_reason": "Deployment refit has no new held-out labels; its complete forward experiment remains ranked separately",
                    "timestamp": e.get("completed_at_utc", datetime.fromtimestamp((args.final / "experiment-ledger.json").stat().st_mtime, timezone.utc).isoformat())}
                    for e in ledger if e.get("experiment", "").startswith("selected-final-full-fit-")]
                report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                report["monitor_source_sha256"] = sha256(Path(__file__))
                atomic_write(source / "leaderboard.json", format_json(report, source / "leaderboard.json", source))
                markdown = render(report)
                markdown += "\n## Completed diagnostics (excluded from ranking and selection)\n\n"
                for e in report["diagnostic_experiments"]:
                    markdown += f"- {e['experiment_id']}: MAE {e['mean_mae']:.3f}s; completed, unranked.\n"
                markdown += "\nDiagnostic and refit records, including unavailable metric reasons, are retained in `leaderboard.json`.\n"
                markdown += "\nContinuous full monitoring: `python leaderboard_monitor.py --data PATH --runs BASE SEARCH GUARD_V2 FINAL --selections BASE FINAL --final FINAL`.\n"
                atomic_write(source / "LEADERBOARD.md", markdown)
                previous_cv, previous_diagnostics = current_cv, current_diagnostics
                print(report["updated_at_utc"], "UPDATED", len(report["entries"]), "CV entries", len(report["diagnostic_experiments"]), "diagnostics", flush=True)
        except (ValueError, OSError, KeyError) as exc:
            print("RETRY_INCOMPLETE_WRITE", str(exc), flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

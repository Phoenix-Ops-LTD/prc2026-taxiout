# SPDX-License-Identifier: GPL-3.0-only
"""Complete a frozen selection using its exact unknown-airport fallback rule."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_features import clean_departures, metrics, purged_split
from forecast_model import ModelSpec
from forecast_novel_guard import build_model, load_model
from forecast_novel_finalize import make_submission
from forecast_run import log
from forward_leaderboard import score
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json


def airport_holdouts(data: pd.DataFrame, specs: dict[str, ModelSpec], weights: dict[str, float], output: Path) -> dict[str, Any]:
    """Exact: every selected learner overwrites an unseen airport with fit mean."""
    if not all(specs[n].kind in ["catboost", "lightgbm", "hist", "airport_lightgbm"] for n in weights):
        raise ValueError("Analytic fallback is restricted to the verified learned families")
    fit, valid, purged = purged_split(data, 11)
    counts = data.loc[fit, "ADEP_mvt"].value_counts()
    reports: dict[str, Any] = {}
    for airport in sorted({str(counts.index[0]), str(counts.index[-1])}):
        if airport == "UNKNOWN":
            raise ValueError("Reserved missing category is not an unseen airport")
        held = valid & data["ADEP_mvt"].eq(airport).to_numpy()
        other = fit & ~data["ADEP_mvt"].eq(airport).to_numpy()
        frame = data.loc[held, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        prediction = np.zeros(int(held.sum()))
        mean = float(data.loc[other, TARGET].mean())
        for weight in weights.values():
            prediction += weight * mean
        aircraft = data["AIRCRAFT_TYPE_mvt"].fillna("UNKNOWN").astype(str)
        frame["unseen_aircraft"] = ~aircraft.loc[held].isin(set(aircraft.loc[other]))
        frame["unseen_airport_stand"] = True
        frame["prediction"] = prediction
        frame.to_parquet(output / f"airport-stress-{airport}.parquet", index=False)
        reports[airport] = {"status": "EXACT_ANALYTIC_UNKNOWN_AIRPORT_GLOBAL_MEAN_RULE_NO_REDUNDANT_NATIVE_FIT",
            "training_rows": int(other.sum()), "purged_flight_rows": purged, "fit_global_mean_seconds": mean,
            "prediction_sha256": sha256(output / f"airport-stress-{airport}.parquet"),
            **metrics(frame[TARGET], prediction, frame["ADEP_mvt"]), **score(frame, "prediction")}
    write_json(output / "airport-stress.json", reports)
    return reports


def finish(source: Path, base: Path, search: Path, data_root: Path, final_root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new completion directory")
    output.mkdir(parents=True)
    selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    weights = selection["weights"]
    specs = {n: ModelSpec.model_validate(s) for n, s in plan["models"].items()}
    environment = json.loads((source / "environment.json").read_text(encoding="utf-8"))
    for name, digest in environment["source_hashes"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen prediction/validation source changed")
    environment["source_hashes"][Path(__file__).name] = sha256(Path(__file__))
    write_json(output / "environment.json", environment)
    write_json(output / "plan.json", {"phase": "FIXED_SELECTION_COMPLETION_NO_RETUNING", "models": {n: specs[n].model_dump() for n in weights},
        "weights": weights, "source_plan_sha256": sha256(source / "plan.json"),
        "selection_sha256": sha256(source / "selection.json"), "selected_at_utc": selection["selected_at_utc"],
        "completion_plan_at_utc": datetime.now(timezone.utc).isoformat(),
        "airport_rule": "Exact weighted fit global mean; all selected model branches overwrite unseen-airport predictions",
        "official_feedback_used": False, "model_selection_changed": False})
    for name in ["selection.json", "audit.json", "feature-schema.json", "leaderboard-provenance.json"]:
        shutil.copyfile(source / name, output / name)
    log("WAIT_FOR_FIXED_SEASONAL_DIAGNOSTICS_BEFORE_FULL_FIT")
    before_wait = time.monotonic()
    while True:
        try:
            diagnostic_report = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
            if set(diagnostic_report) != {"2", "6", "7"} or not all((source / f"fixed-evaluation-{m:02}.parquet").exists() for m in [2, 6, 7, 12]):
                raise FileNotFoundError
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() - before_wait > 21600:
                raise TimeoutError("Fixed seasonal diagnostics are not complete")
            time.sleep(5)
    for name in ["diagnostics.json", "fixed-evaluation-02.parquet", "fixed-evaluation-06.parquet", "fixed-evaluation-07.parquet", "fixed-evaluation-12.parquet"]:
        shutil.copyfile(source / name, output / name)
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    paths = sorted(data_root.glob("training_*.parquet"))
    if len(paths) != 12 or any(sha256(p) != audit["training"][p.name]["sha256"] for p in paths):
        raise ValueError("Training inputs changed")
    columns = [ID, PHASE, TARGET, TIME, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt", "STAND_mvt"]
    data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
    x = pd.read_parquet(base / "features.parquet")
    if sha256(search / "plan.json") != plan["search_plan_sha256"] or sha256(base / "plan.json") != plan["base_plan_sha256"]:
        raise ValueError("Frozen component plans changed")
    search_plan = json.loads((search / "plan.json").read_text(encoding="utf-8"))
    if sha256(base / "features.parquet") != search_plan["feature_sha256"] or len(x) != len(data) or not x.index.equals(data.index):
        raise ValueError("Audited feature cache changed")
    airport_holdouts(data, specs, weights, output)
    log("EXACT_ANALYTIC_AIRPORT_HOLDOUTS_COMPLETE")
    locked = pd.read_parquet(output / "fixed-evaluation-12.parquet")
    positions = pd.Index(data[ID]).get_indexer(locked[ID])
    if (positions < 0).any():
        raise ValueError("Locked identities are absent from training corpus")
    locked_report = metrics(locked[TARGET], locked["prediction"], locked["ADEP_mvt"])
    model_root = output / "model"
    model_root.mkdir()
    fit = data[TARGET].ge(0)
    ledger = []
    for name in weights:
        log("COMPLETION_FULL_FIT " + name)
        started_at = datetime.now(timezone.utc).isoformat()
        before = time.monotonic()
        model = build_model(name, specs[name]).fit(x.loc[fit], data.loc[fit, TARGET])
        training_seconds = time.monotonic() - before
        model.save(model_root / name)
        original_started = time.monotonic()
        original = model.predict(x.iloc[positions])
        inference_seconds = time.monotonic() - original_started
        loaded = load_model(model_root / name)
        if not np.array_equal(original, loaded.predict(x.iloc[positions])):
            raise ValueError("Saved native model parity differs")
        ledger.append({"experiment": "selected-final-full-fit-" + name, "model": name, "parameters": specs[name].model_dump(),
            "features": model.columns, "training_rows": int(fit.sum()), "training_seconds": time.monotonic() - before,
            "training_time_seconds": training_seconds, "inference_time_seconds": inference_seconds,
            "inference_rows": len(locked), "native_reload_exact_rows": len(locked), "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "experiment-ledger.json", ledger)
    write_json(model_root / "model-manifest.json", {"model_version": "prc2026-forecast/1.3.0", "schema_version": "forecast/1.0.0",
        "weights": weights, "data_class": "competition", "permission_ref": plan["permission_ref"],
        "interval_radius_seconds": float(np.quantile(np.abs(locked[TARGET] - locked["prediction"]), .9)),
        "interval_status": "EMPIRICAL_DECEMBER_RESIDUAL_NOT_CALIBRATED_FUTURE_PROBABILITY",
        "product_use": "REQUIRES_SEPARATELY_LICENSED_DATA_OR_EXPLICIT_MODEL_USE_RIGHTS"})
    submissions = {"ranking": make_submission(model_root, data_root / "ranking.parquet", data_root / "submitting.parquet", output / "submissions" / "ranking_submission.parquet"),
        "final": make_submission(model_root, final_root / "final_ranking.parquet", final_root / "final_submitting.parquet", output / "submissions" / "final_submission.parquet")}
    if sha256(source / "selection.json") != sha256(output / "selection.json"):
        raise ValueError("Frozen selection changed during completion")
    report = {"BEST_CV_RMSE": selection["scores"][selection["selected"]]["rmse"], "JAN_PROXY_RMSE": locked_report["rmse"],
        "JAN_PROXY_DEFINITION": "Locked December forward winter proxy, not January calendar validation",
        "FEB_PROXY_RMSE": diagnostic_report["2"]["rmse"], "JUN_PROXY_RMSE": diagnostic_report["6"]["rmse"], "JUL_PROXY_RMSE": diagnostic_report["7"]["rmse"],
        "WORST_AIRPORT": max(locked_report["per_airport"], key=lambda a: locked_report["per_airport"][a]["rmse"]),
        "BEST_MODEL": selection["selected"], "ENSEMBLE": weights,
        "LEAKAGE_CHECK": "ALLOWLIST_AND_PURGED_FORWARD_PASS; HISTORICAL_SCHEDULE_ASOF_ASSUMPTION_DOCUMENTED",
        "FINAL_SUBMISSION": str(output / "submissions" / "final_submission.parquet"),
        "REPRODUCIBLE": "FROZEN_SELECTION_SOURCE_ENVIRONMENT_INPUT_MODEL_HASHES_NATIVE_RELOAD_PARITY",
        "PUBLIC_REPO_READY": "SOURCE_EXPORT_FINAL_VERIFICATION_PENDING", "PRODUCT_MODEL_READY": "PORTABLE_INTERFACE_TESTED_COMPETITION_WEIGHTS_RESTRICTED",
        "locked": locked_report, "diagnostics": diagnostic_report, "submissions": submissions,
        "training_departures": len(data), "nonnegative_fitting_departures": int(fit.sum()),
        "airport_stress": json.loads((output / "airport-stress.json").read_text(encoding="utf-8")),
        "source_selection_sha256": sha256(source / "selection.json"), "selection_unchanged_after_diagnostics": True}
    write_json(output / "final-report.json", report)
    log("COMPLETION_FINAL_READY " + json.dumps({k: v for k, v in report.items() if k.isupper()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "base", "search", "data", "final-data", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    finish(args.source, args.base, args.search, args.data, args.final_data, args.output)


if __name__ == "__main__":
    main()

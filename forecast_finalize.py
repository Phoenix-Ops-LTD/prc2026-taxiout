# SPDX-License-Identifier: GPL-3.0-only
"""Combine fixed independent forward searches, freeze selection, then evaluate and fit."""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from forecast_features import clean_departures, metrics, purged_split
from forecast_model import FittedForecast, ModelSpec
from forecast_run import ForecastPlan, log, make_submission, selected_fold
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json


def choose(frame: pd.DataFrame, names: list[str]) -> tuple[dict[str, float], dict[str, Any]]:
    """Fixed equal pairs need >=1s pooled gain and improvement in every selection month."""
    month = pd.to_datetime(frame[TIME], utc=True).dt.month
    candidates: dict[str, dict[str, float]] = {n: {n: 1.0} for n in names}
    for a, b in itertools.combinations([n for n in names if n not in ["mean", "median", "airport_time"]], 2):
        candidates[a + "+" + b] = {a: .5, b: .5}
    scores: dict[str, Any] = {}
    for name, weights in candidates.items():
        prediction = sum(w * frame[n].to_numpy() for n, w in weights.items())
        error = np.asarray(prediction) - frame[TARGET].to_numpy()
        scores[name] = {"weights": weights, "rmse": float(np.sqrt(np.mean(error ** 2))),
            "per_month": {str(m): float(np.sqrt(np.mean(error[month.eq(m).to_numpy()] ** 2))) for m in [9, 10, 11]}}
    single = min(names, key=lambda n: (scores[n]["rmse"], n))
    eligible = names.copy()
    for name in candidates:
        if name in names:
            scores[name]["eligible"] = True
        else:
            passed = (scores[single]["rmse"] - scores[name]["rmse"] >= 1.0
                      and all(scores[name]["per_month"][str(m)] < scores[single]["per_month"][str(m)] for m in [9, 10, 11]))
            scores[name]["eligible"] = passed
            if passed:
                eligible.append(name)
    selected = min(eligible, key=lambda n: (scores[n]["rmse"], len(candidates[n]), n))
    return candidates[selected], {"selected": selected, "best_single": single, "scores": scores,
        "rule": "Minimum pooled Sep/Oct/Nov RMSE; equal pairs require 1-second pooled gain and improvement in all three months",
        "leaderboard_used": False, "december_used_for_selection": False}


def finalize(base: Path, search: Path, data_root: Path, final_root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new final directory")
    output.mkdir(parents=True)
    base_plan = ForecastPlan.model_validate_json((base / "plan.json").read_text(encoding="utf-8"))
    search_plan = json.loads((search / "plan.json").read_text(encoding="utf-8"))
    specs = dict(base_plan.models) | {n: ModelSpec.model_validate(s) for n, s in search_plan["models"].items()}
    plan = base_plan.model_copy(update={"models": specs})
    write_json(output / "plan.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {n: s.model_dump() for n, s in specs.items()}, "selection_months": [9, 10, 11],
        "blend_rule": "Equal pairs, minimum 1-second pooled gain and improvement in all three months over best single",
        "locked_evaluation": 12, "diagnostics": [2, 6, 7], "data_class": "competition", "permission_ref": plan.permission_ref,
        "base_plan_sha256": sha256(base / "plan.json"), "search_plan_sha256": sha256(search / "plan.json")})
    source_root = Path(__file__).parent
    source_names = ["forecast_finalize.py", "forecast_search.py", "forecast_features.py", "forecast_model.py", "forecast_run.py", "pipeline.py"]
    environment = json.loads((base / "environment.json").read_text(encoding="utf-8"))
    environment["source_hashes"] = {n: sha256(source_root / n) for n in source_names}
    write_json(output / "environment.json", environment)
    log("WAITING_FOR_INDEPENDENT_FIXED_FORWARD_RESULTS")
    # Background wait does not block the conversation; no label or final-score feedback.
    started = time.monotonic()
    while True:
        try:
            if not (base / "selection-oof.parquet").exists() or not (search / "selection-oof.parquet").exists():
                raise FileNotFoundError
            left = pd.read_parquet(base / "selection-oof.parquet")
            right = pd.read_parquet(search / "selection-oof.parquet")
            break
        except (FileNotFoundError, pa.ArrowInvalid):
            if time.monotonic() - started > 7200:
                raise TimeoutError("Independent searches have not completed; no final selection made")
            time.sleep(10)
    keys = [ID, TARGET, TIME, "ADEP_mvt"]
    if not left[keys].equals(right[keys]):
        raise ValueError("Independent OOF IDs, order, labels or dates differ")
    for name in search_plan["models"]:
        left[name] = right[name].to_numpy()
    del right
    weights, selection = choose(left, list(specs))
    selection["weights"] = weights
    selection["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    selection["other_base_pipeline_december_already_scored"] = (base / "fixed-evaluation-12.parquet").exists()
    write_json(output / "selection.json", selection)
    left.to_parquet(output / "selection-oof.parquet", index=False)
    log("FINAL SELECTION " + selection["selected"] + " RMSE=" + str(selection["scores"][selection["selected"]]["rmse"]))
    audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    paths = sorted(data_root.glob("training_*.parquet"))
    for path in paths:
        if sha256(path) != audit["training"][path.name]["sha256"]:
            raise ValueError("Training input digest differs from audited source")
    if sha256(base / "features.parquet") != search_plan["feature_sha256"]:
        raise ValueError("Verified feature digest changed")
    columns = [ID, PHASE, TIME, TARGET, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"]
    data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
    x = pd.read_parquet(base / "features.parquet")
    if not x.index.equals(data.index) or len(x) != audit["training_departures"]:
        raise ValueError("Feature/departure identity mismatch")
    for name in ["audit.json", "feature-schema.json"]:
        shutil.copyfile(base / name, output / name)
    ledger = json.loads((base / "experiment-ledger.json").read_text(encoding="utf-8")) + json.loads((search / "experiment-ledger.json").read_text(encoding="utf-8"))
    selected_score = selection["scores"][selection["selected"]]
    pooled = metrics(left[TARGET], sum(w * left[n].to_numpy() for n, w in weights.items()), left["ADEP_mvt"])
    locked, locked_report = selected_fold(data, x, plan, weights, 12, output)
    diagnostics = {}
    for month in [2, 6, 7]:
        _, report = selected_fold(data, x, plan, weights, month, output)
        diagnostics[str(month)] = report
        write_json(output / "diagnostics.json", diagnostics)
    fit, valid, _ = purged_split(data, 11)
    counts = data.loc[fit, "ADEP_mvt"].value_counts()
    stress = {}
    for airport in sorted({str(counts.index[0]), str(counts.index[-1])}):
        held = valid & data["ADEP_mvt"].eq(airport).to_numpy()
        other = fit & ~data["ADEP_mvt"].eq(airport).to_numpy()
        prediction = np.zeros(int(held.sum()))
        for name, w in weights.items():
            model = FittedForecast(specs[name]).fit(x.loc[other], data.loc[other, TARGET])
            prediction += w * model.predict(x.loc[held])
        stress[airport] = metrics(data.loc[held, TARGET], prediction, data.loc[held, "ADEP_mvt"])
    write_json(output / "airport-stress.json", stress)
    model_root = output / "model"
    model_root.mkdir()
    fit_all = data[TARGET].ge(0)
    for name in weights:
        log("FINAL FULL FIT " + name)
        before = time.monotonic()
        model = FittedForecast(specs[name]).fit(x.loc[fit_all], data.loc[fit_all, TARGET])
        model.save(model_root / name)
        reloaded = FittedForecast.load(model_root / name)
        if not np.array_equal(model.predict(x.loc[locked.index]), reloaded.predict(x.loc[locked.index])):
            raise ValueError("Final native reload parity differs")
        ledger.append({"experiment": "selected-final-full-fit-" + name, "parameters": specs[name].model_dump(),
            "training_rows": int(fit_all.sum()), "training_seconds": time.monotonic() - before,
            "native_reload_exact_rows": len(locked), "features": model.columns})
    write_json(output / "experiment-ledger.json", ledger)
    write_json(model_root / "model-manifest.json", {"model_version": "prc2026-forecast/1.1.0", "schema_version": "forecast/1.0.0",
        "weights": weights, "data_class": "competition", "permission_ref": plan.permission_ref,
        "interval_radius_seconds": float(np.quantile(np.abs(locked[TARGET] - locked["prediction"]), .9)),
        "interval_status": "EMPIRICAL_DECEMBER_RESIDUAL_NOT_CALIBRATED_FUTURE_PROBABILITY",
        "product_use": "REQUIRES_SEPARATELY_LICENSED_DATA_OR_EXPLICIT_MODEL_USE_RIGHTS"})
    submissions = {"ranking": make_submission(model_root, data_root / "ranking.parquet", data_root / "submitting.parquet", output / "submissions" / "ranking_submission.parquet"),
        "final": make_submission(model_root, final_root / "final_ranking.parquet", final_root / "final_submitting.parquet", output / "submissions" / "final_submission.parquet")}
    final = {"BEST_CV_RMSE": selected_score["rmse"], "LOCKED_DECEMBER_RMSE": locked_report["rmse"],
        "JAN_PROXY_RMSE": locked_report["rmse"], "JAN_PROXY_DEFINITION": "December forward winter proxy, not January calendar validation",
        "FEB_PROXY_RMSE": diagnostics["2"]["rmse"], "JUN_PROXY_RMSE": diagnostics["6"]["rmse"], "JUL_PROXY_RMSE": diagnostics["7"]["rmse"],
        "WORST_AIRPORT": max(locked_report["per_airport"], key=lambda a: locked_report["per_airport"][a]["rmse"]),
        "BEST_MODEL": selection["selected"], "ENSEMBLE": weights,
        "LEAKAGE_CHECK": "ALLOWLIST_AND_PURGED_FORWARD_PASS; SCHEDULE_ASOF_ASSUMPTION_DOCUMENTED",
        "FINAL_SUBMISSION": str(output / "submissions" / "final_submission.parquet"),
        "REPRODUCIBLE": "FROZEN_SOURCE_ENVIRONMENT_INPUT_MODEL_HASHES_NATIVE_RELOAD_PARITY",
        "PUBLIC_REPO_READY": "SOURCE_EXPORT_AND_VERIFICATION_PENDING", "PRODUCT_MODEL_READY": "PORTABLE_INTERFACE_TESTED_COMPETITION_WEIGHTS_RESTRICTED",
        "selection": pooled, "selection_per_month": selected_score["per_month"], "locked": locked_report,
        "diagnostics": diagnostics, "submissions": submissions, "training_departures": len(data)}
    write_json(output / "final-report.json", final)
    log("FINAL COMPLETE " + json.dumps({k: v for k, v in final.items() if k.isupper()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "search", "data", "final-data", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    finalize(args.base, args.search, args.data, args.final_data, args.output)

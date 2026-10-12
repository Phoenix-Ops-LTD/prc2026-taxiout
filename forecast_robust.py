# SPDX-License-Identifier: GPL-3.0-only
"""One deadline-bounded robust-fitting challenger; no official score or label input."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_delivery import verify
from forecast_features import clean_departures, metrics, purged_split
from forecast_model import ModelSpec
from forecast_novel_finalize import make_submission
from forecast_novel_guard import NovelGuardedForecast, load_model
from forecast_run import log
from pipeline import ID, PHASE, TARGET, TIME, sha256, write_json

NAME = "robust_deep_cap7200_novel_guarded"
CAP = 7200.0
DEADLINE = datetime(2026, 10, 14, 10, tzinfo=timezone.utc)
SPEC = ModelSpec(kind="lightgbm", depth=9, iterations=1000, learning_rate=.035, demand=False, threads=2)


def require_budget(start: float) -> None:
    if time.monotonic() - start > 3600 or (DEADLINE - datetime.now(timezone.utc)).total_seconds() < 7200:
        raise ValueError("Cycle budget or two-hour deadline reserve exhausted")


def capped_targets(y: pd.Series[Any]) -> pd.Series[Any]:
    if not len(y) or not np.isfinite(y).all() or y.lt(0).any():
        raise ValueError("Fitting targets must be finite nonnegative values")
    return y.clip(upper=CAP)


def save_model(model: NovelGuardedForecast, root: Path) -> None:
    metadata = model.save(root)
    metadata["fit_target_transform"] = {"policy": "upper-cap-fitting-only/1.0", "cap_seconds": CAP,
        "validation_targets_transformed": False}
    write_json(root / "metadata.json", metadata)


def run(data_root: Path, final_data: Path, base: Path, incumbent: Path, delivery: Path, output: Path) -> None:
    started = time.monotonic()
    require_budget(started)
    if output.exists():
        raise ValueError("Use a fresh immutable challenger directory")
    output.mkdir(parents=True)
    source = Path(__file__).parent
    sources = ["forecast_robust.py", "forecast_delivery.py", "forecast_novel_finalize.py", "forecast_novel_guard.py",
        "forecast_guard.py", "forecast_model.py", "forecast_features.py", "forecast_run.py", "pipeline.py"]
    bindings = {n: sha256(source / n) for n in sources}
    audit = json.loads((base / "audit.json").read_text())
    paths = sorted(data_root.glob("training_*.parquet"))
    if len(paths) != 12:
        raise ValueError("All twelve audited training months required")
    for path in paths:
        if sha256(path) != audit["training"][path.name]["sha256"]:
            raise ValueError("Training input differs from prior audit")
    delivery_audit = json.loads((delivery / "audit.json").read_text())
    evaluations = delivery_audit["evaluation"]
    for label, ranking, template in [("ranking", data_root / "ranking.parquet", data_root / "submitting.parquet"),
        ("final", final_data / "final_ranking.parquet", final_data / "final_submitting.parquet")]:
        if sha256(ranking) != evaluations[label]["ranking_sha256"] or sha256(template) != evaluations[label]["template_sha256"]:
            raise ValueError("Evaluation input differs from delivery audit")
    base_binding = json.loads((base / "environment.json").read_text())["source_hashes"]
    if any(sha256(source / n) != digest for n, digest in base_binding.items()):
        raise ValueError("Feature-cache training source changed")
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty_at_training": True}
    write_json(output / "plan.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {NAME: SPEC.model_dump()}, "selection_months": [9, 10, 11], "execution_unit": "CPU",
        "training_overrides": {NAME: {"fit_target_cap_seconds": CAP}}, "source_hashes": bindings,
        "feature_sha256": sha256(base / "features.parquet"),
        "training_input_hashes": {p.name: sha256(p) for p in paths},
        "reused_catboost_oof_sha256": sha256(incumbent / "selection-oof.parquet"),
        "fixed_candidates": [NAME, "catboost+" + NAME], "maximum_cycle_seconds": 3600,
        "deadline_utc": DEADLINE.isoformat(), "minimum_deadline_reserve_seconds": 7200,
        "trial_qualification": "Complete native-parity forward validation; no fold RMSE more than 5% above incumbent; minimum pooled RMSE of fixed single/pair",
        "official_feedback_used_for_training": False, "validation_targets_capped": False,
        "official_retention_policy": "Existing best official score governs retention; all uploads use new increasing versions"})
    write_json(output / "environment.json", {"source_hashes": bindings})
    write_json(output / "leaderboard-provenance.json", provenance)
    columns = [ID, PHASE, TIME, TARGET, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"]
    data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True), training=True)
    x = pd.read_parquet(base / "features.parquet")
    if len(x) != len(data) or not x.index.equals(data.index):
        raise ValueError("Feature-cache identity differs")
    reference = pd.read_parquet(incumbent / "selection-oof.parquet")
    ledger: list[dict[str, Any]] = []
    parts = []
    for month in [9, 10, 11]:
        require_budget(started)
        fit, valid, purged = purged_split(data, month)
        result = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy().reset_index(drop=True)
        old = reference.loc[pd.to_datetime(reference[TIME], utc=True).dt.month.eq(month)].reset_index(drop=True)
        if not result.equals(old[[ID, TARGET, TIME, "ADEP_mvt"]]):
            raise ValueError("Reused forward checkpoint identity differs")
        log(f"FIT robust-selection-{month:02} {NAME}")
        before = time.monotonic()
        model = NovelGuardedForecast(SPEC).fit(x.loc[fit], capped_targets(data.loc[fit, TARGET]))
        fit_seconds = time.monotonic() - before
        native = output / f"native-{month:02}"
        save_model(model, native)
        before = time.monotonic()
        result[NAME] = model.predict(x.loc[valid])
        inference_seconds = time.monotonic() - before
        if not np.array_equal(result[NAME].to_numpy(), load_model(native).predict(x.loc[valid])):
            raise ValueError("Native all-row reload parity failed")
        result["catboost"] = old["catboost"]
        report = metrics(result[TARGET], result[NAME], result["ADEP_mvt"])
        result.to_parquet(output / f"selection-{month:02}.parquet", index=False)
        ledger.append({"experiment": f"forward-{month:02}-{NAME}", "model": NAME,
            "parameters": {**SPEC.model_dump(), "fit_target_cap_seconds": CAP}, "features": model.columns,
            "training_rows": int(fit.sum()), "purged_flight_rows": purged,
            "fit_targets_capped": int(data.loc[fit, TARGET].gt(CAP).sum()), "validation_month": month,
            "training_seconds": fit_seconds + inference_seconds, "training_time_seconds": fit_seconds,
            "inference_time_seconds": inference_seconds, "native_artifact_hashes": json.loads((native / "metadata.json").read_text())["artifacts"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), **report})
        write_json(output / "experiment-ledger.json", ledger)
        parts.append(result)
        log(f"RESULT selection-{month:02} {NAME} RMSE={report['rmse']:.6f}")
    pooled = pd.concat(parts, ignore_index=True)
    pooled.to_parquet(output / "selection-oof.parquet", index=False)
    baseline = json.loads((incumbent / "selection.json").read_text())
    incumbent_scores = baseline["scores"][baseline["selected"]]
    candidates = {NAME: {NAME: 1.0}, "catboost+" + NAME: {"catboost": .5, NAME: .5}}
    scores: dict[str, Any] = {}
    months = pd.to_datetime(pooled[TIME], utc=True).dt.month
    for name, weights in candidates.items():
        predicted = np.asarray(sum(w * pooled[n].to_numpy() for n, w in weights.items()), dtype=float)
        per_month = {str(m): float(np.sqrt(np.mean((predicted[months.eq(m)] - pooled.loc[months.eq(m), TARGET]) ** 2))) for m in [9, 10, 11]}
        scores[name] = {"weights": weights, **metrics(pooled[TARGET], predicted, pooled["ADEP_mvt"]),
            "per_month": per_month, "eligible": all(per_month[str(m)] <= 1.05 * incumbent_scores["per_month"][str(m)] for m in [9, 10, 11])}
    eligible = [name for name, score in scores.items() if score["eligible"]]
    selected = min(eligible, key=lambda n: scores[n]["rmse"]) if eligible else None
    write_json(output / "selection.json", {"selected": selected, "scores": scores,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(), "leaderboard_used": False,
        "official_trial_qualified": bool(eligible), "incumbent_rmse": incumbent_scores["rmse"]})
    if selected is None:
        log("ROBUST_TRIAL_NOT_QUALIFIED; no full refit or submission")
        return
    require_budget(started)
    model_root = output / "model"
    model_root.mkdir()
    log("FIT robust-full-2025 " + NAME)
    fit = data[TARGET].ge(0).to_numpy()
    before = time.monotonic()
    model = NovelGuardedForecast(SPEC).fit(x.loc[fit], capped_targets(data.loc[fit, TARGET]))
    save_model(model, model_root / NAME)
    full_seconds = time.monotonic() - before
    weights = candidates[selected]
    if "catboost" in weights:
        shutil.copytree(delivery / "model/catboost", model_root / "catboost")
    original = json.loads((delivery / "model/model-manifest.json").read_text())
    write_json(model_root / "model-manifest.json", {**original, "model_version": "prc2026-forecast/1.4.0-cap7200",
        "weights": weights, "fit_target_cap_seconds": CAP, "full_fit_seconds": full_seconds})
    for label, ranking, template in [("ranking", data_root / "ranking.parquet", data_root / "submitting.parquet"),
        ("final", final_data / "final_ranking.parquet", final_data / "final_submitting.parquet")]:
        require_budget(started)
        artifact = output / f"submissions/{label}_submission.parquet"
        make_submission(model_root, ranking, template, artifact)
        verify(output, ranking, template, artifact, output / f"{label}-verification.json")
    if any(sha256(source / n) != h for n, h in bindings.items()):
        raise ValueError("Frozen challenger source changed")
    write_json(output / "report.json", {"status": "COMPLETE_ALL_ROWS_NATIVE_VERIFIED_OFFICIAL_TRIAL_READY",
        "selected": selected, "scores": scores, "full_fit_seconds": full_seconds,
        "elapsed_seconds": time.monotonic() - started, "official_score": None})
    log("ROBUST_TRIAL_READY " + selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "final-data", "base", "incumbent", "delivery", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.final_data, args.base, args.incumbent, args.delivery, args.output)

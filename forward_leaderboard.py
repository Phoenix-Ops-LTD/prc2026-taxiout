# SPDX-License-Identifier: GPL-3.0-only
"""Live, source-bound MAE leaderboard; never reads official challenge scores."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast_features import clean_departures, purged_split
from pipeline import ID, PHASE, TARGET, TIME, sha256

MONTHS = (9, 10, 11)
CATASTROPHIC_SECONDS = 3600.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def format_json(report: dict[str, Any], output: Path, source: Path) -> str:
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Honor the parent's formatter when installed; standalone research needs no JS.
    root = source.parents[1]
    executable = "biome.exe" if os.name == "nt" else "biome"
    binaries = sorted(root.glob(f"node_modules/.pnpm/@biomejs+cli-*/node_modules/@biomejs/cli-*/{executable}"))
    if binaries:
        result = subprocess.run([str(binaries[0]), "format", "--stdin-file-path", str(output)],
            input=content, text=True, capture_output=True, cwd=root, check=True)
        return result.stdout
    return content


def cohorts(data: pd.DataFrame) -> pd.DataFrame:
    """Novelty is measured against each actual purged, nonnegative fitting set."""
    parts = []
    aircraft = data["AIRCRAFT_TYPE_mvt"].fillna("UNKNOWN").astype(str)
    airport = data["ADEP_mvt"].fillna("UNKNOWN").astype(str)
    stand = airport + ":" + data["STAND_mvt"].fillna("UNKNOWN").astype(str)
    for month in MONTHS:
        fit, valid, _ = purged_split(data, month)
        part = data.loc[valid, [ID, TARGET, TIME, "ADEP_mvt"]].copy()
        part["unseen_aircraft"] = ~aircraft.loc[valid].isin(set(aircraft.loc[fit]))
        # Missing stand is unknown provenance, not an observed new stand.
        part["unseen_airport_stand"] = (~airport.loc[valid].isin(set(airport.loc[fit]))
            | (data.loc[valid, "STAND_mvt"].notna() & ~stand.loc[valid].isin(set(stand.loc[fit & data["STAND_mvt"].notna().to_numpy()]))))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def score(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    actual = frame[TARGET].to_numpy(dtype=float)
    predicted = frame[name].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Invalid scored predictions")
    error = np.abs(predicted - actual)
    months = pd.to_datetime(frame[TIME], utc=True).dt.month.to_numpy()
    fold_mae = {str(m): float(error[months == m].mean()) for m in sorted(set(months))}
    worst = np.sort(error)[-max(1, math.ceil(len(error) / 10)):]
    groups: dict[str, Any] = {}
    for flag in ["unseen_aircraft", "unseen_airport_stand"]:
        mask = frame[flag].to_numpy(dtype=bool)
        groups[flag + "_mae"] = float(error[mask].mean()) if mask.any() else None
        groups[flag + "_rows"] = int(mask.sum())
    return {"mean_mae": float(np.mean(list(fold_mae.values()))),
        "median_mae": float(np.median(list(fold_mae.values()))),
        "worst_fold_mae": max(fold_mae.values()), "worst_decile_mae": float(worst.mean()),
        "pooled_mae": float(error.mean()), "per_fold_mae": fold_mae,
        "catastrophic_error_rate": float((error > CATASTROPHIC_SECONDS).mean()),
        "validation_rows": len(error), **groups}


def rank_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted([e for e in entries if e["status"] == "COMPLETED_VALIDATED"],
                    key=lambda e: (e["mean_mae"], e["worst_fold_mae"], e["experiment_id"]))
    for rank, entry in enumerate(ranked, 1):
        entry["rank"] = rank
        entry["champion"] = rank == 1
    pending = sorted([e for e in entries if e["status"] != "COMPLETED_VALIDATED"], key=lambda e: e["experiment_id"])
    for entry in pending:
        entry["rank"] = None
        entry["champion"] = False
    return ranked + pending


def completed_time(run: Path, month: int, model: str) -> str:
    log = run.with_suffix(".log")
    pattern = re.compile(r"^(\S+) (?:RESULT selection-|PARALLEL RESULT )(\d+) (\S+) RMSE=")
    if log.exists():
        raw = log.read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        for line in raw.decode(encoding).splitlines():
            match = pattern.match(line)
            if match and int(match[2]) == month and match[3] == model:
                return match[1]
    return datetime.fromtimestamp((run / f"selection-{month:02}.parquet").stat().st_mtime, timezone.utc).isoformat()


def snapshot(runs: list[Path], context: pd.DataFrame, source: Path, commit: str,
             blend_runs: list[Path] | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in runs:
        try:
            plan = load_json(run / "plan.json")
            binding = load_json(run / "environment.json") if (run / "environment.json").exists() else plan
            bound = binding.get("source_hashes", {})
            if not bound:
                raise ValueError("No recorded training-source hashes")
            preserved = run / "leaderboard-source-bindings"
            for n, h in bound.items():
                if Path(n).name != n:
                    raise ValueError("Source binding must be a module basename")
                saved = preserved / n
                if not saved.exists() and sha256(source / n) == h:
                    preserved.mkdir(exist_ok=True)
                    shutil.copyfile(source / n, saved)
                if not saved.exists() or sha256(saved) != h:
                    raise ValueError("Recorded training-source hashes do not match preserved source")
            provenance_path = run / "leaderboard-provenance.json"
            if not provenance_path.exists():
                atomic_write(provenance_path, json.dumps({"git_commit": commit,
                    "git_dirty_at_training": True, "commit_provenance": "SESSION_HEAD_AT_FIRST_OBSERVATION; source hashes identify uncommitted implementation"}))
            provenance = load_json(provenance_path)
            ledger = load_json(run / "experiment-ledger.json") if (run / "experiment-ledger.json").exists() else []
            results: dict[int, pd.DataFrame] = {}
            for month in MONTHS:
                path = run / f"selection-{month:02}.parquet"
                if not path.exists():
                    continue
                result = pd.read_parquet(path)
                expected = context.loc[pd.to_datetime(context[TIME], utc=True).dt.month.eq(month)]
                keys = [ID, TARGET, TIME, "ADEP_mvt"]
                if not result[keys].reset_index(drop=True).equals(expected[keys].reset_index(drop=True)):
                    raise ValueError("Prediction identity, labels or folds differ from purged validation")
                results[month] = result.merge(expected[[ID, "unseen_aircraft", "unseen_airport_stand"]], on=ID, validate="one_to_one")
            for name, spec in plan["models"].items():
                if "leaderboard_models" in plan and name not in plan["leaderboard_models"]:
                    continue
                completed: list[pd.DataFrame] = []
                timing = 0.0
                timestamps = []
                for month in MONTHS:
                    rows = [e for e in ledger if e.get("experiment") == f"forward-{month:02}-{name}"]
                    if not rows or month not in results:
                        continue
                    result = results[month]
                    if name not in result:
                        continue
                    measured = score(result, name)
                    if not math.isclose(measured["pooled_mae"], rows[0]["mae"], abs_tol=1e-8):
                        raise ValueError("Prediction MAE does not match experiment ledger")
                    stamp = completed_time(run, month, name)
                    seconds = float(rows[0]["training_seconds"])
                    metadata = {"model": name, "model_family": spec["kind"], "device": "CPU",
                        **provenance, "source_hashes": bound,
                        "training_time_seconds": None, "inference_time_seconds": None,
                        "timing_status": "LEGACY_COMBINED_FIT_PREDICT_METRICS_ONLY", "combined_time_seconds": seconds}
                    folds.append({**metadata, **measured, "experiment_id": run.name + "/" + rows[0]["experiment"],
                        "rank": None, "status": "COMPLETED_SINGLE_FOLD_NOT_COMPARABLE_TO_FULL_CV",
                        "validation_folds": [f"2025-{month:02}"], "timestamp": stamp})
                    completed.append(result)
                    timing += seconds
                    timestamps.append(stamp)
                entry: dict[str, Any] = {"experiment_id": run.name + "/" + name, "model": name,
                    "model_family": spec["kind"], "parameters": spec, "device": "CPU", **provenance,
                    "source_hashes": bound, "timestamp": max(timestamps) if timestamps else None,
                    "validation_folds": [f"2025-{int(pd.to_datetime(p[TIME], utc=True).dt.month.iloc[0]):02}" for p in completed],
                    "training_time_seconds": None, "inference_time_seconds": None,
                    "timing_status": "LEGACY_COMBINED_FIT_PREDICT_METRICS_ONLY", "combined_time_seconds": timing,
                    "status": "COMPLETED_VALIDATED" if len(completed) == len(MONTHS) else "INCOMPLETE_UNRANKED",
                    **{k: None for k in ["mean_mae", "median_mae", "worst_fold_mae", "worst_decile_mae", "unseen_aircraft_mae", "unseen_airport_stand_mae", "catastrophic_error_rate"]}}
                if completed:
                    entry.update(score(pd.concat(completed, ignore_index=True), name))
                entries.append(entry)
        except (ValueError, KeyError, OSError) as exc:
            warnings.append(run.name + ": " + str(exc))
    # Register only fixed ensembles actually recorded by the frozen selectors.
    validated = {e["model"]: e for e in entries if e["status"] == "COMPLETED_VALIDATED"}
    for run in blend_runs or []:
        selection_path = run / "selection.json"
        oof_path = run / "selection-oof.parquet"
        if not selection_path.exists() or not oof_path.exists():
            continue
        try:
            selection = load_json(selection_path)
            prediction = pd.read_parquet(oof_path)
            keys = [ID, TARGET, TIME, "ADEP_mvt"]
            if not prediction[keys].equals(context[keys]):
                raise ValueError("Ensemble OOF identity does not match forward context")
            prediction = prediction.merge(context[[ID, "unseen_aircraft", "unseen_airport_stand"]], on=ID, validate="one_to_one")
            for name, candidate in selection["scores"].items():
                weights = candidate["weights"]
                if len(weights) < 2:
                    continue
                if not set(weights).issubset(validated):
                    raise ValueError("Ensemble component has not completed validated forward CV")
                prediction["__blend"] = sum(float(w) * prediction[n].to_numpy() for n, w in weights.items())
                measured = score(prediction, "__blend")
                rmse = float(np.sqrt(np.mean((prediction["__blend"] - prediction[TARGET]) ** 2)))
                if not math.isclose(rmse, candidate["rmse"], abs_tol=1e-6):
                    raise ValueError("Ensemble predictions do not match recorded score")
                components = [validated[n] for n in weights]
                entries.append({**measured, "model": name, "model_family": "fixed_equal_ensemble",
                    "experiment_id": run.name + "/" + name, "status": "COMPLETED_VALIDATED",
                    "validation_folds": ["2025-09", "2025-10", "2025-11"], "device": "CPU",
                    "timestamp": selection.get("selected_at_utc") or datetime.fromtimestamp(oof_path.stat().st_mtime, timezone.utc).isoformat(),
                    "git_commit": components[0]["git_commit"], "git_dirty_at_training": True,
                    "source_hashes": {k: v for c in components for k, v in c["source_hashes"].items()},
                    "parameters": {"weights": weights}, "frozen_rmse_selection_eligible": candidate.get("eligible", True),
                    "training_time_seconds": None, "inference_time_seconds": None,
                    "combined_time_seconds": sum(c["combined_time_seconds"] for c in components),
                    "timing_status": "SUM_OF_LEGACY_COMPONENT_TIMES; no retraining for fixed blend"})
        except (ValueError, KeyError, OSError) as exc:
            warnings.append(run.name + " ensembles: " + str(exc))
    return {"schema_version": "prc-forward-leaderboard/1.0", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "Rank completed Sep/Oct/Nov purged forward CV by arithmetic mean fold MAE, lower is better",
        "mean_mae_definition": "Equal-weight mean of fold MAEs; median_mae is median of fold MAEs",
        "catastrophic_threshold_seconds": CATASTROPHIC_SECONDS, "catastrophic_comparison": "absolute error > threshold",
        "worst_decile_definition": "Mean of largest ceil(10% * pooled validation rows) absolute errors",
        "unseen_definition": "Category absent from corresponding purged, nonnegative training fold; airport/stand is union of unseen airport and unseen nonmissing airport-scoped stand",
        "stand_provenance": "Retrospective subgroup diagnostic only; stand excluded from model features",
        "availability_caveat": "Operational clocks excluded; supplied published schedules assumed available, historical as-of snapshots unverified",
        "official_feedback_used": False, "locked_december_used_for_ranking": False,
        "submission_selection": "Existing frozen submission protocol uses RMSE; MAE CHAMPION is reported independently",
        "timing_caveat": "Existing training_seconds includes fitting, prediction and metrics; separate times unavailable and explicitly null",
        "entries": rank_entries(entries), "completed_fold_experiments": folds, "warnings": warnings}


def render(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "N/A"
        return f"{value:.3f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    lines = ["# PRC forward-validation model leaderboard", "", "Updated UTC: " + report["updated_at_utc"], "",
        "Lower mean MAE wins. Only completed, source-bound September/October/November purged forward experiments receive a rank. Official scores and locked December results never enter this board.", "",
        "Mean/median/worst-fold MAE summarize fold MAEs; worst-decile MAE uses the worst 10% of pooled errors. All MAEs and times are seconds. Catastrophic rate means absolute error > 3,600 seconds. Unseen cohorts use each fold's fitting set. Stand is a retrospective diagnostic and is excluded from predictors.", "",
        "Published schedule availability remains an explicit assumption because historical as-of snapshots are absent. Existing timers combine fitting, prediction and metrics: separate train/inference times are N/A. The frozen submission protocol selects by RMSE; its selected model may differ from the MAE CHAMPION.", "",
        "| Rank | Model | Experiment id | Mean MAE | Median MAE | Worst-fold MAE | Worst-decile MAE | Unseen aircraft MAE | Unseen airport/stand MAE | Catastrophic rate | Training time | Inference time | Combined time | Folds | Device | Timestamp UTC | Git commit |", "|" + " --- |" * 17]
    for entry in report["entries"]:
        if entry["rank"] is None:
            continue
        rank = str(entry["rank"]) + (" **CHAMPION**" if entry["champion"] else "")
        fields = [rank, entry["model"], entry["experiment_id"], *[entry.get(k) for k in ["mean_mae", "median_mae", "worst_fold_mae", "worst_decile_mae", "unseen_aircraft_mae", "unseen_airport_stand_mae"]],
            f"{100 * entry['catastrophic_error_rate']:.4f}%", entry["training_time_seconds"], entry["inference_time_seconds"],
            entry["combined_time_seconds"], ", ".join(entry["validation_folds"]), entry["device"], entry["timestamp"], entry["git_commit"][:12] + " + dirty source hashes"]
        lines.append("| " + " | ".join(cell(v) for v in fields) + " |")
    lines += ["", "## Incomplete experiments (unranked)", ""]
    for entry in report["entries"]:
        if entry["rank"] is None:
            lines.append(f"- {entry['experiment_id']}: {len(entry['validation_folds'])}/3 folds complete; no rank.")
    lines += ["", "Every completed individual fold, its scores, timestamp, source hashes and timing limitations are retained in `leaderboard.json` under `completed_fold_experiments`. Historical official, retrospective and incompatible protocols are preserved in their original reports and excluded from this comparable forward board.", "", "Refresh once: `python forward_leaderboard.py --data PATH --runs runs/generalization-v3 runs/generalization-search-v1`", "", "Continuous update: add `--watch --interval 5`. The watcher updates after each newly committed experiment ledger/prediction pair and atomically replaces each output. It does not change frozen training sources.", ""]
    if report["warnings"]:
        lines += ["Validation warnings:", "", *["- " + w for w in report["warnings"]], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--selections", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("Polling interval must be positive")
    source = Path(__file__).parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    cache = source / "runs" / "leaderboard-cohorts.parquet"
    audit = load_json(args.runs[0] / "audit.json")
    inputs = sorted(args.data.glob("training_*.parquet"))
    if len(inputs) != 12 or any(sha256(p) != audit["training"][p.name]["sha256"] for p in inputs):
        raise ValueError("Training inputs differ from audited source")
    signature = json.dumps(audit["training"], sort_keys=True)
    receipt = cache.with_suffix(".json")
    if not cache.exists() or not receipt.exists() or load_json(receipt).get("audit_signature") != signature:
        columns = [ID, PHASE, TARGET, TIME, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt", "STAND_mvt"]
        data = clean_departures(pd.concat([pd.read_parquet(p, columns=columns) for p in inputs], ignore_index=True), training=True)
        cohorts(data).to_parquet(cache, index=False)
        atomic_write(receipt, json.dumps({"audit_signature": signature, "sha256": sha256(cache)}))
    if sha256(cache) != load_json(receipt)["sha256"]:
        raise ValueError("Cohort cache digest changed")
    context = pd.read_parquet(cache)
    previous: list[tuple[str, int, int]] | None = None
    while True:
        files = [p for r in args.runs for p in [r / "experiment-ledger.json", *r.glob("selection-*.parquet")]]
        files += [p for r in args.selections for p in [r / "selection.json", r / "selection-oof.parquet"]]
        current = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()]
        if current != previous:
            try:
                report = snapshot(args.runs, context, source, commit, args.selections)
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_write(args.output / "leaderboard.json", format_json(report, args.output / "leaderboard.json", source))
                atomic_write(args.output / "LEADERBOARD.md", render(report))
                print(report["updated_at_utc"], "UPDATED", len(report["completed_fold_experiments"]), "completed folds", flush=True)
                previous = current
            except (OSError, ValueError) as exc:
                print("RETRY_TRANSIENT_WRITE", str(exc), flush=True)
                if not args.watch:
                    raise
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

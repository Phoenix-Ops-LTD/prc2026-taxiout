# SPDX-License-Identifier: GPL-3.0-only
"""Loopback-only live local experiments and read-only official ranking observer."""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from forward_leaderboard import atomic_write
from leaderboard import URL, best_per_team


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def official_snapshot(rows: list[dict[str, Any]], team_name: str) -> dict[str, Any]:
    normalized = []
    for row in rows:
        if row.get("score") is not None and (not math.isfinite(float(row["score"])) or float(row["score"]) < 0):
            raise ValueError("Invalid official score")
        normalized.append({**row, "score": float(row["score"]) if row.get("score") is not None else None})
    teams = best_per_team(normalized)
    own = next((r for r in teams if r["teamName"] == team_name), None)
    if own is None:
        raise ValueError("Own team absent from complete public ranking")
    rank = 1 + sum(r["score"] < own["score"] for r in teams)
    return {"status": "OBSERVED_COMPLETE_PUBLIC_RANKING", "observed_at_utc": now(), "team": team_name,
        "rank": rank, "teams": len(teams), "score": own["score"], "submission": own.get("filename"),
        "submission_processed_at": own.get("processedAt"), "leader_score": teams[0]["score"],
        "gap_to_first": own["score"] - teams[0]["score"], "source": URL,
        "used_for_model_selection": False, "submissions_count": len(rows)}


def fetch_observed(root: Path) -> list[dict[str, Any]]:
    """Publish collection progress, but never rank an incomplete API traversal."""
    (root / "runs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    start = time.monotonic()
    started_at = now()
    pages = 0
    while True:
        atomic_write(root / "runs/live-official-fetch.json", json.dumps({"status": "FETCHING_ALL_PUBLIC_PAGES",
            "started_at_utc": started_at, "pages_received": pages, "submissions_received": len(rows)}))
        if time.monotonic() - start > 900:
            raise TimeoutError("Complete public ranking exceeded the observation time budget")
        url = URL + ("?cursor=" + urllib.parse.quote(cursor, safe="") if cursor else "")
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
        rows.extend(payload["items"])
        pages += 1
        cursor = payload.get("nextCursor")
        if not cursor:
            identities = [r["submissionId"] for r in rows]
            if len(identities) != len(set(identities)):
                raise ValueError("Public ranking changed during pagination; retry without ranking partial pages")
            atomic_write(root / "runs/live-official-fetch.json", json.dumps({"status": "COMPLETE",
                "started_at_utc": started_at, "completed_at_utc": now(), "pages_received": pages,
                "submissions_received": len(rows), "duration_seconds": time.monotonic() - start}))
            return rows
        if cursor in seen:
            raise ValueError("Repeated API cursor; incomplete ranking rejected")
        seen.add(cursor)


def poll_official(root: Path, team: str, interval: float, stop: threading.Event) -> None:
    (root / "runs").mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "runs/live-official-ranking.json"
    history_path = root / "runs/live-official-history.json"
    while not stop.is_set():
        try:
            result = official_snapshot(fetch_observed(root), team)
            old = read_json(snapshot_path, {})
            history = read_json(history_path, [])
            fields = ["rank", "score", "leader_score", "teams"]
            if not history or any(result[k] != old.get(k) for k in fields):
                history.append({k: result[k] for k in ["observed_at_utc", *fields]})
                atomic_write(history_path, json.dumps(history[-2000:], indent=2))
            atomic_write(snapshot_path, json.dumps(result, indent=2))
            print(now(), "OFFICIAL_OBSERVED", result["rank"], result["teams"], result["score"], flush=True)
        except Exception as exc:
            previous = read_json(snapshot_path, {})
            previous.update(status="STALE_PUBLIC_OBSERVER_ERROR", last_attempt_at_utc=now(), error=type(exc).__name__)
            atomic_write(snapshot_path, json.dumps(previous, indent=2))
            atomic_write(root / "runs/live-official-fetch.json", json.dumps({"status": "RETRY_PENDING", "error": type(exc).__name__}))
            print(now(), "OFFICIAL_OBSERVER_RETRY", type(exc).__name__, flush=True)
        stop.wait(interval)


def gpu_progress(root: Path) -> dict[str, Any]:
    path = root / "runs/generalization-gpu-search-v1.log"
    if not path.exists():
        return {"status": "NOT_STARTED"}
    with path.open("rb") as handle:
        prefix = handle.read(2)
        handle.seek(max(0, path.stat().st_size - 32768))
        raw = handle.read()
    text = raw.decode("utf-16-le" if prefix == b"\xff\xfe" else "utf-8-sig", errors="replace")
    if "PRC_RESUME_TRANSFER_AND_FROZEN_FIT" in text:
        text = text.rsplit("PRC_RESUME_TRANSFER_AND_FROZEN_FIT", 1)[1]
    fits = re.findall(r"(\d{4}-\d\d-\d\dT\S+) FIT (\d+) (gpu_\S+)", text)
    current_text = text.rsplit(" FIT ", 1)[-1] if fits else ""
    steps = re.findall(r"(\d+):\s+learn:\s+([\d.]+).*?remaining:\s*([^\r\n]+)", current_text)
    status = "COMPLETED" if "REMOTE_EXIT 0" in text else "FAILED" if re.search(r"REMOTE_EXIT [1-9]|Traceback", text) else "RUNNING"
    return {"status": status, "model": fits[-1][2] if fits else None,
        "validation_month": int(fits[-1][1]) if fits else None,
        "started_at_utc": fits[-1][0] if fits else None, "iteration": int(steps[-1][0]) if steps else None,
        "remaining": steps[-1][2] if steps else None}


def dashboard_state(root: Path) -> dict[str, Any]:
    board = read_json(root / "leaderboard.json", {"entries": []})
    entries = board["entries"]
    keys = ["rank", "champion", "model", "experiment_id", "mean_mae", "median_mae", "worst_fold_mae", "worst_decile_mae",
        "unseen_aircraft_mae", "unseen_airport_stand_mae", "catastrophic_error_rate", "training_time_seconds",
        "inference_time_seconds", "device", "validation_folds", "timestamp", "status"]
    safe_entries = [{k: e.get(k) for k in keys} for e in entries]
    gpu_run = root / "runs/generalization-gpu-search-v1"
    gpu_report = read_json(gpu_run / "report.json", {})
    comparison = read_json(root / "runs/generalization-gpu-compare-v1/selection.json", {})
    compared_scores = comparison.get("scores", {})
    best_new = min(compared_scores, key=lambda n: compared_scores[n]["rmse"]) if compared_scores else None
    ledger = read_json(gpu_run / "experiment-ledger.json", [])
    plan = read_json(gpu_run / "plan.json", {"models": {}})
    incumbent = read_json(root / "runs/generalization-delivery-v4/final-report.json", {})
    gpu_models = []
    for name, spec in plan["models"].items():
        records = [r for r in ledger if r["model"] == name]
        completed = len(records) == 3 and {r["validation_month"] for r in records} == {9, 10, 11}
        score = gpu_report.get("scores", {}).get(name, {}) if completed else {}
        if completed and "rmse" not in score:
            score = {"rmse": math.sqrt(sum(r["n"] * r["rmse"] ** 2 for r in records) / sum(r["n"] for r in records))}
        gpu_models.append({"model": name, "folds_completed": len(records), "iterations": spec["iterations"],
            "folds": [{"month": r["validation_month"], "rmse": r["rmse"], "mae": r["mae"],
                       "training_seconds": r["training_time_seconds"]} for r in records],
            "pooled_rmse": score.get("rmse"),
            "gain_vs_incumbent": incumbent["BEST_CV_RMSE"] - score["rmse"] if "rmse" in score and "BEST_CV_RMSE" in incumbent else None})
    return {"served_at_utc": now(), "local_refresh_seconds": 5, "official_refresh_seconds": 180,
        "board_updated_at_utc": board.get("updated_at_utc"), "local_models": safe_entries[:20],
        "completed_candidates": sum(e["rank"] is not None for e in entries),
        "incomplete_candidates": sum(e["rank"] is None for e in entries),
        "champion": next((e for e in safe_entries if e["champion"]), None),
        "official": read_json(root / "runs/live-official-ranking.json", {"status": "FIRST_OBSERVATION_PENDING"}),
        "official_history": read_json(root / "runs/live-official-history.json", []),
        "official_fetch": read_json(root / "runs/live-official-fetch.json", {}),
        "gpu": gpu_progress(root), "gpu_models": gpu_models,
        "gpu_comparison": {"promoted": comparison.get("promoted"), "selected": comparison.get("selected"),
            "best_new_model": best_new, "best_new_rmse": compared_scores[best_new]["rmse"] if best_new else None,
            "gain_vs_incumbent": compared_scores[best_new]["gain_vs_incumbent"] if best_new else None},
        "incumbent_cv_rmse": incumbent.get("BEST_CV_RMSE"), "incumbent_model": incumbent.get("BEST_MODEL"),
        "official_feedback_used_for_selection": False, "submission_uploaded": False}


def handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ["/", "/dashboard.html"]:
                body = (root / "dashboard.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            elif path == "/api/status":
                body = json.dumps(dashboard_state(root), allow_nan=False).encode("utf-8")
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--team", default="zestful-fountain")
    parser.add_argument("--no-official", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    stop = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(root))
    if not args.no_official:
        threading.Thread(target=poll_official, args=(root, args.team, 180., stop), daemon=True).start()
    print(now(), f"PRC_DASHBOARD http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-3.0-only
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import prc_dashboard as dashboard


def test_official_ranking_uses_best_per_team_and_competition_ties() -> None:
    rows = [{"teamId": t, "teamName": t, "score": s} for t, s in [("own", "9"), ("first", 1.), ("own", 2.), ("tied", 2.), ("unscored", None)]]
    result = dashboard.official_snapshot(rows, "own")
    assert (result["rank"], result["teams"], result["score"], result["gap_to_first"]) == (2, 3, 2., 1.)
    assert result["used_for_model_selection"] is False
    with pytest.raises(ValueError, match="absent"):
        dashboard.official_snapshot(rows, "missing")
    with pytest.raises(ValueError, match="Invalid"):
        dashboard.official_snapshot(rows + [{"score": float("nan")}], "own")


@pytest.mark.parametrize("pages", [
    [{"items": [], "nextCursor": "a"}, {"items": [], "nextCursor": "a"}],
    [{"items": [{"submissionId": "same"}], "nextCursor": "a"}, {"items": [{"submissionId": "same"}]}],
])
def test_incomplete_or_unstable_pagination_never_produces_a_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pages: list[dict[str, object]]) -> None:
    responses = iter(pages)
    monkeypatch.setattr(dashboard.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(next(responses)).encode()))
    with pytest.raises(ValueError):
        dashboard.fetch_observed(tmp_path)
    assert not (tmp_path / "runs/live-official-ranking.json").exists()


def test_failed_observation_keeps_last_good_time_and_does_not_mutate_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    original = {"rank": 4, "observed_at_utc": "old", "score": 12.}
    (runs / "live-official-ranking.json").write_text(json.dumps(original))
    (tmp_path / "leaderboard.json").write_text("selection unchanged")
    stop = threading.Event()

    def fail(_: Path) -> list[dict[str, object]]:
        stop.set()
        raise TimeoutError()

    monkeypatch.setattr(dashboard, "fetch_observed", fail)
    dashboard.poll_official(tmp_path, "own", 0, stop)
    result = json.loads((runs / "live-official-ranking.json").read_text())
    assert result["status"] == "STALE_PUBLIC_OBSERVER_ERROR"
    assert result["rank"] == 4 and result["observed_at_utc"] == "old"
    assert (tmp_path / "leaderboard.json").read_text() == "selection unchanged"


def test_gpu_cv_gain_requires_all_three_completed_folds(tmp_path: Path) -> None:
    run = tmp_path / "runs/generalization-gpu-search-v1"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"models": {"test": {"iterations": 1}}}))
    records = [{"model": "test", "validation_month": m, "n": 10, "rmse": 2., "mae": 1., "training_time_seconds": .5} for m in [9, 10, 11]]
    path = run / "experiment-ledger.json"
    path.write_text(json.dumps(records[:2]))
    assert dashboard.dashboard_state(tmp_path)["gpu_models"][0]["pooled_rmse"] is None
    path.write_text(json.dumps(records))
    assert dashboard.dashboard_state(tmp_path)["gpu_models"][0]["pooled_rmse"] == 2.
    assert dashboard.dashboard_state(tmp_path)["completed_candidates"] == 0


def test_resume_progress_does_not_show_previous_failure_or_previous_trees(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "generalization-gpu-search-v1.log").write_text("Traceback\n4999: learn: 1 remaining: 0us\nPRC_RESUME_TRANSFER_AND_FROZEN_FIT\n2026-10-09T00:00:00Z FIT 11 gpu_new {}\n")
    result = dashboard.gpu_progress(tmp_path)
    assert result["status"] == "RUNNING" and result["iteration"] is None


def test_http_observer_does_not_serve_runtime_files(tmp_path: Path) -> None:
    (tmp_path / "dashboard.html").write_text("<title>dashboard</title>")
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url + "/api/status") as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response)["official_feedback_used_for_selection"] is False
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(url + "/runs/credentials.json")
        assert failure.value.code == 404
    finally:
        server.shutdown()
        server.server_close()

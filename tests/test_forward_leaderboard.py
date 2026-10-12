# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path

from forward_leaderboard import cohorts, rank_entries, score, snapshot
from pipeline import ID, TARGET, TIME, sha256


def test_complete_runs_rank_by_equal_fold_mae_and_partial_never_wins() -> None:
    entries = [
        {"experiment_id": "partial", "status": "INCOMPLETE_UNRANKED", "mean_mae": 0},
        {"experiment_id": "high", "status": "COMPLETED_VALIDATED", "mean_mae": 10, "worst_fold_mae": 12},
        {"experiment_id": "low", "status": "COMPLETED_VALIDATED", "mean_mae": 5, "worst_fold_mae": 15},
    ]
    ranked = rank_entries(entries)
    assert [e["experiment_id"] for e in ranked] == ["low", "high", "partial"]
    assert ranked[0]["champion"] and ranked[0]["rank"] == 1
    assert ranked[2]["rank"] is None and not ranked[2]["champion"]


def test_mae_tail_and_catastrophic_metrics_have_independent_oracle() -> None:
    frame = pd.DataFrame({TARGET: [0.0] * 10, "model": [1.0] * 8 + [3600.0, 4000.0],
        TIME: pd.to_datetime(["2025-09-01"] * 8 + ["2025-10-01", "2025-11-01"], utc=True),
        "unseen_aircraft": [False] * 9 + [True], "unseen_airport_stand": [False] * 10})
    result = score(frame, "model")
    assert result["mean_mae"] == pytest.approx((1 + 3600 + 4000) / 3)
    assert result["median_mae"] == 3600
    assert result["worst_fold_mae"] == result["worst_decile_mae"] == 4000
    assert result["catastrophic_error_rate"] == .1
    assert result["unseen_aircraft_mae"] == 4000
    assert result["unseen_airport_stand_mae"] is None
    frame.loc[0, "model"] = np.nan
    with pytest.raises(ValueError):
        score(frame, "model")


def test_novelty_uses_purged_fitting_set_and_observed_stands() -> None:
    frame = pd.DataFrame({ID: [1, 2, 3, 4, 5], TARGET: [10.] * 5,
        TIME: pd.to_datetime(["2025-01-01", "2025-02-01", "2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
        "ADEP_mvt": ["AAA"] * 5, "AIRCRAFT_TYPE_mvt": ["A", "B", "A", "A", "A"],
        "FLIGHT_ID_mvt": ["repeated", "other", "repeated", "oct", "nov"],
        "STAND_mvt": ["1", "2", "1", None, "3"]})
    result = cohorts(frame).set_index(ID)
    assert bool(result.loc[3, "unseen_aircraft"])
    assert bool(result.loc[3, "unseen_airport_stand"])
    assert not bool(result.loc[4, "unseen_airport_stand"])
    assert bool(result.loc[5, "unseen_airport_stand"])


def test_source_bound_snapshot_reads_utf16_logs_and_rejects_corruption(tmp_path: Path) -> None:
    run = tmp_path / "trial"
    run.mkdir()
    source = tmp_path / "model.py"
    source.write_text("# original GPL model\n", encoding="utf-8")
    plan = {"models": {"candidate": {"kind": "lightgbm"}}, "source_hashes": {source.name: sha256(source)}}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    context = pd.DataFrame({ID: [1, 2, 3], TARGET: [10., 20., 30.], "ADEP_mvt": ["AAA"] * 3,
        TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
        "unseen_aircraft": [False] * 3, "unseen_airport_stand": [False] * 3})
    ledger = []
    for index, month in enumerate([9, 10, 11]):
        result = context.iloc[[index]][[ID, TARGET, TIME, "ADEP_mvt"]].copy()
        result["candidate"] = result[TARGET] + 1
        result.to_parquet(run / f"selection-{month:02}.parquet", index=False)
        ledger.append({"experiment": f"forward-{month:02}-candidate", "mae": 1., "training_seconds": 2.})
    (run / "experiment-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    run.with_suffix(".log").write_text("2026-10-09T00:00:00+00:00 PARALLEL RESULT 11 candidate RMSE=1\n", encoding="utf-16")
    report = snapshot([run], context, tmp_path, "abc123")
    assert not report["warnings"]
    assert report["entries"][0]["champion"]
    assert report["entries"][0]["mean_mae"] == 1
    assert len(report["completed_fold_experiments"]) == 3
    (run / "experiment-ledger.json").write_text(json.dumps([dict(e, mae=2) for e in ledger]), encoding="utf-8")
    broken = snapshot([run], context, tmp_path, "abc123")
    assert not broken["entries"] and broken["warnings"]

# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import pandas as pd

from leaderboard_monitor import diagnostics, discover_runs
from pipeline import ID, TARGET, TIME, write_json


def test_completed_diagnostic_is_visible_but_never_ranked(tmp_path: Path) -> None:
    write_json(tmp_path / "selection.json", {"selected": "fixed_ensemble"})
    write_json(tmp_path / "leaderboard-provenance.json", {"git_commit": "abc123"})
    write_json(tmp_path / "environment.json", {"source_hashes": {"model.py": "digest"}})
    context = pd.DataFrame({ID: [1], TARGET: [10.], TIME: pd.to_datetime(["2025-02-01"], utc=True),
        "ADEP_mvt": ["AAA"], "unseen_aircraft": [True], "unseen_airport_stand": [True]})
    predicted = context[[ID, TARGET, TIME, "ADEP_mvt"]].copy()
    predicted["prediction"] = 4010.
    predicted.to_parquet(tmp_path / "fixed-evaluation-02.parquet", index=False)
    rows = diagnostics(tmp_path, context)
    assert len(rows) == 1
    assert rows[0]["mean_mae"] == rows[0]["worst_decile_mae"] == 4000
    assert rows[0]["unseen_aircraft_mae"] == rows[0]["unseen_airport_stand_mae"] == 4000
    assert rows[0]["catastrophic_error_rate"] == 1
    assert rows[0]["rank"] is None and not rows[0]["champion"]
    assert not rows[0]["used_for_model_selection"]


def test_discovery_excludes_official_and_unreviewed_feature_scopes(tmp_path: Path) -> None:
    for name, months, feature in [("valid", [9, 10, 11], "airport"),
                                 ("official", [1, 7], "airport"),
                                 ("leaky", [9, 10, 11], "actual_outcome")]:
        run = tmp_path / name
        run.mkdir()
        write_json(run / "plan.json", {"selection_months": months, "models": {"m": {"kind": "lightgbm"}}})
        write_json(run / "experiment-ledger.json", [{"experiment": "forward-09-m", "features": [feature]}])
    assert discover_runs(tmp_path, {"airport"}) == [(tmp_path / "valid").resolve()]

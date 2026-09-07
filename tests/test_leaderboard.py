# SPDX-License-Identifier: GPL-3.0-only
from leaderboard import best_per_team


def test_rankings_use_best_submission_per_team() -> None:
    rows = [
        {"teamId": "a", "teamName": "alpha", "score": 10.},
        {"teamId": "a", "teamName": "alpha", "score": 20.},
        {"teamId": "b", "teamName": "beta", "score": 15.},
        {"teamId": "c", "teamName": "gamma", "score": None},
    ]
    assert [(row["teamId"], row["score"]) for row in best_per_team(rows)] == [("a", 10.), ("b", 15.)]

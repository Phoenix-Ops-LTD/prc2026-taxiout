# SPDX-License-Identifier: GPL-3.0-only
"""Read every public API page and rank each team's best official score."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

URL = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"


def fetch() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        url = URL + ("?cursor=" + urllib.parse.quote(cursor, safe="") if cursor else "")
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
        rows.extend(payload["items"])
        cursor = payload.get("nextCursor")
        if not cursor:
            return rows
        if cursor in seen:
            raise ValueError("API cursor repeated; ranking would be incomplete")
        seen.add(cursor)


def best_per_team(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("score") is None:
            continue
        team = row["teamId"]
        if team not in best or row["score"] < best[team]["score"]:
            best[team] = row
    return sorted(best.values(), key=lambda row: (row["score"], row["teamName"]))


def main() -> None:
    rows = fetch()
    teams = best_per_team(rows)
    print("Checked UTC:", datetime.now(timezone.utc).isoformat())
    print(f"{len(rows)} submissions / {len(teams)} scored teams; best RMSE per team")
    for row in teams:
        rank = 1 + sum(other["score"] < row["score"] for other in teams)
        print(f"{rank:3}  {row['teamName']:28}  {row['score']:10.4f}  {row['filename']}")


if __name__ == "__main__":
    main()

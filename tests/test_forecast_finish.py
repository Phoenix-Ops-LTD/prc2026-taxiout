# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import numpy as np
import pytest

from forecast_features import schedule_features
from forecast_finish import airport_holdouts
from forecast_model import ModelSpec
from forecast_novel_guard import NovelGuardedForecast
from pipeline import TARGET
from test_forecast import fixture


@pytest.mark.parametrize("kind", ["catboost", "lightgbm"])
def test_analytic_airport_rule_equals_native_unknown_airport_predictions(kind: str, tmp_path: Path) -> None:
    raw = fixture().assign(STAND_mvt="1")
    other = raw["ADEP_mvt"].eq("BBB")
    x = schedule_features(raw, raw)
    spec = ModelSpec(kind=kind, iterations=5, depth=2, threads=1)  # type: ignore[arg-type]
    model = NovelGuardedForecast(spec).fit(x.loc[other], raw.loc[other, TARGET])
    held = x.loc[~other].copy()
    # Both known and novel aircraft must reduce to the exact same global fallback.
    held.iloc[0, held.columns.get_loc("AIRCRAFT_TYPE_mvt")] = "NOVEL"
    expected = float(raw.loc[other, TARGET].mean())
    np.testing.assert_array_equal(model.predict(held), np.full(len(held), expected))
    report = airport_holdouts(raw, {"m": spec}, {"m": 1.0}, tmp_path)
    assert set(report) == {"AAA", "BBB"}
    assert all(r["unseen_airport_stand_rows"] == r["n"] for r in report.values())
    assert all(r["unseen_airport_stand_mae"] == r["mae"] for r in report.values())

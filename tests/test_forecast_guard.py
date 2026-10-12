# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_features import schedule_features
from forecast_guard import GuardedForecast, load_model
from forecast_guard_finalize import choose
from forecast_model import FittedForecast, ModelSpec
from forecast_novel_guard import NovelGuardedForecast, load_model as load_novel_model
from pipeline import TARGET, TIME, write_json
from taxiout import ScheduleContext, TaxiOutInput, TaxiOutPredictor
from test_forecast import fixture


def test_guard_preserves_known_predictions_and_uses_training_only_hierarchy(tmp_path: Path) -> None:
    raw = fixture()
    x = schedule_features(raw, raw)
    model = GuardedForecast(ModelSpec(kind="lightgbm", iterations=5, threads=1)).fit(x, raw[TARGET])
    original = FittedForecast.predict(model, x)
    np.testing.assert_array_equal(model.predict(x), original)
    unknown = x.iloc[[0, 1, 2]].copy()
    unknown["AIRCRAFT_TYPE_mvt"] = ["NEW", "UNKNOWN", "NEW"]
    unknown.loc[unknown.index[-1], "ADEP_mvt"] = "UNSEEN"
    expected = [raw.loc[raw["ADEP_mvt"].eq("AAA"), TARGET].mean(),
                raw.loc[raw["ADEP_mvt"].eq("BBB"), TARGET].mean(), raw[TARGET].mean()]
    np.testing.assert_allclose(model.predict(unknown), expected)
    model.save(tmp_path / "guarded")
    np.testing.assert_array_equal(load_model(tmp_path / "guarded").predict(unknown), model.predict(unknown))
    write_json(tmp_path / "model-manifest.json", {"weights": {"guarded": 1}, "data_class": "synthetic",
        "model_version": "guard-test/1", "interval_radius_seconds": 60})
    result = TaxiOutPredictor(tmp_path).predict(TaxiOutInput(airport="AAA", aircraft="NEW",
        scheduledTimeUtc="2026-02-01T12:00:00Z"), ScheduleContext(knownAtUtc="2026-02-01T11:00:00Z"))
    assert result.predictionSeconds == expected[0]
    assert "UNSEEN_AIRCRAFT_BASELINE_FALLBACK" in result.reasonCodes


def test_guard_selection_rejects_a_worse_fold_despite_pooled_gain() -> None:
    frame = pd.DataFrame({TARGET: [0.] * 3, TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
        "lightgbm": [10., 10., 10.], "lightgbm_guarded": [1., 1., 12.]})
    weights, report = choose(frame, ["lightgbm", "lightgbm_guarded"])
    assert weights == {"lightgbm": 1.}
    assert not report["scores"]["lightgbm_guarded"]["eligible"]
    frame["lightgbm_guarded"] = [1., 1., 1.]
    weights, _ = choose(frame, ["lightgbm", "lightgbm_guarded"])
    assert weights == {"lightgbm_guarded": 1.}


def test_novel_guard_preserves_observed_missing_and_native_reload(tmp_path: Path) -> None:
    raw = fixture()
    raw.loc[raw.index[:25], "AIRCRAFT_TYPE_mvt"] = None
    x = schedule_features(raw, raw)
    model = NovelGuardedForecast(ModelSpec(kind="lightgbm", iterations=5, threads=1)).fit(x, raw[TARGET])
    assert model.missing_aircraft_seen_in_fit
    np.testing.assert_array_equal(model.predict(x), FittedForecast.predict(model, x))
    model.save(tmp_path / "novel")
    loaded = load_novel_model(tmp_path / "novel")
    assert isinstance(loaded, NovelGuardedForecast) and loaded.missing_aircraft_seen_in_fit
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    write_json(tmp_path / "model-manifest.json", {"weights": {"novel": 1}, "data_class": "synthetic",
        "model_version": "novel-test/1", "interval_radius_seconds": 60})
    predictor = TaxiOutPredictor(tmp_path)
    context = ScheduleContext(knownAtUtc="2026-02-01T11:00:00Z")
    observed_missing = predictor.predict(TaxiOutInput(airport="AAA", aircraft="UNKNOWN", scheduledTimeUtc="2026-02-01T12:00:00Z"), context)
    assert "UNSEEN_AIRCRAFT_BASELINE_FALLBACK" not in observed_missing.reasonCodes
    novel = predictor.predict(TaxiOutInput(airport="AAA", aircraft="NEW", scheduledTimeUtc="2026-02-01T12:00:00Z"), context)
    assert "UNSEEN_AIRCRAFT_BASELINE_FALLBACK" in novel.reasonCodes

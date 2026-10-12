# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
from unittest.mock import Mock

import pytest

from forecast_features import schedule_features
from forecast_model import FittedForecast, ModelSpec
from pipeline import TARGET, write_json
from taxiout import ScheduleContext, ScheduleMovement, TaxiOutInput, TaxiOutPredictor
from test_forecast import fixture


def test_generic_context_exact_self_removal_and_utc_contract(tmp_path: Path) -> None:
    raw = fixture()
    model = FittedForecast(ModelSpec(kind="lightgbm", iterations=5, threads=1)).fit(schedule_features(raw, raw), raw[TARGET])
    model.save(tmp_path / "lightgbm")
    write_json(tmp_path / "model-manifest.json", {"weights": {"lightgbm": 1}, "data_class": "synthetic",
        "model_version": "portable-test/1", "interval_radius_seconds": 60})
    taxiOut = TaxiOutPredictor(tmp_path)
    request = TaxiOutInput(airport="AAA", aircraft="A320", scheduledTimeUtc="2026-02-01T12:00:00Z", movementId="self")
    context = ScheduleContext(knownAtUtc="2026-02-01T11:00:00Z", movements=[
        ScheduleMovement(movementId="self", airport="AAA", phase="DEP", scheduledTimeUtc=request.scheduledTimeUtc)])
    result = taxiOut.predict(request, context)
    assert result == taxiOut.predict(request, context.model_copy(update={"movements": []}))
    assert result.modelVersion == "portable-test/1"
    assert result.range[0] <= result.predictionSeconds <= result.range[1]
    with pytest.raises(ValueError, match="movementId"):
        taxiOut.predict(request.model_copy(update={"movementId": None}), context)
    with pytest.raises(ValueError, match="unique"):
        taxiOut.predict(request, context.model_copy(update={"movements": context.movements * 2}))
    with pytest.raises(ValueError):
        ScheduleContext(knownAtUtc="2026-02-01T11:00:00")
    # A caller's arrival ID must never collide with the adapter's query identity.
    native = taxiOut.adapter.models["lightgbm"]
    spy = Mock(wraps=native.predict)
    native.predict = spy
    arrival = ScheduleMovement(movementId="__query__", airport="AAA", phase="ARR", scheduledTimeUtc=request.scheduledTimeUtc)
    taxiOut.predict(request, context.model_copy(update={"movements": [arrival]}))
    scored_features = spy.call_args.args[0]
    assert scored_features["scheduled_arr_pm5m"].iloc[0] == 1

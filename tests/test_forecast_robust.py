# SPDX-License-Identifier: GPL-3.0-only
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

import forecast_robust as robust
from forecast_features import schedule_features
from forecast_model import ModelSpec
from forecast_novel_guard import NovelGuardedForecast, load_model
from pipeline import TARGET, write_json
from leaderboard_monitor import measured_execution
from test_forecast import fixture


def test_fit_cap_does_not_modify_evaluation_labels() -> None:
    labels = pd.Series([10., 7200., 131167.])
    original = labels.copy()
    assert robust.capped_targets(labels).tolist() == [10., 7200., 7200.]
    pd.testing.assert_series_equal(labels, original)


@pytest.mark.parametrize("labels", [[-1.], [float("nan")], []])
def test_fit_cap_rejects_invalid_fitting_targets(labels: list[float]) -> None:
    with pytest.raises(ValueError):
        robust.capped_targets(pd.Series(labels, dtype=float))


def test_budget_enforces_deadline_reserve_and_elapsed_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robust, "DEADLINE", datetime.now(timezone.utc) + timedelta(hours=3))
    robust.require_budget(time.monotonic())
    with pytest.raises(ValueError, match="budget"):
        robust.require_budget(time.monotonic() - 3601)
    monkeypatch.setattr(robust, "DEADLINE", datetime(2000, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="reserve"):
        robust.require_budget(time.monotonic())


def test_saved_cap_model_keeps_fit_policy_and_exact_native_predictions(tmp_path: Path) -> None:
    raw = fixture()
    raw.loc[0, TARGET] = 131167.
    x = schedule_features(raw, raw)
    model = NovelGuardedForecast(ModelSpec(kind="lightgbm", iterations=5, threads=1)).fit(x, robust.capped_targets(raw[TARGET]))
    robust.save_model(model, tmp_path / "model")
    np.testing.assert_array_equal(model.predict(x), load_model(tmp_path / "model").predict(x))
    assert model.baseline["mean"] < raw[TARGET].mean()
    assert raw.loc[0, TARGET] == 131167.


def test_cpu_clocks_and_fit_cap_are_retained_without_promoting_a_partial(tmp_path: Path) -> None:
    run = tmp_path / "cpu"
    run.mkdir()
    write_json(run / "plan.json", {"execution_unit": "CPU", "training_overrides": {"m": {"fit_target_cap_seconds": 7200}}})
    write_json(run / "experiment-ledger.json", [{"experiment": "forward-09-m", "model": "m", "validation_month": 9,
        "training_time_seconds": 4., "inference_time_seconds": .2, "completed_at_utc": "now"}])
    entry = {"experiment_id": "cpu/m", "model": "m", "validation_folds": ["2025-09"], "rank": None}
    measured_execution({"entries": [entry], "completed_fold_experiments": []}, [run])
    assert entry["rank"] is None and entry["device"] == "CPU"
    assert entry["training_time_seconds"] == 4. and entry["inference_time_seconds"] == .2
    assert entry["parameters"]["fit_target_cap_seconds"] == 7200

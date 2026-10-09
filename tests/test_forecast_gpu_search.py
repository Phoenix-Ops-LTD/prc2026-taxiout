# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import pandas as pd
import pytest

from forecast_gpu_search import candidates, overrides, validate_inputs
from forecast_gpu_prepare import prepare
from leaderboard_monitor import measured_execution
from pipeline import ID, TARGET, TIME, write_json


def test_gpu_search_rejects_locked_month_and_observed_features() -> None:
    data = pd.DataFrame({ID: [1], TARGET: [10.], TIME: pd.to_datetime(["2025-11-01"], utc=True)})
    x = pd.DataFrame({"hour_utc": [12.]})
    validate_inputs(data, x)
    with pytest.raises(ValueError, match="outside"):
        validate_inputs(data, x.assign(MVT_TIME_UTC_mvt=123))
    with pytest.raises(ValueError, match="December"):
        validate_inputs(data.assign(**{TIME: pd.to_datetime(["2025-12-01"], utc=True)}), x)


def test_gpu_timing_annotation_preserves_partial_rank_and_legacy_metrics(tmp_path: Path) -> None:
    run = tmp_path / "gpu"
    run.mkdir()
    write_json(run / "plan.json", {"execution_unit": "GPU", "training_overrides": {"m": {"task_type": "GPU"}}})
    write_json(run / "experiment-ledger.json", [{"experiment": "forward-09-m", "model": "m", "validation_month": 9,
        "training_time_seconds": 4., "inference_time_seconds": .2, "completed_at_utc": "2026-10-09T16:00:00Z"}])
    entry = {"experiment_id": "gpu/m", "model": "m", "validation_folds": ["2025-09"], "rank": None, "mean_mae": 12.}
    report = {"entries": [entry], "completed_fold_experiments": []}
    measured_execution(report, [run])
    assert entry["rank"] is None and entry["mean_mae"] == 12.
    assert entry["training_time_seconds"] == 4. and entry["inference_time_seconds"] == .2
    assert entry["device"] == "GPU RTX4090 / CPU inference"


def test_gpu_candidates_are_fixed_and_native_spec_is_portable() -> None:
    specs = candidates()
    assert len(specs) == 3
    assert all(spec.kind == "catboost" and spec.iterations >= 3000 for spec in specs.values())
    assert all(overrides(name)["gpu_ram_part"] == .35 and overrides(name)["max_ctr_complexity"] == 1 for name in specs)


def test_preparation_rejects_changed_feature_cache_before_reading_training(tmp_path: Path) -> None:
    feature_run = tmp_path / "base"
    search = tmp_path / "search"
    feature_run.mkdir()
    search.mkdir()
    (feature_run / "features.parquet").write_bytes(b"changed")
    write_json(search / "plan.json", {"feature_sha256": "different"})
    with pytest.raises(ValueError, match="Feature cache"):
        prepare(tmp_path / "absent-private-data", feature_run, search, tmp_path / "output")
    assert not (tmp_path / "output").exists()

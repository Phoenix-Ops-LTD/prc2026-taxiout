# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import ID, TARGET, RunConfig, departures, features, predict, rmse, train, validate_submission


def fixture() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", "2025-12-31", periods=300, tz="UTC")
    return pd.DataFrame({ID: np.arange(300), "PHASE_mvt": "DEP", "SCHED_TIME_UTC_mvt": dates, "ADEP_mvt": ["LFPG", "EGLL"] * 150, "AIRCRAFT_TYPE_mvt": ["A320", "B738"] * 150, TARGET: [600 + (i % 2) * 200 + (i % 7) * 15 for i in range(300)]})


def test_features_exclude_actual_times_and_targets() -> None:
    frame = fixture()
    original = features(frame)
    for column in [TARGET, "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt", "ARVT_3_flt"]:
        frame[column] = 999999
    pd.testing.assert_frame_equal(original, features(frame))


def test_template_guards() -> None:
    template = pd.DataFrame({ID: [8, 3], TARGET: [np.nan, np.nan]})
    valid = template.copy()
    valid[TARGET] = [650.0, 710.0]
    validate_submission(template, valid)
    for invalid in [valid.iloc[::-1], valid.iloc[:1], pd.concat([valid, valid]), valid.assign(TAXITIME_SEC_mvt=[np.nan, 20]), valid.assign(TAXITIME_SEC_mvt=[-1, 20]), valid.assign(TAXITIME_SEC_mvt=[np.inf, 20]), valid.assign(extra=1)]:
        with pytest.raises(ValueError):
            validate_submission(template, invalid)


def test_temporal_holdout_and_reproducible_end_to_end(tmp_path: Path) -> None:
    data = fixture()
    training = tmp_path / "training.parquet"
    data.to_parquet(training)
    config = RunConfig(iterations=30, depth=3)
    first = train([training], tmp_path / "first", config)
    second = train([training], tmp_path / "second", config)
    assert first["rmse_seconds"] == second["rmse_seconds"]
    assert first["status"] == "SYNTHETIC_TEST_ONLY"
    ranking = data.iloc[:12].copy()
    ranking[ID] += 1000
    ranking[TARGET] = np.nan
    ranking_path = tmp_path / "ranking.parquet"
    ranking.to_parquet(ranking_path)
    template = ranking[[ID, TARGET]].iloc[::-1]
    template_path = tmp_path / "submitting.parquet"
    template.to_parquet(template_path)
    output = tmp_path / "zestful-fountain_v1.parquet"
    manifest = predict(tmp_path / "first", ranking_path, template_path, output)
    assert manifest["data_class"] == "synthetic"
    validate_submission(template, pd.read_parquet(output))
    with pytest.raises(ValueError):
        predict(tmp_path / "first", ranking_path, template_path, output)


def test_invalid_training_and_metric() -> None:
    with pytest.raises(ValueError):
        departures(fixture().assign(TAXITIME_SEC_mvt=-1), training=True)
    with pytest.raises(ValueError):
        departures(pd.concat([fixture(), fixture()]))
    assert rmse([0, 3], [0, 0]) == pytest.approx(np.sqrt(4.5))


def test_challenge_context_excludes_block_time_proxies() -> None:
    frame = fixture().assign(RUNWAY_mvt="09", STAND_mvt="A12")
    frame["MVT_TIME_UTC_mvt"] = frame["SCHED_TIME_UTC_mvt"] + pd.Timedelta(minutes=25)
    original = features(frame, "challenge_context")
    assert "airport_STAND_mvt" in original
    for column in [TARGET, "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt"]:
        frame[column] = 888888
    pd.testing.assert_frame_equal(original, features(frame, "challenge_context"))
    with pytest.raises(ValueError):
        features(frame, "unknown")


def test_cleaning_keeps_negative_validation_targets_visible(tmp_path: Path) -> None:
    frame = fixture()
    frame.loc[0, TARGET] = -3
    frame.loc[299, TARGET] = -7
    source = tmp_path / "training.parquet"
    frame.to_parquet(source)
    with pytest.raises(ValueError, match="Negative taxi targets"):
        train([source], tmp_path / "rejected", RunConfig(iterations=10, depth=2))
    report = train([source], tmp_path / "cleaned", RunConfig(iterations=10, depth=2, negative_target_policy="drop"))
    assert report["negative_targets_excluded_from_fit"] == 1
    assert report["negative_targets_retained_in_validation"] == 1
    assert report["final_refit_departures"] == 298


def test_validation_only_run_cannot_be_submitted(tmp_path: Path) -> None:
    source = tmp_path / "training.parquet"
    fixture().to_parquet(source)
    run = tmp_path / "validation"
    report = train([source], run, RunConfig(iterations=10, depth=2, refit_full=False))
    assert report["submission_ready"] is False
    assert report["final_refit_departures"] == 0
    assert not (run / "model.cbm").exists()
    with pytest.raises(ValueError, match="Validation-only"):
        predict(run, source, source, tmp_path / "invalid.parquet")

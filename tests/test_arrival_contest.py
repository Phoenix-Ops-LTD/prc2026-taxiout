# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from arrival_contest import ArrivalConfig, blend, matrix, predict
from arrival_features import completed_arrival_features
from arrival_specialist import SEEDS, SpecialistConfig, matrix as specialist_matrix
from carrier_contest import specialist_scope
from ensemble_contest import categorical
from pipeline import ID, TARGET, sha256, write_json
from traffic_features import nm_schedule_fallback_baseline
from test_duration_contest import sample


def observations() -> pd.DataFrame:
    return pd.DataFrame({"PHASE_mvt": ["ARR", "ARR", "DEP"], "ADES_mvt": ["EDDF", "EDDF", "ZZZZ"],
        "ADEP_mvt": ["ZZZZ", "ZZZZ", "EDDF"], "RUNWAY_mvt": ["07C"] * 3, "STAND_mvt": ["A1"] * 3,
        "MVT_TIME_UTC_mvt": ["2025-01-01T11:45:00Z", "2025-01-01T11:55:00Z", "2025-01-01T12:00:00Z"],
        "BLOCK_TIME_UTC_mvt": ["2025-01-01T11:50:00Z", "2025-01-01T12:05:00Z", "FORBIDDEN"],
        TARGET: [300., 600., "FORBIDDEN"], "AIRCRAFT_TYPE_mvt": ["A320"] * 3,
        "FLIGHT_mvt": ["ABC1", "ABC2", "ABC3"]}, index=[8, 9, 3])


def test_only_completed_arrivals_enter_context() -> None:
    raw = observations()
    dep = raw.loc[[3]]
    actual = completed_arrival_features(dep, raw)
    assert actual.shape == (1, 26)
    assert actual.loc[3, "completed_arrival_airport_15m_count"] == 1
    assert actual.loc[3, "completed_arrival_airport_15m_mean"] == 300
    assert actual.loc[3, "completed_arrival_stand_last_age_seconds"] == 600
    changed = raw.copy()
    changed.loc[3, TARGET] = 999999
    changed.loc[3, "BLOCK_TIME_UTC_mvt"] = "2099-01-01T00:00:00Z"
    changed.loc[9, TARGET] = 1800
    changed.loc[9, "BLOCK_TIME_UTC_mvt"] = "2025-01-01T12:00:00Z"
    pd.testing.assert_frame_equal(actual, completed_arrival_features(dep, changed))
    pd.testing.assert_frame_equal(actual, completed_arrival_features(dep, raw.iloc[::-1]))


def test_ties_missing_arrivals_and_missing_departure_clock() -> None:
    raw = observations()
    duplicate = raw.loc[[8]].copy()
    duplicate.index = [12]
    duplicate[TARGET] = 400.
    tied = pd.concat([raw, duplicate])
    actual = completed_arrival_features(raw.loc[[3]], tied)
    assert actual.loc[3, "completed_arrival_airport_last_taxi_seconds"] == 350
    assert pd.isna(actual.loc[3, "completed_arrival_stand_last_age_seconds"])
    pd.testing.assert_frame_equal(actual, completed_arrival_features(raw.loc[[3]], tied.iloc[::-1]))
    empty = completed_arrival_features(raw.loc[[3]], raw.loc[[3]])
    assert empty.completed_arrival_airport_15m_count.eq(0).all()
    with pytest.raises(ValueError, match="complete"):
        completed_arrival_features(raw.loc[[3]].assign(MVT_TIME_UTC_mvt=pd.NaT), raw)


def test_matrix_never_reads_departure_targets_or_block_times() -> None:
    raw = sample().assign(BLOCK_TIME_UTC_mvt=pd.NaT)
    dep, x = matrix(raw)
    _, changed = matrix(raw.assign(TAXITIME_SEC_mvt=-999999, BLOCK_TIME_UTC_mvt="FORBIDDEN",
        MVT_ID_mvt=raw[ID] + 10000, FLIGHT_ID_mvt=777))
    assert len(x.columns) == 123 and x.index.equals(dep.index)
    pd.testing.assert_frame_equal(x, changed)


def test_fixed_blend_and_configuration_guards() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG"], "missing_IOBT_flt": [1, 1]}, index=[4, 2])
    base = pd.Series([123.456789, 200.], index=x.index)
    extra = pd.Series([999., 400.], index=x.index)
    assert blend(x, base, extra).tolist() == [123.456789, 250.]
    for invalid, message in [(extra.iloc[::-1], "alignment"), (extra * np.nan, "finite"), (-extra, "nonnegative")]:
        with pytest.raises(ValueError, match=message):
            blend(x, base, invalid)
    for config in [{"data_permission_ref": " "}, {"data_permission_ref": "test", "trees": 0},
                   {"data_permission_ref": "test", "model_version": "invalid"}]:
        with pytest.raises(ValidationError):
            ArrivalConfig.model_validate(config)


def test_prediction_round_trip_blends_correct_scope_and_checks_digests(tmp_path: Path) -> None:
    raw = sample().assign(BLOCK_TIME_UTC_mvt=pd.NaT)
    raw.loc[:7, "ADEP_mvt"] = "LIRF"
    raw.loc[:7, "IOBT_flt"] = pd.NaT
    dep, x = matrix(raw)
    scope = specialist_scope(x)
    model = lgb.LGBMRegressor(n_estimators=3, num_leaves=4, verbosity=-1, n_jobs=1)
    model.fit(categorical(x), np.arange(len(x), dtype=float))
    run = tmp_path / "model"
    run.mkdir()
    model_path = run / "model.txt"
    model.booster_.save_model(str(model_path))
    config = ArrivalConfig(trees=3, data_permission_ref="synthetic-test")
    write_json(run / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE", "config": config.model_dump(),
        "features": list(x), "model_sha256": sha256(model_path)})
    expert_run = tmp_path / "experts"
    expert_run.mkdir()
    _, sx = specialist_matrix(raw)
    experts = []
    predictions = []
    for seed in SEEDS:
        expert_model = CatBoostRegressor(iterations=3, depth=2, random_seed=seed,
            thread_count=1, allow_writing_files=False, verbose=False)
        expert_model.fit(sx, np.arange(len(sx), dtype=float) * 10,
            cat_features=list(sx.select_dtypes(include=["str", "object"]).columns))
        expert_path = expert_run / f"seed-{seed}.cbm"
        expert_model.save_model(str(expert_path))
        experts.append({"seed": seed, "file": expert_path.name, "sha256": sha256(expert_path)})
        predictions.append(np.maximum(0, expert_model.predict(sx) + nm_schedule_fallback_baseline(sx)))
    write_json(expert_run / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE",
        "config": SpecialistConfig(iterations=3, data_permission_ref="synthetic-test").model_dump(),
        "features": list(sx), "experts": experts})
    ranking = tmp_path / "ranking.parquet"
    raw[TARGET] = np.nan
    raw.to_parquet(ranking, index=False)
    template = tmp_path / "template.parquet"
    baseline = tmp_path / "baseline.parquet"
    original = dep[[ID, TARGET]].copy().iloc[::-1].reset_index(drop=True)
    original[TARGET] = np.arange(len(original), dtype=float) + 123.456789
    original.to_parquet(template, index=False)
    original.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {"model_version": "prc2026-specialist-bagging/6.0.0",
        "submission_sha256": sha256(baseline), "ranking_sha256": sha256(ranking), "template_sha256": sha256(template)})
    output = tmp_path / "candidate.parquet"
    predict(run, baseline, ranking, template, output, "synthetic-test", expert_run)
    actual = pd.read_parquet(output)
    special = original[ID].isin(dep.loc[scope, ID])
    expected_expert = pd.Series(np.mean(predictions, axis=0), index=dep.loc[scope, ID])
    np.testing.assert_array_equal(actual.loc[special, TARGET],
        .5 * original.loc[special, TARGET] + .5 * original.loc[special, ID].map(expected_expert))
    assert not actual.loc[~special, TARGET].equals(original.loc[~special, TARGET])
    with pytest.raises(ValueError, match="fresh"):
        predict(run, baseline, ranking, template, output, "synthetic-test", expert_run)
    with model_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        predict(run, baseline, ranking, template, tmp_path / "rejected.parquet", "synthetic-test", expert_run)

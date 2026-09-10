# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from carrier_contest import matrix, specialist_scope
from pipeline import ID, TARGET, sha256, write_json
from specialist_ensemble import BaggingConfig, average_specialist, predict
from test_duration_contest import sample


def test_seed_ensemble_only_changes_its_declared_scope() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LFPG", "LIRF", "LIRF", "LIRF"],
        "missing_IOBT_flt": [1, 1, 0, 1]}, index=[7, 3, 9, 2])
    base = pd.Series([123.456789, 100., 987.654321, 200.], index=x.index)
    experts = np.array([[0., 400.], [200., 0.], [300., 800.], [400., 100.]], dtype=float)
    actual = average_specialist(x, base, experts)
    assert actual.loc[[3, 2]].tolist() == [200., 300.]
    pd.testing.assert_series_equal(actual.loc[[7, 9]], base.loc[[7, 9]])
    # Reordering the full frame must retain the correct expert-to-scope mapping.
    pd.testing.assert_series_equal(actual.iloc[::-1], average_specialist(x.iloc[::-1], base.iloc[::-1], experts[:, ::-1]))
    with pytest.raises(ValueError, match="alignment"):
        average_specialist(x, base.iloc[::-1], experts)
    with pytest.raises(ValueError, match="shape"):
        average_specialist(x, base, experts[:, :1])
    with pytest.raises(ValueError, match="finite"):
        average_specialist(x, base, experts * np.nan)
    with pytest.raises(ValueError, match="nonnegative"):
        average_specialist(x, base, -experts)


def test_no_specialist_rows_keeps_baseline_exact() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LFPG"], "missing_IOBT_flt": [1]}, index=[4])
    base = pd.Series([42.], index=x.index)
    pd.testing.assert_series_equal(base, average_specialist(x, base, np.empty((4, 0))))


def test_bagging_configuration_rejects_seed_reuse_and_missing_authorization() -> None:
    for seeds in [(), (20260907,), (11, 11), (-1,)]:
        with pytest.raises(ValidationError):
            BaggingConfig(additional_seeds=seeds, data_permission_ref="synthetic-test")
    with pytest.raises(ValidationError):
        BaggingConfig(data_permission_ref=" ")
    assert len(BaggingConfig(data_permission_ref="synthetic-test").additional_seeds) == 4


def test_prediction_round_trip_preserves_other_rows_and_checks_model_digest(tmp_path) -> None:
    raw = sample()
    raw.loc[raw.index[:8], "ADEP_mvt"] = "LIRF"
    raw.loc[raw.index[:8], "IOBT_flt"] = pd.NaT
    dep, x = matrix(raw)
    scope = specialist_scope(x)
    categories = list(x.select_dtypes(include=["object", "str"]).columns)
    model = CatBoostRegressor(iterations=3, depth=2, random_seed=8, thread_count=1,
        allow_writing_files=False, verbose=False)
    model.fit(x.loc[scope], np.arange(int(scope.sum()), dtype=float) * 10, cat_features=categories)
    run = tmp_path / "models"
    run.mkdir()
    model_path = run / "seed-8.cbm"
    model.save_model(str(model_path))
    config = BaggingConfig(additional_seeds=(8,), iterations=3, data_permission_ref="synthetic-test")
    write_json(run / "report.json", {"status": "FULL_FIT_NOT_OFFICIAL_SCORE", "config": config.model_dump(),
        "features": list(x), "experts": [{"seed": 8, "file": model_path.name, "sha256": sha256(model_path)}]})
    ranking = tmp_path / "ranking.parquet"
    raw.to_parquet(ranking, index=False)
    template = tmp_path / "template.parquet"
    baseline = tmp_path / "baseline.parquet"
    original = dep[[ID, TARGET]].copy().reset_index(drop=True)
    original[TARGET] = np.arange(len(original), dtype=float) + 123.456789
    original.to_parquet(template, index=False)
    original.to_parquet(baseline, index=False)
    write_json(baseline.with_suffix(".manifest.json"), {"model_version": "prc2026-duration-blend/5.0.0",
        "submission_sha256": sha256(baseline), "ranking_sha256": sha256(ranking), "template_sha256": sha256(template)})
    output = tmp_path / "candidate.parquet"
    predict(run, baseline, ranking, template, output, "synthetic-test")
    actual = pd.read_parquet(output)
    outside = ~original[ID].isin(dep.loc[scope, ID])
    pd.testing.assert_frame_equal(actual.loc[outside], original.loc[outside])
    assert not actual.loc[~outside, TARGET].equals(original.loc[~outside, TARGET])
    with model_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        predict(run, baseline, ranking, template, tmp_path / "rejected.parquet", "synthetic-test")

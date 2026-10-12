# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pydantic import ValidationError

from forecast_features import CATEGORIES
from forecast_gpu_airport import AirportCatBoostForecast, SpecialistSpec, candidates
from forecast_model import ModelSpec
from forecast_novel_guard import load_model


def test_airport_native_roundtrip_and_unknown_fallback(tmp_path: Path) -> None:
    x = pd.DataFrame({c: ["KNOWN"] * 60 for c in CATEGORIES})
    x["ADEP_mvt"] = "TEST"
    x["hour_utc"] = np.arange(60) % 24
    y = pd.Series(np.arange(60, dtype=float) + 500.)
    model = AirportCatBoostForecast(ModelSpec(kind="catboost", iterations=3, depth=2, threads=1, demand=False)).fit(x, y)
    local = CatBoostRegressor(iterations=3, depth=2, verbose=False, thread_count=1, allow_writing_files=False)
    local.fit(model.encode(x), y + 100, cat_features=CATEGORIES)
    model.specialists["TEST"] = local
    known = model.predict(x)
    unknown = x.iloc[:1].assign(ADEP_mvt="NOVEL", AIRCRAFT_TYPE_mvt="NOVEL")
    expected = model.predict(unknown)
    root = tmp_path / "model"
    model.save(root)
    saved = AirportCatBoostForecast.load(root)
    np.testing.assert_array_equal(known, saved.predict(x))
    np.testing.assert_array_equal(expected, saved.predict(unknown))
    assert expected[0] == float(y.mean())
    with pytest.raises(ValueError, match="Unsupported"):
        load_model(root)
    (root / "airport-0.cbm").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        AirportCatBoostForecast.load(root)


def test_specialist_weights_and_candidate_pool_are_frozen() -> None:
    assert len(candidates()) == 3
    assert {s.minimum_airport_rows for s in candidates().values()} == {10000}
    assert {s.local_weight for s in candidates().values()} == {.5}
    with pytest.raises(ValidationError):
        SpecialistSpec(global_model="base", iterations=10, depth=2, demand=False, local_weight=.7)

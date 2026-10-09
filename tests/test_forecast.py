# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_features import CATEGORIES, EXCLUDED, clean_departures, metrics, purged_split, schedule_features
from forecast_model import FittedForecast, ModelSpec, PredictInput, TaxiOut
from forecast_run import make_submission
from pipeline import ID, PHASE, TARGET, TIME, write_json


def fixture() -> pd.DataFrame:
    n = 360
    return pd.DataFrame({ID: np.arange(n, dtype=float), PHASE: "DEP",
                         TIME: pd.date_range("2025-01-01", "2025-12-31", periods=n, tz="UTC"),
                         "ADEP_mvt": ["AAA", "BBB"] * (n // 2), "ADES_mvt": "ZZZ",
                         "AIRCRAFT_TYPE_mvt": ["A320", "B738"] * (n // 2),
                         "FLIGHT_mvt": "AB123", TARGET: 600 + np.arange(n) % 17 * 5,
                         "FLIGHT_ID_mvt": np.arange(n)})


def test_schedule_exact_oracle_arrival_identity_self_and_boundaries() -> None:
    base = pd.Timestamp("2025-06-01T12:00:00Z")
    raw = pd.DataFrame({ID: [1, 2, 3, 4, 5], PHASE: ["DEP", "DEP", "ARR", "ARR", "DEP"],
                        TIME: [base, base + pd.Timedelta(minutes=5), base - pd.Timedelta(minutes=5), base, base],
                        "ADEP_mvt": ["AAA", "AAA", "ZZZ", "AAA", "BBB"],
                        "ADES_mvt": ["ZZZ", "ZZZ", "AAA", "BBB", "AAA"],
                        "AIRCRAFT_TYPE_mvt": "A320"})
    x = schedule_features(clean_departures(raw), raw)
    assert x.loc[0, "scheduled_dep_pm5m"] == 1
    assert x.loc[0, "scheduled_arr_pm5m"] == 1
    assert x.loc[0, "scheduled_arr_previous30m"] == 1
    assert x.loc[0, "scheduled_dep_previous30m"] == 0
    assert x.loc[2, "scheduled_arr_pm5m"] == 1
    for idx, row in clean_departures(raw).iterrows():
        for phase, prefix in [("DEP", "dep"), ("ARR", "arr")]:
            for width in [5, 10, 15, 30, 60]:
                airport = raw["ADEP_mvt"].where(raw[PHASE].eq("DEP"), raw["ADES_mvt"])
                valid = raw[PHASE].eq(phase) & airport.eq(row["ADEP_mvt"]) & raw[ID].ne(row[ID])
                valid &= (raw[TIME] - row[TIME]).abs().le(pd.Timedelta(minutes=width))
                assert x.loc[idx, f"scheduled_{prefix}_pm{width}m"] == int(valid.sum())
    # Chunking inference cannot change demand when the caller supplies the same timetable.
    chunk = schedule_features(clean_departures(raw).iloc[[0]], raw)
    pd.testing.assert_frame_equal(x.iloc[[0]], chunk)


def test_poisoned_actual_values_never_change_features() -> None:
    raw = fixture()
    clean = clean_departures(raw, training=True)
    reference = schedule_features(clean, raw)
    poisoned = raw.copy()
    for column in EXCLUDED + ["ARVT_3_flt", "WK_TBL_CAT_flt", "EOBT_1_flt"]:
        if column not in [ID, "FLIGHT_ID_mvt"]:
            poisoned[column] = "POISON"
    pd.testing.assert_frame_equal(reference, schedule_features(clean_departures(poisoned), poisoned))


def test_forward_purges_entire_repeated_flight_and_keeps_negative_validation() -> None:
    raw = fixture()
    valid_id = raw.index[raw[TIME].dt.month.eq(9)][0]
    raw.loc[0, "FLIGHT_ID_mvt"] = raw.loc[valid_id, "FLIGHT_ID_mvt"]
    raw.loc[valid_id, TARGET] = -5
    data = clean_departures(raw, training=True)
    fit, valid, purged = purged_split(data, 9)
    assert purged == 1 and not fit[0] and valid[valid_id]
    assert data.loc[fit, TIME].max() < data.loc[valid, TIME].min()
    assert not (set(data.loc[fit, "FLIGHT_ID_mvt"]) & set(data.loc[valid, "FLIGHT_ID_mvt"]))
    assert metrics(data.loc[valid, TARGET], np.zeros(valid.sum()), data.loc[valid, "ADEP_mvt"])["negative_labels_retained"] == 1


@pytest.mark.parametrize("kind", ["mean", "median", "airport_time", "lightgbm", "catboost", "hist", "airport_lightgbm"])
def test_model_fit_only_vocab_fallback_and_saved_native_parity(tmp_path: Path, kind: str) -> None:
    raw = fixture()
    x = schedule_features(raw, raw)
    spec = ModelSpec(kind=kind, iterations=5, depth=3, threads=1)  # type: ignore[arg-type]
    model = FittedForecast(spec).fit(x.iloc[:200], raw[TARGET].iloc[:200])
    query = x.iloc[200:].copy()
    query.loc[query.index[0], "ADEP_mvt"] = "UNSEEN"
    query.loc[query.index[1], "AIRCRAFT_TYPE_mvt"] = "NEW_AIRCRAFT"
    assert "NEW_AIRCRAFT" not in model.vocabulary["AIRCRAFT_TYPE_mvt"]
    p = model.predict(query)
    assert p[0] == pytest.approx(float(raw[TARGET].iloc[:200].median() if kind == "median" else raw[TARGET].iloc[:200].mean()))
    assert np.isfinite(p).all() and (p >= 0).all()
    model.save(tmp_path / "saved")
    np.testing.assert_array_equal(p, FittedForecast.load(tmp_path / "saved").predict(query))
    assert set(model.columns).intersection(EXCLUDED) == set()
    assert set(CATEGORIES).issubset(model.columns)


def test_forecast_submission_and_typed_inference(tmp_path: Path) -> None:
    raw = fixture()
    x = schedule_features(raw, raw)
    root = tmp_path / "model"
    root.mkdir()
    model = FittedForecast(ModelSpec(kind="lightgbm", iterations=10, depth=3, threads=1)).fit(x, raw[TARGET])
    model.save(root / "lightgbm")
    write_json(root / "model-manifest.json", {"weights": {"lightgbm": 1}, "data_class": "synthetic",
        "model_version": "test/1", "permission_ref": "synthetic", "interval_radius_seconds": 60})
    ranking = raw.iloc[:10].drop(columns=TARGET)
    ranking.to_parquet(tmp_path / "ranking.parquet", index=False)
    template = ranking[[ID]].iloc[::-1].assign(**{TARGET: np.nan})
    template.to_parquet(tmp_path / "template.parquet", index=False)
    receipt = make_submission(root, tmp_path / "ranking.parquet", tmp_path / "template.parquet", tmp_path / "submission.parquet")
    assert receipt["rows"] == 10
    actual = pd.read_parquet(tmp_path / "submission.parquet")
    pd.testing.assert_series_equal(template[ID].reset_index(drop=True), actual[ID])
    response = TaxiOut(root).predict(PredictInput(airport="AAA", aircraft="A320", runway="09", stand="A1",
        scheduledTimeUtc="2026-02-01T12:00:00Z", scheduleKnownAtUtc="2026-02-01T10:00:00Z"), ranking)
    assert response.predictionSeconds >= 0 and response.confidence is None
    assert "UNVERIFIED_RUNWAY_STAND_UNUSED" in response.reasonCodes
    with pytest.raises(ValueError):
        PredictInput(airport="AAA", scheduledTimeUtc="2026-02-01T12:00:00", scheduleKnownAtUtc="2026-02-01T10:00:00Z")
    with pytest.raises(ValueError, match="snapshot"):
        TaxiOut(root).predict(PredictInput(airport="AAA", scheduledTimeUtc="2026-02-01T12:00:00Z",
            scheduleKnownAtUtc="2026-02-01T13:00:00Z"), ranking)
    with pytest.raises(ValueError, match="new"):
        make_submission(root, tmp_path / "ranking.parquet", tmp_path / "template.parquet", tmp_path / "submission.parquet")

# SPDX-License-Identifier: GPL-3.0-only
import pandas as pd

from traffic_features import movement_context
from traffic_features import nm_schedule_fallback_baseline
from traffic_features import add_carrier_features
from contest import matrix
from test_pipeline import fixture


def test_arrivals_use_destination_and_exclude_self_ties_and_other_airports() -> None:
    context = pd.DataFrame({
        "PHASE_mvt": ["ARR", "DEP", "DEP", "ARR", "ARR"],
        "ADEP_mvt": ["BBBB", "AAAA", "AAAA", "BBBB", "AAAA"],
        "ADES_mvt": ["AAAA", "BBBB", "BBBB", "AAAA", "BBBB"],
        "RUNWAY_mvt": ["09", "09", "09", "27", "09"],
        "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-01T10:01Z", "2025-01-01T10:03Z", "2025-01-01T10:05Z", "2025-01-01T10:05Z", "2025-01-01T10:02Z"]),
    })
    query = context.iloc[[2]]
    out = movement_context(query, context).iloc[0]
    assert out["airport_arr_previous_15m"] == 1
    assert out["airport_dep_previous_15m"] == 1
    assert out["runway_arr_headway_seconds"] == 240
    assert out["runway_dep_headway_seconds"] == 120
    contaminated = context.assign(TAXITIME_SEC_mvt=123456, BLOCK_TIME_UTC_mvt="not used", AOBT_3_flt="not used")
    pd.testing.assert_frame_equal(movement_context(query, context), movement_context(query, contaminated))


def test_month_gap_produces_zero_trailing_counts() -> None:
    context = pd.DataFrame({"PHASE_mvt": ["DEP", "DEP"], "ADEP_mvt": "AAAA", "ADES_mvt": "BBBB", "RUNWAY_mvt": "09", "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-31T23:00Z", "2025-07-01T00:00Z"])})
    out = movement_context(context.iloc[[1]], context).iloc[0]
    assert out["airport_dep_previous_60m"] == 0
    assert out["airport_dep_headway_seconds"] == 3600


def test_contest_matrix_does_not_read_airport_target_or_block_time() -> None:
    raw = fixture()
    raw["MVT_TIME_UTC_mvt"] = raw["SCHED_TIME_UTC_mvt"] + pd.Timedelta(minutes=20)
    raw["ADES_mvt"] = "LSZH"
    raw["RUNWAY_mvt"] = "09"
    raw["STAND_mvt"] = "A01"
    for column in ["IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "LOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]:
        raw[column] = raw["SCHED_TIME_UTC_mvt"]
    _, original = matrix(raw)
    changed = raw.assign(TAXITIME_SEC_mvt=-999999, BLOCK_TIME_UTC_mvt="unused")
    _, actual = matrix(changed)
    pd.testing.assert_frame_equal(original, actual)
    assert len(actual.columns) == 65


def test_unmatched_lirf_fallback_extrapolates_without_altering_other_airports() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG", "LIRF"],
        "missing_IOBT_flt": [1, 1, 0], "nm_departure_matches": [0, 0, 1],
        "movement_schedule_delta_sec": [87000., 87000., 87000.],
        "movement_minus_EOBT_1_flt": [float("nan"), float("nan"), 1200.],
        "movement_minus_LOBT_flt": [float("nan"), float("nan"), 850.],
        "movement_minus_AOBT_3_flt": [float("nan"), float("nan"), 800.]})
    assert nm_schedule_fallback_baseline(x).tolist() == [87000., 1000., 800.]


def test_carrier_features_exclude_flight_number_and_private_target_fields() -> None:
    raw = pd.DataFrame({"FLIGHT_mvt": ["RYR1234", "RYR9876", None],
        "SCHED_TIME_UTC_mvt": pd.to_datetime(["2025-01-02T09:00Z"] * 3),
        "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-02T09:20:30Z"] * 3)})
    base = pd.DataFrame({"ADEP_mvt": ["LIRF"] * 3})
    x = add_carrier_features(base, raw)
    assert x.movement_carrier.tolist() == ["RYR", "RYR", "UNKNOWN"]
    pd.testing.assert_series_equal(x.iloc[0], x.iloc[1], check_names=False)
    pd.testing.assert_frame_equal(x, add_carrier_features(base, raw.assign(TAXITIME_SEC_mvt=-1, FLIGHT_ID_mvt=42)))

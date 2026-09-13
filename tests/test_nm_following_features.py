# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest

from nm_following_features import following_nm_neighbors


def movements(time: str = "2025-01-15T12:00:00Z") -> pd.DataFrame:
    moved = pd.Series(pd.Timestamp(time) + pd.to_timedelta(
        [-1, 0, 1, 1, 900, 901, 3600, 3601, 7200, 7201], unit="s"))
    return pd.DataFrame({"PHASE_mvt": "DEP", "ADEP_mvt": "EDDF", "ADEP_flt": "EDDF",
        "RUNWAY_mvt": "09", "STAND_mvt": "A1", "MVT_TIME_UTC_mvt": moved,
        "AOBT_3_flt": moved - pd.to_timedelta([30, 600, 300, 900, 1200, 1500, 1800, 2100, 2400, 2700], unit="s"),
        "TAXITIME_SEC_mvt": "FORBIDDEN", "BLOCK_TIME_UTC_mvt": "FORBIDDEN", "MVT_ID_mvt": "FORBIDDEN"})


def test_strict_following_windows_inclusive_upper_limits_and_ties() -> None:
    raw = movements()
    result = following_nm_neighbors(raw.loc[[1]], raw)
    assert result.shape == (1, 30) and all(dtype == np.dtype("float32") for dtype in result.dtypes)
    for label in ["airport", "runway", "stand"]:
        prefix = "nm_following_" + label + "_"
        assert result[prefix + "15m_count"].iloc[0] == 3
        assert result[prefix + "15m_mean"].iloc[0] == 800
        np.testing.assert_allclose(result[prefix + "15m_std"], np.std([300, 900, 1200]), rtol=1e-6)
        assert result[prefix + "15m_own_minus_mean"].iloc[0] == -200
        assert result[prefix + "60m_count"].iloc[0] == 5
        assert result[prefix + "60m_mean"].iloc[0] == 1140
        assert result[prefix + "next_wait_seconds"].iloc[0] == 1
        assert result[prefix + "next_proxy"].iloc[0] == 600


def test_month_boundary_and_nearest_two_hour_limit() -> None:
    raw = movements("2025-01-31T23:30:00Z")
    result = following_nm_neighbors(raw.loc[[1]], raw)
    assert result.nm_following_airport_60m_count.iloc[0] == 4
    assert result.nm_following_airport_60m_mean.iloc[0] == 975
    raw = movements()
    result = following_nm_neighbors(raw.loc[[1]], raw.loc[[8, 9]])
    assert result.nm_following_airport_next_wait_seconds.iloc[0] == 7200
    assert result.nm_following_airport_next_proxy.iloc[0] == 2400
    result = following_nm_neighbors(raw.loc[[1]], raw.loc[[9]])
    assert result.nm_following_airport_next_proxy.isna().all()


def test_forbidden_fields_earlier_rows_and_permutations_cannot_change_values() -> None:
    raw = movements()
    queries = raw.loc[[1, 4, 6]]
    expected = following_nm_neighbors(queries, raw)
    altered = raw.assign(TAXITIME_SEC_mvt=-999999, BLOCK_TIME_UTC_mvt="2099-01-01", MVT_ID_mvt=123)
    altered.loc[0, "AOBT_3_flt"] = pd.Timestamp("1990-01-01T00:00Z")
    pd.testing.assert_frame_equal(expected, following_nm_neighbors(queries, altered))
    pd.testing.assert_frame_equal(expected, following_nm_neighbors(queries, raw.iloc[::-1]))
    actual = following_nm_neighbors(queries.iloc[::-1], raw).loc[queries.index]
    pd.testing.assert_frame_equal(expected, actual)


def test_invalid_pool_values_arrivals_and_origin_mismatches_are_excluded() -> None:
    raw = movements()
    raw.loc[2, "PHASE_mvt"] = "ARR"
    raw.loc[3, "ADEP_flt"] = "LFPG"
    result = following_nm_neighbors(raw.loc[[1]], raw)
    assert result.nm_following_airport_15m_count.iloc[0] == 1
    raw = movements()
    raw.loc[2, "AOBT_3_flt"] = pd.Timestamp("2025-01-15T10:00:01Z")
    raw.loc[3, "AOBT_3_flt"] = pd.Timestamp("2025-01-15T12:00:02Z")
    raw.loc[4, "AOBT_3_flt"] = pd.Timestamp("2025-01-15T10:14:59Z")
    result = following_nm_neighbors(raw.loc[[1]], raw)
    assert result.nm_following_airport_15m_count.iloc[0] == 1
    assert result.nm_following_airport_next_proxy.iloc[0] == 7200


def test_normalization_missing_groups_and_missing_own_proxy() -> None:
    raw = movements()
    raw["ADEP_mvt"] = " EDDF "
    raw["ADEP_flt"] = "eddf"
    raw["RUNWAY_mvt"] = " "
    raw["STAND_mvt"] = None
    raw.loc[1, "AOBT_3_flt"] = pd.NaT
    result = following_nm_neighbors(raw.loc[[1]], raw)
    assert result.nm_following_airport_15m_count.iloc[0] == 3
    assert result.nm_following_airport_15m_own_minus_mean.isna().all()
    for label in ["runway", "stand"]:
        assert result[f"nm_following_{label}_60m_count"].eq(0).all()
        assert result[f"nm_following_{label}_next_proxy"].isna().all()


def test_empty_inputs_and_missing_query_timestamp() -> None:
    raw = movements()
    result = following_nm_neighbors(raw.loc[[1]], raw.iloc[:0])
    assert result.filter(like="_count").eq(0).all().all()
    assert result.drop(columns=list(result.filter(like="_count"))).isna().all().all()
    assert following_nm_neighbors(raw.iloc[:0], raw).shape == (0, 30)
    missing = raw.loc[[1]].copy()
    missing["MVT_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=missing.index, dtype="datetime64[ns, UTC]")
    with pytest.raises(ValueError, match="complete"):
        following_nm_neighbors(missing, raw)


def test_timestamp_storage_units_preserve_microsecond_clock_context() -> None:
    raw = movements("2025-01-15T12:00:00.000123Z")
    expected = following_nm_neighbors(raw.loc[[1]], raw)
    ns = raw.copy()
    for column in ["MVT_TIME_UTC_mvt", "AOBT_3_flt"]:
        ns[column] = ns[column].dt.as_unit("ns")
    pd.testing.assert_frame_equal(expected, following_nm_neighbors(ns.loc[[1]], ns))

# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic contract tests; no competition files, targets or models are read."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nm_clock_overlay import CLOCKS, MOVEMENT, TARGET, apply_clock_overlay


def sample(actual_seconds: list[float], last_seconds: list[float]) -> tuple[pd.DataFrame, pd.Series]:
    assert len(actual_seconds) == len(last_seconds)
    index = pd.Index([f"synthetic-{i}" for i in range(len(actual_seconds))], name="MVT_ID_mvt")
    instant = pd.Timestamp("2025-03-01T12:00:00Z")
    rows = pd.DataFrame({"PHASE_mvt": "DEP", "ADEP_mvt": "LFPG", "ADEP_flt": "LFPG",
        MOVEMENT: instant, "IOBT_flt": instant - pd.Timedelta(minutes=20),
        "AOBT_3_flt": instant - pd.to_timedelta(actual_seconds, unit="s"),
        "LOBT_flt": instant - pd.to_timedelta(last_seconds, unit="s")}, index=index)
    values = pd.Series(np.arange(len(index), dtype=float) + 123.456789, index=index, name="incumbent")
    return rows, values


def test_threshold_is_strict_signed_and_full_replacement() -> None:
    rows, baseline = sample([1000., 1000., 9000., 1000.], [8200., 8200.001, 1000., 9000.])
    actual = apply_clock_overlay(rows, baseline)
    np.testing.assert_array_equal(actual, [baseline.iloc[0], 8200.001, baseline.iloc[2], 9000.])
    assert actual.name == baseline.name


def test_both_proxy_bounds_are_inclusive_and_invalid_actual_never_uses_valid_last() -> None:
    rows, baseline = sample([-7200., -7200.001, 165599., 165600., 165600., 172801., 1000.],
        [1., 1., 172800., 172800., 172800.001, 1000., 604801.])
    actual = apply_clock_overlay(rows, baseline)
    expected = baseline.copy()
    expected.iloc[0], expected.iloc[2] = 1., 172800.
    pd.testing.assert_series_equal(actual, expected)


def test_nonpositive_last_cannot_trigger_and_predictions_keep_nonnegative_floor() -> None:
    # With actual >= -7200, last <= 0 cannot also satisfy last-actual > 7200.
    rows, baseline = sample([-7200., -7200., -7200., -7200.001], [0., -.001, -7200., 0.])
    pd.testing.assert_series_equal(apply_clock_overlay(rows, baseline), baseline)


def test_origin_match_missing_clocks_and_specialist_preservation() -> None:
    rows, baseline = sample([1000.] * 9, [9000.] * 9)
    rows.loc["synthetic-0", "ADEP_flt"] = "EHAM"
    rows.loc["synthetic-1", "ADEP_flt"] = None
    rows.loc["synthetic-2", "AOBT_3_flt"] = pd.NaT
    rows.loc["synthetic-3", "LOBT_flt"] = pd.NaT
    rows.loc["synthetic-4", MOVEMENT] = pd.NaT
    rows.loc[["synthetic-5", "synthetic-6"], ["ADEP_mvt", "ADEP_flt"]] = "LIRF"
    rows.loc["synthetic-5", "IOBT_flt"] = pd.NaT
    rows.loc["synthetic-7", "IOBT_flt"] = pd.NaT
    original_rows, original_values = rows.copy(deep=True), baseline.copy(deep=True)
    actual = apply_clock_overlay(rows, baseline)
    expected = baseline.copy()
    expected.iloc[6:] = 9000.
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(rows, original_rows)
    pd.testing.assert_series_equal(baseline, original_values)


def test_label_alignment_preserves_prediction_order_index_name_and_frame_column() -> None:
    rows, baseline = sample([1000., 1000., 1000.], [9000., 8200., 10000.])
    baseline = baseline.iloc[np.array([2, 0, 1])]
    actual = apply_clock_overlay(rows.iloc[::-1], baseline)
    expected = pd.Series([10000., 9000., baseline.iloc[2]], index=baseline.index, name=baseline.name)
    pd.testing.assert_series_equal(actual, expected)
    frame = baseline.rename(TARGET).to_frame()
    frame.columns.name = "full_predictions"
    original_frame = frame.copy(deep=True)
    expected_frame = expected.rename(TARGET).to_frame()
    expected_frame.columns.name = frame.columns.name
    pd.testing.assert_frame_equal(apply_clock_overlay(rows, frame), expected_frame)
    pd.testing.assert_frame_equal(frame, original_frame)


@pytest.mark.parametrize("side", ["movements", "predictions"])
@pytest.mark.parametrize("kind", ["duplicate", "missing", "multi"])
def test_rejects_ambiguous_ids(side: str, kind: str) -> None:
    rows, baseline = sample([1000., 1000.], [9000., 9000.])
    index = (pd.Index(["same", "same"]) if kind == "duplicate" else
        pd.Index(["synthetic-0", None]) if kind == "missing" else
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
    if side == "movements":
        rows.index = index
    else:
        baseline.index = index
    with pytest.raises(ValueError, match="unique, non-null, single-level"):
        apply_clock_overlay(rows, baseline)


def test_rejects_missing_extra_and_type_coerced_ids() -> None:
    rows, baseline = sample([1000., 1000.], [9000., 9000.])
    for bad in [baseline.iloc[:1], baseline.rename(index={"synthetic-1": "other"})]:
        with pytest.raises(ValueError, match="IDs must match exactly"):
            apply_clock_overlay(rows, bad)
    rows.index = pd.Index([1, 2])
    baseline.index = pd.Index([1., 2.])
    with pytest.raises(ValueError, match="IDs must match exactly"):
        apply_clock_overlay(rows, baseline)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-np.inf], [-1.], [True], ["12"], [1 + 2j]])
@pytest.mark.parametrize("last_seconds", [8200., 9000.])
def test_rejects_invalid_predictions(values: list[object], last_seconds: float) -> None:
    rows, baseline = sample([1000.], [last_seconds])
    with pytest.raises(ValueError, match="real numeric, finite and nonnegative"):
        apply_clock_overlay(rows, pd.Series(values, index=baseline.index))


def test_numeric_conversion_cannot_corrupt_unaffected_values() -> None:
    rows, baseline = sample([1000., 1000.], [8200., 9000.])
    single = baseline.astype("float32")
    assert apply_clock_overlay(rows, single).iloc[0] == float(single.iloc[0])
    integral = pd.Series([2 ** 53, 0], index=baseline.index, dtype="int64")
    np.testing.assert_array_equal(apply_clock_overlay(rows, integral), [float(2 ** 53), 9000.])
    integral.iloc[0] = 2 ** 53 + 1
    with pytest.raises(ValueError, match="exactly representable"):
        apply_clock_overlay(rows, integral)
    negative_zero = pd.Series([-0., 1.], index=baseline.index)
    assert np.signbit(apply_clock_overlay(rows, negative_zero).iloc[0])


@pytest.mark.parametrize("column", list(CLOCKS))
@pytest.mark.parametrize("invalid", ["naive", "object", "other_timezone"])
def test_rejects_ambiguous_or_non_utc_clock_types(column: str, invalid: str) -> None:
    rows, baseline = sample([1000.], [9000.])
    rows[column] = (rows[column].dt.tz_localize(None) if invalid == "naive" else
        rows[column].astype(str) if invalid == "object" else rows[column].dt.tz_convert("Europe/Rome"))
    with pytest.raises(ValueError, match="typed UTC-aware"):
        apply_clock_overlay(rows, baseline)


def test_rejects_wrong_phase_columns_airports_and_ambiguous_prediction_frames() -> None:
    rows, baseline = sample([1000.], [9000.])
    for phase in ["ARR", None]:
        with pytest.raises(ValueError, match="Only original DEP"):
            apply_clock_overlay(rows.assign(PHASE_mvt=phase), baseline)
    with pytest.raises(ValueError, match="required column"):
        apply_clock_overlay(rows.drop(columns="IOBT_flt"), baseline)
    with pytest.raises(ValueError, match="required column"):
        apply_clock_overlay(pd.concat([rows, rows[[MOVEMENT]]], axis=1), baseline)
    with pytest.raises(ValueError, match="airport strings"):
        apply_clock_overlay(rows.assign(ADEP_flt=123), baseline)
    with pytest.raises(ValueError, match="departure airport must be present"):
        apply_clock_overlay(rows.assign(ADEP_mvt=None), baseline)
    for frame in [baseline.to_frame(), baseline.rename(TARGET).to_frame().assign(extra=1.)]:
        with pytest.raises(ValueError, match="exactly the"):
            apply_clock_overlay(rows, frame)


def test_empty_typed_departure_batch_is_preserved() -> None:
    rows, baseline = sample([], [])
    pd.testing.assert_series_equal(apply_clock_overlay(rows, baseline), baseline)


def test_unrelated_labels_and_clocks_are_unused_and_origins_are_not_normalized() -> None:
    rows, baseline = sample([1000.] * 4, [9000.] * 4)
    rows.loc["synthetic-0", "ADEP_flt"] = "lfpg"
    rows.loc["synthetic-1", "ADEP_flt"] = " LFPG "
    rows.loc["synthetic-2", "ADEP_mvt"] = "UNKNOWN"
    rows.loc["synthetic-2", "ADEP_flt"] = None
    expected = baseline.copy()
    expected.iloc[2:] = 9000.
    pd.testing.assert_series_equal(apply_clock_overlay(rows, baseline), expected)
    poisoned = rows.assign(TAXITIME_SEC_mvt="not a target", BLOCK_TIME_UTC_mvt="unusable",
        EOBT_1_flt="unusable", SCHED_TIME_UTC_mvt="unusable")
    pd.testing.assert_series_equal(apply_clock_overlay(poisoned, baseline), expected)

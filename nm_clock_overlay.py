# SPDX-License-Identifier: GPL-3.0-only
"""Fixed clock overlay for retrospective PRC departure predictions.

This pure function reads no files, targets or models and performs no fitting.
Supplied NM clocks are retrospective observations. This rule does not establish
their availability before departure or a general reliability ordering of clocks.
Applying the rule alone does not establish an official improvement.
"""
from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

VERSION = "prc2026-nm-clock-rule/1.0.0"
TARGET = "TAXITIME_SEC_mvt"
MOVEMENT = "MVT_TIME_UTC_mvt"
CLOCKS = (MOVEMENT, "IOBT_flt", "AOBT_3_flt", "LOBT_flt")
REQUIRED = ("PHASE_mvt", "ADEP_mvt", "ADEP_flt", *CLOCKS)


def _check_index(values: pd.Series | pd.DataFrame, label: str) -> None:
    if isinstance(values.index, pd.MultiIndex) or not values.index.is_unique or values.index.hasnans:
        raise ValueError(f"{label} must have unique, non-null, single-level movement IDs")


def _prediction_series(predictions: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(predictions, pd.DataFrame):
        if list(predictions.columns) != [TARGET]:
            raise ValueError(f"Prediction DataFrame must have exactly the {TARGET} column")
        values = predictions[TARGET]
    elif isinstance(predictions, pd.Series):
        values = predictions
    else:
        raise ValueError("Predictions must be an indexed Series or DataFrame")
    _check_index(values, "Predictions")
    if values.isna().any() or is_bool_dtype(values.dtype) or not (
            is_float_dtype(values.dtype) or is_integer_dtype(values.dtype)):
        raise ValueError("Predictions must be real numeric, finite and nonnegative")
    if isinstance(values.dtype, np.dtype) and is_float_dtype(values.dtype) and values.dtype.itemsize > 8:
        raise ValueError("Prediction precision wider than float64 is unsupported")
    # Every accepted integer is exactly representable in the output float64.
    # Reject larger integers instead of silently altering unaffected predictions.
    if is_integer_dtype(values.dtype) and values.gt(2 ** 53).any():
        raise ValueError("Integer predictions must be exactly representable within 0..2**53")
    result = values.astype("float64").copy()
    if not np.isfinite(result).all() or result.lt(0).any():
        raise ValueError("Predictions must be real numeric, finite and nonnegative")
    return result


@overload
def apply_clock_overlay(movements: pd.DataFrame, predictions: pd.Series) -> pd.Series: ...


@overload
def apply_clock_overlay(movements: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame: ...


def apply_clock_overlay(
    movements: pd.DataFrame, predictions: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Apply the frozen rule to supplied full departure predictions, by ID.

    Inputs must contain the same unique, non-null movement IDs. Movement order
    may differ; the result keeps prediction order, index and name. A prediction
    DataFrame must contain only TAXITIME_SEC_mvt. Output values are float64;
    accepted input values outside the trigger remain numerically exact.

    All movement rows must be DEP. Four clock columns must be typed UTC-aware
    datetimes; NaT is allowed. Missing movement/AOBT/LOBT makes a row ineligible.
    LIRF with missing IOBT is always preserved. No input is mutated.

    On ordinary origin-matched rows only, both clipped movement-minus-clock
    proxies must lie in [-7200, 172800], inclusive, and AOBT minus LOBT must be
    strictly greater than 7200 seconds. The entire prediction is then replaced
    with max(0, movement minus LOBT), not a residual adjustment or a blend.
    """
    if not isinstance(movements, pd.DataFrame):
        raise ValueError("Movements must be an indexed DataFrame")
    _check_index(movements, "Movements")
    baseline = _prediction_series(predictions)
    if movements.index.dtype != baseline.index.dtype or len(movements) != len(baseline) or not (
            baseline.index.isin(movements.index).all() and movements.index.isin(baseline.index).all()):
        raise ValueError("Movement and prediction IDs must match exactly")
    if not movements.columns.is_unique or not set(REQUIRED).issubset(movements.columns):
        raise ValueError("Unique original movement columns, including every required column, are required")
    rows = movements.loc[baseline.index, list(REQUIRED)]
    if not rows.PHASE_mvt.eq("DEP").fillna(False).all():
        raise ValueError("Only original DEP movement rows are accepted")
    for column in CLOCKS:
        dtype = rows[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
            raise ValueError(f"{column} must contain typed UTC-aware datetimes or NaT")
    for column in ("ADEP_mvt", "ADEP_flt"):
        observed = rows[column].dropna()
        if not observed.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain original airport strings or missing values")
    if rows.ADEP_mvt.isna().any():
        raise ValueError("Original movement departure airport must be present")

    origin_matches = rows.ADEP_flt.fillna("UNKNOWN").eq(rows.ADEP_mvt)
    special = rows.ADEP_mvt.eq("LIRF") & rows.IOBT_flt.isna()
    try:
        actual = (rows[MOVEMENT] - rows.AOBT_3_flt).dt.total_seconds().clip(-604800, 604800)
        last = (rows[MOVEMENT] - rows.LOBT_flt).dt.total_seconds().clip(-604800, 604800)
        disagreement = (rows.AOBT_3_flt - rows.LOBT_flt).dt.total_seconds()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Supplied UTC clock differences cannot be represented") from exc
    actual_valid = actual.notna() & actual.between(-7200, 172800) & origin_matches
    last_valid = last.notna() & last.between(-7200, 172800) & origin_matches
    trigger = ~special & origin_matches & actual_valid & last_valid & disagreement.gt(7200)
    result = baseline.copy()
    result.loc[trigger] = last.loc[trigger].clip(lower=0)
    if not np.isfinite(result).all() or result.lt(0).any():
        raise ValueError("Clock overlay produced an invalid prediction")
    if not np.array_equal(result.loc[~trigger], baseline.loc[~trigger]):
        raise ValueError("Clock overlay altered predictions outside its fixed scope")
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
        frame[TARGET] = result
        return frame
    return result

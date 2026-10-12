# SPDX-License-Identifier: GPL-3.0-only
"""Forecast allowlist and schedule-only context, independent of historical contest code."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline import ID, PHASE, TARGET, TIME, require_columns

SCHEMA_VERSION = "forecast/1.0.0"
CATEGORIES = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "flight_operator"]
WINDOWS = (5, 10, 15, 30, 60)
# Final observed assignments and NM snapshots have no as-of provenance in this dataset.
EXCLUDED = [TARGET, ID, "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "RUNWAY_mvt", "STAND_mvt", "LOBT_flt", "AOBT_3_flt", "ATOT_3_flt",
            "IOBT_flt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]


def clean_departures(raw: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    require_columns(raw, [ID, PHASE, TIME, "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"])
    if raw[ID].isna().any() or raw[ID].duplicated().any():
        raise ValueError("Movement IDs must be non-null and unique, including arrivals")
    if not raw[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unexpected movement phase")
    data = raw.loc[raw[PHASE].eq("DEP")].copy().reset_index(drop=True)
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors="raise")
    if data.empty or data[TIME].isna().any() or data["ADEP_mvt"].isna().any():
        raise ValueError("Departures require airport and scheduled UTC timestamps")
    if training:
        require_columns(data, [TARGET])
        data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(float)
        if not np.isfinite(data[TARGET]).all():
            raise ValueError("Training targets must be finite; negatives are fit-only excluded")
    return data


def categorical(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    for col in CATEGORIES[:-1]:
        result[col] = data[col].fillna("UNKNOWN").astype(str) if col in data else "UNKNOWN"
    # Published flight designator; no identifier itself or observed NM callsign is a predictor.
    flight = data["FLIGHT_mvt"].fillna("").astype(str) if "FLIGHT_mvt" in data else pd.Series("", index=data.index)
    result["flight_operator"] = flight.str.extract(r"^([A-Za-z]{2,3})(?=[0-9])", expand=False).str.upper().fillna("UNKNOWN")
    return result


def schedule_features(data: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Count known timetable movements. Never read actual times, targets, stand or runway.

    The caller must provide the timetable known at prediction time. Static PRC schedule
    snapshots approximate this contract, but do not prove historical snapshot availability.
    Arrival identity is destination, departure identity is origin. Self is excluded.
    """
    require_columns(schedule, [ID, PHASE, TIME, "ADEP_mvt", "ADES_mvt"])
    if schedule[ID].isna().any() or schedule[ID].duplicated().any():
        raise ValueError("Schedule IDs must be unique")
    if not schedule[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unexpected schedule phase")
    clock = pd.to_datetime(data[TIME], utc=True, errors="raise")
    timetable = pd.to_datetime(schedule[TIME], utc=True, errors="raise")
    if clock.isna().any() or timetable.isna().any():
        raise ValueError("Missing schedule time")
    result = categorical(data)
    seconds = clock.dt.hour * 3600 + clock.dt.minute * 60 + clock.dt.second
    for name, values in {"hour_utc": clock.dt.hour, "weekday_utc": clock.dt.dayofweek,
                         "month_utc": clock.dt.month, "weekend": (clock.dt.dayofweek >= 5).astype(int),
                         "hour_sin": np.sin(seconds * 2 * np.pi / 86400),
                         "hour_cos": np.cos(seconds * 2 * np.pi / 86400),
                         "year_sin": np.sin((clock.dt.dayofyear - 1) * 2 * np.pi / 365.25),
                         "year_cos": np.cos((clock.dt.dayofyear - 1) * 2 * np.pi / 365.25)}.items():
        result[name] = np.asarray(values, dtype=float)
    # Explicit ns conversion: pandas timestamp resolution is not assumed.
    query = clock.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    known = timetable.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    airports = schedule["ADEP_mvt"].where(schedule[PHASE].eq("DEP"), schedule["ADES_mvt"])
    for phase, prefix in [("DEP", "dep"), ("ARR", "arr")]:
        for width in WINDOWS:
            result[f"scheduled_{prefix}_pm{width}m"] = 0.0
        result[f"scheduled_{prefix}_previous30m"] = 0.0
        for airport, indices in data.groupby("ADEP_mvt", sort=False).groups.items():
            idx = data.index.get_indexer(indices)
            mask = airports.eq(str(airport)) & schedule[PHASE].eq(phase)
            source = np.sort(known[mask.to_numpy()])
            point = query[idx]
            # Remove self only when the schedule actually contains that same ID/time/phase.
            relevant = schedule.loc[mask, [ID]].copy()
            relevant["clock"] = known[mask.to_numpy()]
            membership = pd.Series(relevant["clock"].to_numpy(), index=relevant[ID])
            self_count = (data.loc[indices, ID].map(membership).to_numpy() == point).astype(int)
            for width in WINDOWS:
                delta = width * 60 * 1_000_000_000
                counts = np.searchsorted(source, point + delta, side="right") - np.searchsorted(source, point - delta, side="left") - self_count
                result.loc[indices, f"scheduled_{prefix}_pm{width}m"] = counts.astype(float)
            # Strictly preceding times, so self and simultaneous movements never enter queue proxy.
            prior = np.searchsorted(source, point, side="left") - np.searchsorted(source, point - 1800 * 1_000_000_000, side="left")
            result.loc[indices, f"scheduled_{prefix}_previous30m"] = prior.astype(float)
    return result


def purged_split(data: pd.DataFrame, month: int, year: int = 2025) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    fit = (data[TIME] < start).to_numpy()
    valid = ((data[TIME] >= start) & (data[TIME] < end)).to_numpy()
    purged = 0
    if "FLIGHT_ID_mvt" in data:
        groups = set(data.loc[valid, "FLIGHT_ID_mvt"].dropna())
        overlap = fit & data["FLIGHT_ID_mvt"].isin(groups).to_numpy()
        purged = int(overlap.sum())
        fit = fit & ~overlap
    fit = fit & data[TARGET].ge(0).to_numpy()
    if not fit.any() or not valid.any():
        raise ValueError(f"Empty chronological fit or validation for {year}-{month:02}")
    return fit, valid, purged


def metrics(actual: Any, predicted: Any, airports: Any) -> dict[str, Any]:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite matching nonempty vectors")
    error = p - y
    absolute = np.abs(error)
    worst = np.argsort(absolute, kind="stable")[-max(1, int(np.ceil(len(y) * .1))):]
    report: dict[str, Any] = {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
                              "bias_seconds": float(error.mean()), "mae": float(absolute.mean()),
                              "p90_absolute_error": float(np.quantile(absolute, .9)),
                              "worst_decile_rmse": float(np.sqrt(np.mean(error[worst] ** 2))),
                              "negative_labels_retained": int((y < 0).sum()),
                              "actual_quantiles": np.quantile(y, [.1, .5, .9, .99]).tolist(),
                              "prediction_quantiles": np.quantile(p, [.1, .5, .9, .99]).tolist()}
    airport = np.asarray(airports)
    report["per_airport"] = {str(a): {"n": int((airport == a).sum()),
                                      "rmse": float(np.sqrt(np.mean(error[airport == a] ** 2))),
                                      "bias_seconds": float(error[airport == a].mean())} for a in np.unique(airport)}
    return report

# SPDX-License-Identifier: GPL-3.0-only
"""Label-free retrospective traffic context from organizer-supplied movements."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import features, require_columns


def movement_context(departures: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Trailing airport/runway demand; exact-time ties are excluded, including self."""
    columns = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
    require_columns(context, columns)
    require_columns(departures, columns)
    base = context[columns].copy()
    base["airport"] = base["ADEP_mvt"].where(base["PHASE_mvt"].eq("DEP"), base["ADES_mvt"])
    base["runway"] = base["RUNWAY_mvt"].fillna("UNKNOWN").astype(str)
    # pandas 3 timestamps use microsecond storage by default: explicitly normalize to seconds.
    base["seconds"] = pd.to_datetime(base["MVT_TIME_UTC_mvt"], utc=True).dt.as_unit("ns").astype("int64") / 1e9
    times = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    airports = departures["ADEP_mvt"].fillna("UNKNOWN").astype(str).to_numpy()
    runways = departures["RUNWAY_mvt"].fillna("UNKNOWN").astype(str).to_numpy()
    out = pd.DataFrame(index=departures.index)
    for phase in ["DEP", "ARR"]:
        for runway_scope in [False, True]:
            prefix = ("runway" if runway_scope else "airport") + "_" + phase.lower()
            groups = ["airport", "runway"] if runway_scope else ["airport"]
            pool = base.loc[base["PHASE_mvt"].eq(phase)]
            lookup = {key if isinstance(key, tuple) else (key,): np.sort(group["seconds"].to_numpy()) for key, group in pool.groupby(groups, dropna=False)}
            counts = {minutes: np.zeros(len(departures), dtype=np.float32) for minutes in [15, 30, 60]}
            headway = np.full(len(departures), 3600, dtype=np.float32)
            keys = pd.DataFrame({"airport": airports, "runway": runways})
            for key, indices in keys.groupby(groups, dropna=False).indices.items():
                query = times[indices]
                events = lookup.get(key if isinstance(key, tuple) else (key,))
                if events is None or not len(events):
                    continue
                right = np.searchsorted(events, query, side="left")
                for minutes, values in counts.items():
                    values[indices] = right - np.searchsorted(events, query - minutes * 60, side="left")
                previous = np.maximum(0, right - 1)
                headway[indices] = np.where(right > 0, np.minimum(3600, query - events[previous]), 3600)
            for minutes, values in counts.items():
                out[f"{prefix}_previous_{minutes}m"] = values
            out[f"{prefix}_headway_seconds"] = headway
    return out


def enriched_features(departures: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    out = features(departures, "challenge_context")
    out = pd.concat([out, movement_context(departures, context)], axis=1)
    scheduled = pd.to_datetime(departures["SCHED_TIME_UTC_mvt"], utc=True)
    movement = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
    out["scheduled_minute_of_day"] = scheduled.dt.hour * 60 + scheduled.dt.minute
    out["movement_minute_of_day"] = movement.dt.hour * 60 + movement.dt.minute
    out["movement_day_of_year"] = movement.dt.dayofyear
    out["airport_operator"] = out["ADEP_mvt"] + ":" + out["AIRCRAFT_OPERATOR_flt"]
    out["stand_runway"] = out["airport_STAND_mvt"] + ":" + out["RUNWAY_mvt"]
    return out


def add_planned_features(out: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """Use initial/first-filed estimates, never actual or last-known off-block."""
    out = out.copy()
    movement = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
    scheduled = pd.to_datetime(departures["SCHED_TIME_UTC_mvt"], utc=True)
    for column in ["IOBT_flt", "EOBT_1_flt"]:
        require_columns(departures, [column])
        planned = pd.to_datetime(departures[column], utc=True, errors="coerce")
        out[f"movement_minus_{column}"] = (movement - planned).dt.total_seconds().clip(-604800, 604800)
        out[f"schedule_minus_{column}"] = (scheduled - planned).dt.total_seconds().clip(-604800, 604800)
        out[f"missing_{column}"] = planned.isna().astype(int)
    for column in ["ADEP_flt", "ADES_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt"]:
        out[column] = departures[column].fillna("UNKNOWN").astype(str) if column in departures else "UNKNOWN"
    out["nm_departure_matches"] = (out["ADEP_flt"] == out["ADEP_mvt"]).astype(int)
    out["nm_destination_matches"] = (out["ADES_flt"] == out["ADES_mvt"]).astype(int)
    return out


def planned_baseline(out: pd.DataFrame) -> pd.Series:
    """Continuous extrapolation for long delays using first-filed planned time."""
    delta = out["movement_minus_EOBT_1_flt"]
    valid = delta.notna() & delta.between(-7200, 172800) & out["nm_departure_matches"].eq(1)
    return delta.where(valid, 1000.0).clip(lower=0).astype(float)


def add_nm_observed_features(out: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """Competition-only NM records explicitly retained in the published ranking schema.

    These post-event records are not a predeparture forecast and are never used
    by the product. Airport BLOCK_TIME and TAXITIME remain excluded. Source:
    https://prc-data-challenge-2026.netlify.app/data.html#column-description
    """
    out = out.copy()
    movement = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
    for column in ["AOBT_3_flt", "LOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]:
        require_columns(departures, [column])
        time = pd.to_datetime(departures[column], utc=True, errors="coerce")
        out[f"movement_minus_{column}"] = (movement - time).dt.total_seconds().clip(-604800, 604800)
        out[f"missing_{column}"] = time.isna().astype(int)
    out["nm_actual_vs_estimated_seconds"] = out["movement_minus_EOBT_1_flt"] - out["movement_minus_AOBT_3_flt"]
    out["nm_last_vs_actual_seconds"] = out["movement_minus_AOBT_3_flt"] - out["movement_minus_LOBT_flt"]
    return out


def nm_observed_baseline(out: pd.DataFrame) -> pd.Series:
    result = planned_baseline(out)
    for column in ["LOBT_flt", "AOBT_3_flt"]:
        delta = out[f"movement_minus_{column}"]
        valid = delta.notna() & delta.between(-7200, 172800) & out["nm_departure_matches"].eq(1)
        result = delta.where(valid, result).clip(lower=0).astype(float)
    return result


def nm_schedule_fallback_baseline(out: pd.DataFrame) -> pd.Series:
    """Training-supported LIRF unmatched-source extrapolation; no target edits."""
    result = nm_observed_baseline(out)
    unmatched = out["missing_IOBT_flt"].eq(1) & out["ADEP_mvt"].eq("LIRF")
    schedule_delta = out["movement_schedule_delta_sec"].clip(0, 172800)
    return schedule_delta.where(unmatched, result).astype(float)


def add_carrier_features(out: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """Recover an airline prefix where NM matching lost operator metadata.

    Flight numbers and movement/flight IDs are not predictors. Calendar and
    timestamp precision are supplied movement context, not target information.
    """
    require_columns(departures, ["FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])
    out = out.copy()
    out["movement_carrier"] = departures["FLIGHT_mvt"].str.extract(r"^([A-Za-z]{2,4})", expand=False).fillna("UNKNOWN").str.upper()
    out["airport_movement_carrier"] = out["ADEP_mvt"] + ":" + out["movement_carrier"]
    scheduled = pd.to_datetime(departures["SCHED_TIME_UTC_mvt"], utc=True)
    movement = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
    out["scheduled_day"] = scheduled.dt.day.astype(float)
    out["movement_second"] = movement.dt.second.astype(float)
    out["schedule_second"] = scheduled.dt.second.astype(float)
    return out

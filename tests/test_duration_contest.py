# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest

from duration_contest import add_duration_features, blend, matrix
from test_pipeline import fixture


def sample() -> pd.DataFrame:
    raw = fixture()
    raw["MVT_TIME_UTC_mvt"] = raw["SCHED_TIME_UTC_mvt"] + pd.Timedelta(minutes=20)
    for key, value in {"ADES_mvt": "LSZH", "RUNWAY_mvt": "09", "STAND_mvt": "A01",
        "FLIGHT_mvt": "ABC123", "CALLSIGN_flt": "abc123", "AIRCRAFT_TYPE_flt": "A320",
        "ADEP_flt": "LFPG", "ADES_flt": "LSZH", "FLIGHT_RULE_mvt": "I", "FLIGHT_RULE_flt": "I"}.items():
        raw[key] = value
    for column, minutes in {"IOBT_flt": 0, "EOBT_1_flt": 0, "AOBT_3_flt": 5,
        "LOBT_flt": 5, "ARVT_1_flt": 60, "ARVT_3_flt": 75}.items():
        raw[column] = raw["SCHED_TIME_UTC_mvt"] + pd.Timedelta(minutes=minutes)
    return raw


def test_duration_context_uses_only_supplied_predictors_and_preserves_order() -> None:
    raw = sample()
    dep, x = matrix(raw)
    _, changed = matrix(raw.assign(TAXITIME_SEC_mvt=-999999, BLOCK_TIME_UTC_mvt="unused",
        MVT_ID_mvt=raw.MVT_ID_mvt + 10000, FLIGHT_ID_mvt=42))
    pd.testing.assert_frame_equal(x, changed)
    assert len(x.columns) == 93
    assert x.index.equals(dep.index)
    assert x.nm_planned_flight_duration.eq(3600).all()
    assert x.nm_actual_flight_duration.eq(4200).all()
    assert x.nm_actual_offblock_schedule_delay.eq(300).all()
    assert x.callsign_equal.eq(1).all()
    with pytest.raises(ValueError, match="alignment"):
        add_duration_features(x, dep.iloc[::-1])


def test_missing_nm_and_callsign_values_remain_missing() -> None:
    raw = sample()
    raw.loc[0, "ARVT_3_flt"] = pd.NaT
    raw.loc[0, "CALLSIGN_flt"] = None
    _, x = matrix(raw)
    assert pd.isna(x.loc[0, "nm_actual_flight_duration"])
    assert x.loc[0, "nm_actual_duration_plausible"] == 0
    assert x.loc[0, "callsign_both_present"] == 0
    assert x.loc[0, "callsign_equal"] == 0


def test_blend_keeps_specialist_and_rejects_misaligned_or_invalid_predictions() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LIRF", "LFPG"],
        "missing_IOBT_flt": [1, 0, 1]}, index=[9, 2, 7])
    base = pd.Series([100., 200., 300.], index=x.index)
    duration = pd.Series([1000., 400., 500.], index=x.index)
    assert blend(x, base, duration).tolist() == [100., 350., 450.]
    with pytest.raises(ValueError, match="alignment"):
        blend(x, base.iloc[::-1], duration)
    with pytest.raises(ValueError, match="finite"):
        blend(x, base, duration * np.nan)
    with pytest.raises(ValueError, match="nonnegative"):
        blend(x, base, -duration)

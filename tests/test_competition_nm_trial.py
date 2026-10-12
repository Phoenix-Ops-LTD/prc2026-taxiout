# SPDX-License-Identifier: GPL-3.0-only
import pandas as pd

from competition_nm_trial import masked_inputs


def test_only_departure_labels_are_masked_and_original_inputs_are_preserved() -> None:
    raw = pd.DataFrame({"PHASE_mvt": ["DEP", "ARR"], "TAXITIME_SEC_mvt": [100., 200.],
        "BLOCK_TIME_UTC_mvt": pd.to_datetime(["2025-01-01T10:00:00Z", "2025-01-01T11:00:00Z"]),
        "AOBT_3_flt": ["provided departure observation", "provided arrival observation"]})
    original = raw.copy()
    masked = masked_inputs(raw)
    assert masked.loc[0, ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"]].isna().all()
    assert masked.loc[1, "TAXITIME_SEC_mvt"] == 200.
    assert masked.loc[1, "BLOCK_TIME_UTC_mvt"] == raw.loc[1, "BLOCK_TIME_UTC_mvt"]
    assert masked["AOBT_3_flt"].equals(raw["AOBT_3_flt"])
    pd.testing.assert_frame_equal(raw, original)

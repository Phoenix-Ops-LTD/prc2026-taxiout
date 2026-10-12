# SPDX-License-Identifier: GPL-3.0-only
import pandas as pd

from forecast_finalize import choose
from pipeline import TARGET, TIME


def test_equal_blend_requires_improvement_in_every_forward_month() -> None:
    frame = pd.DataFrame({TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
                          TARGET: [300, 300, 300], "a": [320, 320, 320], "b": [280, 280, 330]})
    weights, report = choose(frame, ["a", "b"])
    assert weights == {"a": 1.0}
    assert report["scores"]["a+b"]["rmse"] < report["scores"]["a"]["rmse"]
    assert not report["scores"]["a+b"]["eligible"]
    assert not report["december_used_for_selection"]


def test_improving_fixed_pair_is_selected_without_adjusting_weights() -> None:
    frame = pd.DataFrame({TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
                          TARGET: [300, 300, 300], "a": [320, 310, 320], "b": [280, 290, 280]})
    weights, report = choose(frame, ["a", "b"])
    assert weights == {"a": .5, "b": .5}
    assert report["scores"]["a+b"]["rmse"] == 0

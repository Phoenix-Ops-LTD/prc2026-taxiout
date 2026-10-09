# SPDX-License-Identifier: GPL-3.0-only
import pandas as pd
import pytest

from forecast_gpu_compare import evaluate
from pipeline import ID, TARGET, TIME


def fixture() -> pd.DataFrame:
    return pd.DataFrame({ID: [1, 2, 3], TARGET: [10., 10., 10.],
        TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-11-01"], utc=True),
        "incumbent": [14., 14., 14.], "gpu": [11., 11., 11.]})


def test_promotion_requires_minimum_gain_and_every_fold_improvement() -> None:
    assert evaluate(fixture(), {"incumbent": 1.}, ["gpu"])["promoted"] is True
    worsening_fold = fixture().assign(gpu=[10., 10., 15.])
    result = evaluate(worsening_fold, {"incumbent": 1.}, ["gpu"])
    assert result["scores"]["gpu"]["rmse"] < result["incumbent"]["rmse"]
    assert result["promoted"] is False
    assert evaluate(fixture().assign(gpu=13.5), {"incumbent": 1.}, ["gpu"])["promoted"] is False


def test_partial_or_later_labels_cannot_enter_comparison() -> None:
    with pytest.raises(ValueError, match="complete"):
        evaluate(fixture().iloc[:2], {"incumbent": 1.}, ["gpu"])
    with pytest.raises(ValueError, match="complete"):
        evaluate(fixture().assign(**{TIME: pd.to_datetime(["2025-09-01", "2025-10-01", "2025-12-01"], utc=True)}), {"incumbent": 1.}, ["gpu"])

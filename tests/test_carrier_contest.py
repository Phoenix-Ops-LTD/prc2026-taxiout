# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest

from carrier_contest import combine


def test_specialist_replaces_only_unmatched_lirf_and_preserves_row_alignment() -> None:
    x = pd.DataFrame({"ADEP_mvt": ["LIRF", "LFPG", "LIRF"],
        "missing_IOBT_flt": [1, 1, 0], "nm_departure_matches": [0, 0, 1],
        "movement_schedule_delta_sec": [87000., 87000., 87000.],
        "movement_minus_EOBT_1_flt": [np.nan, np.nan, 1200.],
        "movement_minus_LOBT_flt": [np.nan, np.nan, 850.],
        "movement_minus_AOBT_3_flt": [np.nan, np.nan, 800.]}, index=[14, 2, 80])
    result = combine(x, np.array([1., 100., -900.]), np.array([-70000.]))
    assert result.index.tolist() == [14, 2, 80]
    assert result.tolist() == [17000., 1100., 0.]
    with pytest.raises(ValueError, match="counts"):
        combine(x, np.array([1., 2., 3.]), np.array([]))
    with pytest.raises(ValueError, match="Non-finite"):
        combine(x, np.array([1., np.nan, 3.]), np.array([1.]))

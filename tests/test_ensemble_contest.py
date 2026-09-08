# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pandas as pd
import pytest
from ensemble_contest import blended_residual


def test_blend_floors_constituents_before_weighting() -> None:
    offset = pd.Series([100., 1000., 50.], index=[8, 3, 22])
    residual = blended_residual(np.array([-200., 200., 50.]), np.array([100., -2000., -100.]), offset)
    np.testing.assert_allclose(residual + offset, [50., 900., 75.])
    with pytest.raises(ValueError, match="lengths"):
        blended_residual(np.array([]), np.array([]), offset)

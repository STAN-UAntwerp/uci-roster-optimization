import numpy as np
import pytest

from workload_analysis import calendar, gini, rolling_peak


def test_gini_includes_zero_members_and_handles_undefined_total():
    assert gini([2, 2, 2, 2]) == pytest.approx(0)
    assert gini([0, 0, 0, 8]) == pytest.approx(.75)
    assert np.isnan(gini([0, 0]))
    with pytest.raises(ValueError):
        gini([-1, 2])


def test_peak_window_boundary_and_rest_day():
    daily = np.zeros(100, dtype=int)
    daily[[0, 29, 30, 59]] = 1
    assert rolling_peak(daily, 30) == 2
    assert rolling_peak(daily, 60) == 4
    dates = calendar({'race/test/2024': {'level': '2.UWT', 'stages': [
        {'date': '2024-01-01'}, {'date': '2024-01-03'}, {'date': '2024-01-03'}]}})
    assert len(dates['race/test/2024']) == 2

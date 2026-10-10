import numpy as np

from pai_training.analysis.trials import enabled_intervals, slice_by_time


def test_enabled_intervals_from_status_flags():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f = np.array([False, True, True, False, True, True, True])
    assert enabled_intervals(t, f) == [(1.0, 3.0), (4.0, 6.0)]


def test_enabled_intervals_none():
    assert enabled_intervals(np.array([0.0, 1.0]), np.array([False, False])) == []


def test_slice_by_time_inclusive_start_exclusive_end():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    v = np.arange(5)
    tt, vv = slice_by_time(t, v, 0.5, 1.5)
    assert tt.tolist() == [0.5, 1.0]
    assert vv.tolist() == [1, 2]

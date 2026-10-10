"""評価中の rosbag を試行ごとに切り出すための補助関数（ROS 非依存）."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def enabled_intervals(times: np.ndarray, enabled: np.ndarray) -> List[Tuple[float, float]]:
    """/policy/status の enabled 列から、方策が有効だった区間 [開始, 終了) を返す.

    最後まで有効なままなら終了は最後の時刻。
    """
    out, start = [], None
    for t, e in zip(times, enabled):
        if e and start is None:
            start = float(t)
        elif not e and start is not None:
            out.append((start, float(t)))
            start = None
    if start is not None:
        out.append((start, float(times[-1])))
    return out


def slice_by_time(times: np.ndarray, values: np.ndarray, start: float, end: float):
    m = (times >= start) & (times < end)
    return times[m], values[m]

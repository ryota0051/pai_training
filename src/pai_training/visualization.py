"""LeRobotDataset を rerun で表示するときのエンティティ名（scripts/visualize_dataset.py）.

関節ごとに指令（action）と観測（state）を同じグラフに重ね、物体位置は別のグラフにする。
"""
from __future__ import annotations

from typing import List, Mapping

GROUPS = {
    'action': ('joints', 'action'),
    'observation.state': ('joints', 'state'),
}


def feature_names(features: Mapping, key: str) -> List[str]:
    """features[key]['names']（なければ添字）を返す."""
    f = features[key]
    names = f.get('names')
    if isinstance(names, dict):   # LeRobot の古い形式 {'motors': [...]}
        names = next(iter(names.values()))
    return list(names) if names else [str(i) for i in range(f['shape'][0])]


def entity_paths(key: str, names: List[str]) -> List[str]:
    if key in GROUPS:
        group, leaf = GROUPS[key]
        return [f'{group}/{n}/{leaf}' for n in names]
    if key == 'observation.environment_state':
        return [f'object/{n}' for n in names]
    return [f'{key}/{n}' for n in names]

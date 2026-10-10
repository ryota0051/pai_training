"""推論ノードと共有する純粋ロジック（torch / ROS に依存しない）."""
from __future__ import annotations

from collections import deque
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pai_training.conversion import ConversionSpec


def compose_observation(spec: ConversionSpec, joints_by_name: Mapping[str, float],
                        object_xyz: Optional[Sequence[float]]) -> Optional[Dict[str, np.ndarray]]:
    """変換時と同じ規則（joint_order・observation sources）で観測ベクトルを作る. 揃っていなければ None."""
    if any(n not in joints_by_name for n in spec.joint_order):
        return None
    if spec.needs_object and object_xyz is None:
        return None
    sources = {'joints': np.array([joints_by_name[n] for n in spec.joint_order], dtype=np.float32)}
    if object_xyz is not None:
        sources['object_xyz'] = np.asarray(object_xyz, dtype=np.float32)[:3]
    return {
        'observation.state': np.concatenate([sources[s] for s in spec.state_sources]),
        'observation.environment_state': np.concatenate([sources[s] for s in spec.env_state_sources]),
    }


class TemporalEnsembler:
    """ACT の temporal ensembling（毎ステップ予測したチャンクの重なりを指数重みで平均）.

    重み w_i = exp(-coeff * i)、i=0 が最も古い予測（ACT 論文・LeRobot 実装と同じ規約）。
    """

    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.reset()

    def reset(self) -> None:
        # 各要素: 将来 k ステップ目に対する過去の予測のリスト（古い順）
        self._buffer = deque()

    def update(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (chunk_size, action_dim)。現在時刻に実行する行動を返す."""
        chunk = np.asarray(chunk)
        for k in range(len(chunk)):
            if k < len(self._buffer):
                self._buffer[k].append(chunk[k])
            else:
                self._buffer.append([chunk[k]])
        preds = np.stack(self._buffer.popleft())
        weights = np.exp(-self.coeff * np.arange(len(preds)))
        return (weights[:, None] * preds).sum(axis=0) / weights.sum()

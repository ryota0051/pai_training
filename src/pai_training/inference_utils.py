"""推論ノードと共有する純粋ロジック（torch / ROS に依存しない）."""
from __future__ import annotations

from collections import deque
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pai_training.conversion import ConversionSpec, ForwardKinematics


def compose_observation(spec: ConversionSpec, joints_by_name: Mapping[str, float],
                        object_xyz: Optional[Sequence[float]],
                        fk: Optional[ForwardKinematics] = None,
                        place_xy: Optional[Sequence[float]] = None) -> Optional[Dict[str, np.ndarray]]:
    """変換時と同じ規則（joint_order・observation sources）で観測ベクトルを作る. 揃っていなければ None."""
    if any(n not in joints_by_name for n in spec.joint_order):
        return None
    if spec.needs_object and object_xyz is None:
        return None
    if spec.needs_place and place_xy is None:
        return None
    sources = {'joints': np.array([joints_by_name[n] for n in spec.joint_order], dtype=np.float32)}
    if object_xyz is not None:
        sources['object_xyz'] = np.asarray(object_xyz, dtype=np.float32)[:3]
    if spec.needs_fk:
        if fk is None:
            raise ValueError('object_minus_ee is requested but no forward kinematics (fk) was given')
        ee = np.asarray(fk(sources['joints'][:-1]), dtype=np.float32)
        sources['object_minus_ee'] = sources['object_xyz'] - ee
    if place_xy is not None:
        sources['place_xy'] = np.asarray(place_xy, dtype=np.float32)[:2]
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


class ObservationCorruptor:
    """物体位置の観測を劣化させる（認識誤差・遅延への頑健性評価用）.

    - delay_steps: 制御ステップ単位の遅延（履歴が溜まるまでは最古の値）
    - noise_std: 各軸に加える独立なガウスノイズの標準偏差 [m]（毎ステップ独立）
    - bias_std: 試行ごとに一定のオフセット（キャリブレーション誤差を模擬）の標準偏差 [m]。reset で再サンプル
    - axes: ノイズを加える軸（既定は x, y, z すべて）
    """

    def __init__(self, noise_std: float, delay_steps: int, seed: Optional[int] = None,
                 axes: Sequence[int] = (0, 1, 2), bias_std: float = 0.0):
        self.noise_std = float(noise_std)
        self.bias_std = float(bias_std)
        self.delay_steps = int(delay_steps)
        self.axes = list(axes)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._history = deque(maxlen=self.delay_steps + 1)
        self._bias = np.zeros(3)
        if self.bias_std > 0:
            self._bias[self.axes] = self.rng.normal(0.0, self.bias_std, size=len(self.axes))

    def apply(self, xyz: Sequence[float]) -> np.ndarray:
        self._history.append(np.asarray(xyz, dtype=np.float64)[:3])
        out = self._history[0].copy() + self._bias
        if self.noise_std > 0:
            out[self.axes] += self.rng.normal(0.0, self.noise_std, size=len(self.axes))
        return out

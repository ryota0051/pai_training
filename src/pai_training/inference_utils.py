"""推論ノードと共有する純粋ロジック（torch / ROS に依存しない）."""
from __future__ import annotations

from collections import deque
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pai_training.conversion import IMAGE_FEATURE_PREFIX, ConversionSpec, ForwardKinematics
from pai_training.images import to_chw_float


def compose_observation(spec: ConversionSpec, joints_by_name: Mapping[str, float],
                        object_xyz: Optional[Sequence[float]],
                        fk: Optional[ForwardKinematics] = None,
                        place_xy: Optional[Sequence[float]] = None,
                        images: Optional[Mapping[str, np.ndarray]] = None) -> Optional[Dict[str, np.ndarray]]:
    """変換時と同じ規則（joint_order・observation sources）で観測ベクトルを作る. 揃っていなければ None.

    images: カメラ名 → (H, W, 3) uint8 の RGB。observation.images.<カメラ名> に (3, H, W) float32 [0, 1] で入れる
    """
    images = images or {}
    if any(n not in joints_by_name for n in spec.joint_order):
        return None
    if spec.needs_object and object_xyz is None:
        return None
    if spec.needs_place and place_xy is None:
        return None
    if any(cam not in images for cam in spec.image_sources):
        return None
    sources = {'joints': np.array([joints_by_name[n] for n in spec.joint_order], dtype=np.float32)}
    if object_xyz is not None:
        sources['object_xyz'] = np.asarray(object_xyz, dtype=np.float32)[:3]
        # 固定（ObjectLatch）は呼び出し側で済ませた値を受け取る
        sources['object_xyz_latched'] = sources['object_xyz']
    if spec.needs_fk:
        if fk is None:
            raise ValueError('object_minus_ee is requested but no forward kinematics (fk) was given')
        ee = np.asarray(fk(sources['joints'][:-1]), dtype=np.float32)
        sources['object_minus_ee'] = sources['object_xyz'] - ee
    if place_xy is not None:
        sources['place_xy'] = np.asarray(place_xy, dtype=np.float32)[:2]
    obs = {'observation.state': np.concatenate([sources[s] for s in spec.state_sources])}
    if spec.has_env_state:
        obs['observation.environment_state'] = np.concatenate([sources[s] for s in spec.env_state_sources])
    for cam in spec.image_sources:
        obs[IMAGE_FEATURE_PREFIX + cam] = to_chw_float(images[cam])
    return obs


class ObjectLatch:
    """グリッパが閉じている間は物体位置を固定する（変換側の latch_while_closed と同じ規則）.

    開いている間: 入力された物体位置をそのまま返し、覚えておく（None なら直前の値）
    閉じている間（関節角 ≥ close になってから ≤ open_ になるまで）: 閉じる直前の値を返し続ける
    """

    def __init__(self, close: float, open_: float):
        self.close = close
        self.open = open_
        self.reset()

    def reset(self) -> None:
        self.closed = False
        self.last: Optional[tuple] = None

    def update(self, object_xyz: Optional[Sequence[float]], gripper: float) -> Optional[tuple]:
        if not self.closed and gripper >= self.close:
            self.closed = True
            if self.last is None and object_xyz is not None:
                self.last = tuple(object_xyz)
        elif self.closed and gripper <= self.open:
            self.closed = False
        if not self.closed and object_xyz is not None:
            self.last = tuple(object_xyz)
        return self.last


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


# 指令（推論の出力先）のトピック。変換設定のものは「ラベルの出どころ」なので推論では使わない
COMMAND_TOPIC_KEYS = ('arm_command', 'gripper_command')


def resolve_topics(default_topics: Mapping[str, str], conversion_topics: Mapping[str, Optional[str]],
                   inference_topics: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """推論ノードのトピック. 観測は変換設定から引き継ぎ、推論設定で上書きする.

    指令（arm_command / gripper_command）は変換設定から引き継がない。DART のデモでは変換設定の arm_command が
    外乱なしの指令（/teleop/expert_arm_command）を指し、推論の出力先にすると腕が動かないため（実験13）。
    """
    observed = {k: v for k, v in conversion_topics.items() if k not in COMMAND_TOPIC_KEYS}
    return {**default_topics, **observed, **inference_topics}

"""rosbag2 → LeRobotDataset 変換の純粋ロジック（rosbags / lerobot に依存しない）.

メッセージは (timestamp_ns, names, values) のタプル列で受け取る。
- joint_states / arm_command / gripper_command: names は関節名、values は位置
- object_pose / place_pose: names は None、values は [x, y, z]
カメラ画像は大きいので、ここでは時刻だけを受け取り、各フレームで使う画像の番号（image_index.<名前>）を返す。
画像そのものの読み出しは呼び出し側（scripts/rosbag2lerobot.py）がエピソードごとに行う。

upstream の rosbag2lerobot.py は /joint_states をメッセージ内の並び順のまま使っていたが、
ここでは joint_order に従って **名前で並べ替える**（指令側の関節順が異なっても整合する）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Message = Tuple[int, Optional[Sequence[str]], Sequence[float]]

# upstream（演習）の /joint_states の並びと同じ既定順
DEFAULT_JOINT_ORDER = [
    'crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4',
    'crane_plus_joint3', 'crane_plus_joint_hand',
]
DEFAULT_TOPICS = {
    'joint_states': '/joint_states',
    'arm_command': '/crane_plus_arm_controller/joint_trajectory',
    'gripper_command': '/crane_plus_gripper_controller/joint_trajectory',
    'object_pose': None,
    'place_pose': None,
}
# object_minus_ee: 物体位置 − 手先位置（順運動学）。人が目で見ている「手先と物体のずれ」を明示的に与える（実験11）
# place_xy: 置き場所の中心 [x, y]。置き場所もエピソードごとに変える場合に使う（実験12）
# object_xyz_latched: グリッパが閉じている間は、閉じる直前の物体位置で固定する（実験12 の改良 A'）。
#   運搬中は物体がグリッパに隠れてカメラで見えないため、実行時（検出器）と学習時（真値）の観測をそろえる
SOURCES = ('joints', 'object_xyz', 'object_minus_ee', 'place_xy', 'object_xyz_latched')
# カメラ画像を観測に使う場合の、フレームごとの画像番号のキー（後ろにカメラ名）と LeRobot の特徴量名
IMAGE_INDEX_PREFIX = 'image_index.'
IMAGE_FEATURE_PREFIX = 'observation.images.'
# グリッパが「閉じた」「開いた」とみなす関節角 [rad]（把持アシストと同じ。間の値では状態を変えない）
DEFAULT_OBJECT_LATCH = {'close': 0.12, 'open': 0.0}
# 順運動学: arm_joints の順の関節角 → 手先（把持中心）の位置 [x, y, z]
ForwardKinematics = Callable[[Sequence[float]], Sequence[float]]


@dataclass
class ConversionSpec:
    joint_order: List[str] = field(default_factory=lambda: list(DEFAULT_JOINT_ORDER))
    target_freq: float = 20.0
    state_sources: List[str] = field(default_factory=lambda: ['joints'])
    env_state_sources: List[str] = field(default_factory=lambda: ['joints'])
    topics: Dict[str, Optional[str]] = field(default_factory=lambda: dict(DEFAULT_TOPICS))
    pad_to_longest: bool = True
    object_latch: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECT_LATCH))
    # カメラ名のリスト。各カメラの画像トピックは topics[カメラ名]。特徴量は observation.images.<カメラ名>
    image_sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        for s in list(self.state_sources) + list(self.env_state_sources):
            if s not in SOURCES:
                raise ValueError(f'unknown observation source: {s} (choose from {SOURCES})')
        for cam in self.image_sources:
            if not self.topics.get(cam):
                raise ValueError(f'image source "{cam}" has no topic (set topics.{cam})')

    @property
    def arm_joints(self) -> List[str]:
        return self.joint_order[:-1]

    @property
    def gripper_joint(self) -> str:
        return self.joint_order[-1]

    @property
    def needs_object(self) -> bool:
        used = self.state_sources + self.env_state_sources
        return 'object_xyz' in used or 'object_minus_ee' in used or 'object_xyz_latched' in used

    @property
    def needs_place(self) -> bool:
        return 'place_xy' in self.state_sources + self.env_state_sources

    @property
    def has_env_state(self) -> bool:
        """observation.environment_state を作るか（画像だけで物体を見る場合は空にする）."""
        return bool(self.env_state_sources)

    @property
    def needs_fk(self) -> bool:
        return 'object_minus_ee' in self.state_sources + self.env_state_sources

    @classmethod
    def from_dict(cls, cfg: Mapping) -> 'ConversionSpec':
        obs = cfg.get('observation', {}) or {}
        topics = dict(DEFAULT_TOPICS)
        topics.update(cfg.get('topics', {}) or {})
        return cls(
            joint_order=list(cfg.get('joint_order', DEFAULT_JOINT_ORDER)),
            target_freq=float(cfg.get('target_freq', 20)),
            state_sources=list(obs.get('state', ['joints'])),
            env_state_sources=list(obs.get('environment_state', ['joints'])),
            topics=topics,
            pad_to_longest=bool(cfg.get('pad_to_longest', True)),
            object_latch={**DEFAULT_OBJECT_LATCH, **(cfg.get('object_latch') or {})},
            image_sources=list(obs.get('images') or []),
        )


def positions_by_name(names: Sequence[str], values: Sequence[float], order: Sequence[str]) -> List[float]:
    table = dict(zip(names, values))
    return [table[n] for n in order]


def sample_and_hold(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """各 target_time 以前で最新の値を返す（最初より前は先頭値）."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    idx = np.searchsorted(np.asarray(times), np.asarray(target_times), side='right') - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def latch_while_closed(values: np.ndarray, gripper: np.ndarray, close: float, open_: float) -> np.ndarray:
    """グリッパが閉じている間（関節角 ≥ close になってから ≤ open_ になるまで）は、閉じる直前の値で固定する.

    閉じた状態で始まった場合は最初の値で固定する。推論側の ObjectLatch と同じ規則。
    """
    out = np.array(values, dtype=float, copy=True)
    closed, held = False, None
    for i, g in enumerate(gripper):
        if not closed and g >= close:
            closed = True
            held = out[i - 1].copy() if i > 0 else out[i].copy()
        elif closed and g <= open_:
            closed = False
        if closed:
            out[i] = held
    return out


def time_window(time_arrays: Sequence[np.ndarray]) -> Tuple[int, int]:
    """全トピックが揃っている区間 (start, end)."""
    return max(int(t[0]) for t in time_arrays), min(int(t[-1]) for t in time_arrays)


def num_frames(start: int, end: int, freq: float) -> int:
    return int((end - start) * freq / 1e9)


def _times(msgs: Sequence[Message]) -> np.ndarray:
    return np.array([m[0] for m in msgs], dtype=np.int64)


def build_episode_arrays(spec: ConversionSpec,
                         joint_states: Sequence[Message],
                         arm_commands: Sequence[Message],
                         gripper_commands: Sequence[Message],
                         object_poses: Optional[Sequence[Message]] = None,
                         fk: Optional[ForwardKinematics] = None,
                         place_poses: Optional[Sequence[Message]] = None,
                         image_times: Optional[Mapping[str, Sequence[int]]] = None) -> Dict[str, np.ndarray]:
    """1 エピソード分の observation.state / observation.environment_state / action を作る.

    image_sources がある場合は image_times（カメラ名 → 各画像の時刻 [ns]）を受け取り、
    image_index.<カメラ名>（各フレームで使う画像の番号。直前の画像）も返す。
    """
    image_times = image_times or {}
    if spec.needs_object and not object_poses:
        raise ValueError('object_xyz is requested but no object pose messages were given')
    if spec.needs_fk and fk is None:
        raise ValueError('object_minus_ee is requested but no forward kinematics (fk) was given')
    if spec.needs_place and not place_poses:
        raise ValueError('place_xy is requested but no place pose messages were given')
    for cam in spec.image_sources:
        if not image_times.get(cam):
            raise ValueError(f'image source "{cam}" is requested but no image timestamps were given')

    streams = [joint_states, arm_commands, gripper_commands]
    if spec.needs_object:
        streams.append(object_poses)
    if spec.needs_place:
        streams.append(place_poses)
    stream_times = [_times(s) for s in streams]
    stream_times += [np.asarray(image_times[cam], dtype=np.int64) for cam in spec.image_sources]
    start, end = time_window(stream_times)
    n = num_frames(start, end, spec.target_freq)
    if n <= 0:
        raise ValueError('episode is too short')
    target = np.linspace(start, end, n, endpoint=False).astype(np.int64)

    joints = np.array([positions_by_name(nm, v, spec.joint_order) for _, nm, v in joint_states])
    arm = np.array([positions_by_name(nm, v, spec.arm_joints) for _, nm, v in arm_commands])
    grip = np.array([positions_by_name(nm, v, [spec.gripper_joint]) for _, nm, v in gripper_commands])

    sources = {'joints': sample_and_hold(_times(joint_states), joints, target)}
    if spec.needs_object:
        obj = np.array([list(v)[:3] for _, _, v in object_poses])
        sources['object_xyz'] = sample_and_hold(_times(object_poses), obj, target)
    if spec.needs_fk:
        ee = np.array([fk(q[:-1]) for q in sources['joints']])
        sources['object_minus_ee'] = sources['object_xyz'] - ee
    if 'object_xyz_latched' in spec.state_sources + spec.env_state_sources:
        sources['object_xyz_latched'] = latch_while_closed(
            sources['object_xyz'], sources['joints'][:, -1], spec.object_latch['close'], spec.object_latch['open'])
    if spec.needs_place:
        place = np.array([list(v)[:2] for _, _, v in place_poses])
        sources['place_xy'] = sample_and_hold(_times(place_poses), place, target)

    action = np.hstack([sample_and_hold(_times(arm_commands), arm, target),
                        sample_and_hold(_times(gripper_commands), grip, target)])
    out = {
        'observation.state': np.hstack([sources[s] for s in spec.state_sources]).astype(np.float32),
        'action': action.astype(np.float32),
    }
    if spec.has_env_state:
        out['observation.environment_state'] =             np.hstack([sources[s] for s in spec.env_state_sources]).astype(np.float32)
    for cam in spec.image_sources:
        times = np.asarray(image_times[cam], dtype=np.int64)
        idx = np.searchsorted(times, target, side='right') - 1
        out[IMAGE_INDEX_PREFIX + cam] = np.clip(idx, 0, len(times) - 1).astype(np.int64)
    return out

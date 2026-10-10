"""rosbag2 → LeRobotDataset 変換の純粋ロジック（rosbags / lerobot に依存しない）.

メッセージは (timestamp_ns, names, values) のタプル列で受け取る。
- joint_states / arm_command / gripper_command: names は関節名、values は位置
- object_pose: names は None、values は [x, y, z]

upstream の rosbag2lerobot.py は /joint_states をメッセージ内の並び順のまま使っていたが、
ここでは joint_order に従って **名前で並べ替える**（指令側の関節順が異なっても整合する）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
}
SOURCES = ('joints', 'object_xyz')


@dataclass
class ConversionSpec:
    joint_order: List[str] = field(default_factory=lambda: list(DEFAULT_JOINT_ORDER))
    target_freq: float = 20.0
    state_sources: List[str] = field(default_factory=lambda: ['joints'])
    env_state_sources: List[str] = field(default_factory=lambda: ['joints'])
    topics: Dict[str, Optional[str]] = field(default_factory=lambda: dict(DEFAULT_TOPICS))
    pad_to_longest: bool = True

    def __post_init__(self):
        for s in list(self.state_sources) + list(self.env_state_sources):
            if s not in SOURCES:
                raise ValueError(f'unknown observation source: {s} (choose from {SOURCES})')

    @property
    def arm_joints(self) -> List[str]:
        return self.joint_order[:-1]

    @property
    def gripper_joint(self) -> str:
        return self.joint_order[-1]

    @property
    def needs_object(self) -> bool:
        return 'object_xyz' in self.state_sources + self.env_state_sources

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
                         object_poses: Optional[Sequence[Message]] = None) -> Dict[str, np.ndarray]:
    """1 エピソード分の observation.state / observation.environment_state / action を作る."""
    if spec.needs_object and not object_poses:
        raise ValueError('object_xyz is requested but no object pose messages were given')

    streams = [joint_states, arm_commands, gripper_commands]
    if spec.needs_object:
        streams.append(object_poses)
    start, end = time_window([_times(s) for s in streams])
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

    action = np.hstack([sample_and_hold(_times(arm_commands), arm, target),
                        sample_and_hold(_times(gripper_commands), grip, target)])
    return {
        'observation.state': np.hstack([sources[s] for s in spec.state_sources]).astype(np.float32),
        'observation.environment_state':
            np.hstack([sources[s] for s in spec.env_state_sources]).astype(np.float32),
        'action': action.astype(np.float32),
    }

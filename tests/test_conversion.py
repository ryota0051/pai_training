import numpy as np
import pytest

from pai_training.conversion import (
    ConversionSpec,
    build_episode_arrays,
    positions_by_name,
    sample_and_hold,
    time_window,
)

ORDER = ['j1', 'j2', 'j4', 'j3', 'hand']


def test_positions_by_name_reorders():
    assert positions_by_name(['j1', 'j2', 'j3', 'j4'], [1, 2, 3, 4], ['j1', 'j2', 'j4', 'j3']) == [1, 2, 4, 3]


def test_positions_by_name_missing_raises():
    with pytest.raises(KeyError):
        positions_by_name(['j1'], [1.0], ['j1', 'j2'])


def test_sample_and_hold_uses_latest_past_value():
    times = np.array([0, 10, 20])
    values = np.array([[0.0], [1.0], [2.0]])
    out = sample_and_hold(times, values, np.array([0, 5, 10, 19, 25]))
    assert out[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]


def test_sample_and_hold_before_first_uses_first():
    out = sample_and_hold(np.array([10, 20]), np.array([[1.0], [2.0]]), np.array([0]))
    assert out[0, 0] == 1.0


def test_time_window_is_intersection():
    assert time_window([np.array([0, 100]), np.array([20, 80]), np.array([10, 90])]) == (20, 80)


def _msgs():
    """20Hz 相当の合成メッセージ（ns 単位の時刻）。joint_states は名前順がバラバラ."""
    ns = 1_000_000_000
    joint_states = [
        (t * ns // 20, ['j1', 'j2', 'j4', 'j3', 'hand'], [0.1 * t, 0.2, 0.4, 0.3, -0.5])
        for t in range(0, 41)
    ]
    arm_cmds = [
        (t * ns // 20, ['j1', 'j2', 'j3', 'j4'], [0.1 * t + 0.05, 0.2, 0.3, 0.4])
        for t in range(0, 41)
    ]
    gripper_cmds = [(t * ns // 20, ['hand'], [0.3 if t > 20 else -0.5]) for t in range(0, 41)]
    object_poses = [(t * ns // 20, None, [0.12, 0.05 + 0.001 * t, 0.02]) for t in range(0, 41)]
    return joint_states, arm_cmds, gripper_cmds, object_poses


def test_build_episode_arrays_objpos():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=10,
                          state_sources=['joints'], env_state_sources=['object_xyz'])
    arrays = build_episode_arrays(spec, js, arm, grip, obj)
    assert arrays['observation.state'].shape[1] == 5
    assert arrays['observation.environment_state'].shape[1] == 3
    assert arrays['action'].shape[1] == 5
    n = arrays['action'].shape[0]
    assert n == 20  # 2 秒 × 10Hz
    # 名前で並べ替えられている: [j1, j2, j4, j3, hand]
    assert arrays['observation.state'][0].tolist() == pytest.approx([0.0, 0.2, 0.4, 0.3, -0.5])
    assert arrays['action'][0].tolist() == pytest.approx([0.05, 0.2, 0.4, 0.3, -0.5])
    assert arrays['observation.environment_state'][0].tolist() == pytest.approx([0.12, 0.05, 0.02])


def test_build_episode_arrays_concat_sources():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=10,
                          state_sources=['joints', 'object_xyz'], env_state_sources=['joints'])
    arrays = build_episode_arrays(spec, js, arm, grip, obj)
    assert arrays['observation.state'].shape[1] == 8
    assert arrays['observation.environment_state'].shape[1] == 5


def test_object_source_required_when_requested():
    js, arm, grip, _ = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=10,
                          state_sources=['joints'], env_state_sources=['object_xyz'])
    with pytest.raises(ValueError):
        build_episode_arrays(spec, js, arm, grip, None)


def test_spec_from_dict_defaults_match_upstream():
    spec = ConversionSpec.from_dict({'target_freq': 20})
    assert spec.joint_order == ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4',
                                'crane_plus_joint3', 'crane_plus_joint_hand']
    assert spec.state_sources == ['joints']
    assert spec.env_state_sources == ['joints']
    assert spec.topics['object_pose'] is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        ConversionSpec.from_dict({'observation': {'state': ['foo']}})


def _fake_fk(arm_joints):
    """テスト用の順運動学: 手先位置 = 先頭 3 関節の値."""
    return [arm_joints[0], arm_joints[1], arm_joints[2]]


def test_object_minus_ee_source_uses_forward_kinematics():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=10,
                          state_sources=['joints'], env_state_sources=['object_xyz', 'object_minus_ee'])
    assert spec.needs_object and spec.needs_fk
    arrays = build_episode_arrays(spec, js, arm, grip, obj, fk=_fake_fk)
    env = arrays['observation.environment_state']
    assert env.shape[1] == 6
    # 先頭フレーム: 物体 (0.12, 0.05, 0.02) − 手先 (j1, j2, j3) = (0.0, 0.2, 0.3)（ORDER は j1, j2, j4, j3）
    assert env[0, 3:].tolist() == pytest.approx([0.12 - 0.0, 0.05 - 0.2, 0.02 - 0.4])


def test_object_minus_ee_requires_fk():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, env_state_sources=['object_minus_ee'])
    with pytest.raises(ValueError):
        build_episode_arrays(spec, js, arm, grip, obj)

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


# ---- 置き場所の位置（place_xy, 実験12: 置き場所もランダムに変える） ----
def _place_msgs():
    ns = 1_000_000_000
    return [(t * ns // 20, None, [0.08, -0.1, 0.0]) for t in range(0, 41)]


def test_place_xy_source_appends_place_position():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=10,
                          state_sources=['joints'], env_state_sources=['object_xyz', 'place_xy'])
    arrays = build_episode_arrays(spec, js, arm, grip, obj, place_poses=_place_msgs())
    env = arrays['observation.environment_state']
    assert env.shape[1] == 5
    assert env[0].tolist() == pytest.approx([0.12, 0.05, 0.02, 0.08, -0.1])


def test_place_xy_requires_place_messages():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'], env_state_sources=['object_xyz', 'place_xy'])
    with pytest.raises(ValueError, match='place'):
        build_episode_arrays(spec, js, arm, grip, obj)


def test_spec_needs_place_and_default_topic():
    spec = ConversionSpec.from_dict({'observation': {'environment_state': ['object_xyz', 'place_xy']},
                                     'topics': {'place_pose': '/task/place_pose'}})
    assert spec.needs_place and spec.topics['place_pose'] == '/task/place_pose'
    assert not ConversionSpec().needs_place


# ---- グリッパが閉じている間は物体位置を固定する（object_xyz_latched, 実験12 の改良 A'） ----
def test_latch_while_closed_holds_value_from_last_open_step():
    from pai_training.conversion import latch_while_closed
    obj = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    grip = np.array([-0.5, -0.5, 0.2, 0.3, 0.05, -0.1, -0.5])   # 閉(≥0.12) → 間(0.05 は閉のまま) → 開(≤0.0)
    out = latch_while_closed(obj, grip, close=0.12, open_=0.0)
    # 閉じた時点（index 2）から開くまで（index 4）は、閉じる直前（index 1）の値
    assert out[:, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 5.0, 6.0]


def test_latch_while_closed_when_closed_from_start_uses_first_value():
    from pai_training.conversion import latch_while_closed
    out = latch_while_closed(np.array([[1.0], [2.0]]), np.array([0.3, 0.3]), close=0.12, open_=0.0)
    assert out[:, 0].tolist() == [1.0, 1.0]


def test_object_xyz_latched_source():
    js, arm, grip, obj = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=20, state_sources=['joints'],
                          env_state_sources=['object_xyz_latched'], object_latch={'close': 0.12, 'open': 0.0})
    arrays = build_episode_arrays(spec, js, arm, grip, obj)
    env = arrays['observation.environment_state']
    # _msgs の物体は y が毎ステップ 0.001 ずつ動くが、hand は -0.5 で開いたままなので固定されない
    assert env[5, 1] > env[0, 1]
    assert spec.needs_object


def test_spec_reads_object_latch():
    spec = ConversionSpec.from_dict({'observation': {'environment_state': ['object_xyz_latched', 'place_xy']},
                                     'object_latch': {'close': 0.1, 'open': -0.1}})
    assert spec.object_latch == {'close': 0.1, 'open': -0.1}


# ---- カメラ画像（A 案: 画像 + 関節角, 物体位置なし） ----
def test_spec_reads_image_sources_and_empty_environment_state():
    spec = ConversionSpec.from_dict({'observation': {'state': ['joints'], 'environment_state': [],
                                                     'images': ['overhead']},
                                     'topics': {'overhead': '/task/camera/image'}})
    assert spec.image_sources == ['overhead']
    assert spec.topics['overhead'] == '/task/camera/image'
    assert not spec.has_env_state
    assert not spec.needs_object


def test_spec_image_source_requires_topic():
    with pytest.raises(ValueError):
        ConversionSpec.from_dict({'observation': {'images': ['overhead']}})


def test_spec_default_has_no_images():
    spec = ConversionSpec.from_dict({})
    assert spec.image_sources == []
    assert spec.has_env_state


def test_build_episode_arrays_image_index_samples_latest_past_image():
    js, arm, grip, _ = _msgs()
    ns = 1_000_000_000
    # 画像は 10Hz（0.1s ごと）、0.05s から 1.95s まで
    image_times = [ns // 20 + t * ns // 10 for t in range(0, 20)]
    spec = ConversionSpec(joint_order=ORDER, target_freq=20, state_sources=['joints'], env_state_sources=[],
                          image_sources=['overhead'], topics={'overhead': '/cam'})
    arrays = build_episode_arrays(spec, js, arm, grip, image_times={'overhead': image_times})
    assert 'observation.environment_state' not in arrays
    idx = arrays['image_index.overhead']
    assert idx.dtype == np.int64
    # 時間窓は最初の画像（0.05s）から始まる。20Hz の各フレームで直前の画像を使う
    n = len(arrays['action'])
    assert len(idx) == n == len(arrays['observation.state'])
    assert idx[:4].tolist() == [0, 0, 1, 1]
    assert idx.max() <= len(image_times) - 1


def test_build_episode_arrays_requires_image_times():
    js, arm, grip, _ = _msgs()
    spec = ConversionSpec(joint_order=ORDER, target_freq=20, env_state_sources=[],
                          image_sources=['overhead'], topics={'overhead': '/cam'})
    with pytest.raises(ValueError):
        build_episode_arrays(spec, js, arm, grip)

import numpy as np
import pytest

from pai_training.conversion import ConversionSpec
from pai_training.inference_utils import TemporalEnsembler, compose_observation

ORDER = ['j1', 'j2', 'j3', 'j4', 'hand']


def test_compose_observation_orders_by_name_and_sources():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'], env_state_sources=['object_xyz'])
    joints = {'j4': 4.0, 'j1': 1.0, 'hand': 5.0, 'j3': 3.0, 'j2': 2.0, 'other': 9.0}
    obs = compose_observation(spec, joints, (0.1, 0.2, 0.3))
    assert obs['observation.state'].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert obs['observation.environment_state'].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert obs['observation.state'].dtype == np.float32


def test_compose_observation_concat():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints', 'object_xyz'], env_state_sources=['joints'])
    joints = {n: float(i) for i, n in enumerate(ORDER)}
    obs = compose_observation(spec, joints, (0.1, 0.2, 0.3))
    assert obs['observation.state'].shape == (8,)


def test_compose_observation_requires_object():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'], env_state_sources=['object_xyz'])
    assert compose_observation(spec, {n: 0.0 for n in ORDER}, None) is None


def test_compose_observation_requires_all_joints():
    spec = ConversionSpec(joint_order=ORDER)
    assert compose_observation(spec, {'j1': 0.0}, None) is None


def test_temporal_ensembler_first_call_returns_first_action():
    te = TemporalEnsembler(coeff=0.01, chunk_size=3)
    chunk = np.array([[1.0], [2.0], [3.0]])
    assert te.update(chunk).tolist() == [1.0]


def test_temporal_ensembler_averages_overlapping_predictions():
    te = TemporalEnsembler(coeff=0.0, chunk_size=3)  # coeff=0 で単純平均
    te.update(np.array([[0.0], [10.0], [20.0]]))
    # 2 回目: 現在時刻に対する予測は 1 回目の 2 番目 (10) と 2 回目の 1 番目 (4) の平均
    out = te.update(np.array([[4.0], [14.0], [24.0]]))
    assert out.tolist() == pytest.approx([7.0])


def test_temporal_ensembler_weights_older_more_with_positive_coeff():
    te = TemporalEnsembler(coeff=1.0, chunk_size=2)
    te.update(np.array([[0.0], [10.0]]))
    out = te.update(np.array([[0.0], [0.0]]))
    w_old, w_new = 1.0, np.exp(-1.0)
    assert out[0] == pytest.approx(10.0 * w_old / (w_old + w_new))


def test_temporal_ensembler_reset():
    te = TemporalEnsembler(coeff=0.0, chunk_size=2)
    te.update(np.array([[0.0], [10.0]]))
    te.reset()
    assert te.update(np.array([[5.0], [6.0]])).tolist() == [5.0]


# ---- 物体位置の観測劣化（ノイズ・遅延）----
from pai_training.inference_utils import ObservationCorruptor  # noqa: E402


def test_corruptor_identity_when_disabled():
    c = ObservationCorruptor(noise_std=0.0, delay_steps=0, seed=0)
    assert c.apply((0.1, 0.2, 0.3)) == pytest.approx((0.1, 0.2, 0.3))


def test_corruptor_delay_returns_past_value():
    c = ObservationCorruptor(noise_std=0.0, delay_steps=2, seed=0)
    outs = [c.apply((float(i), 0.0, 0.0))[0] for i in range(5)]
    # 履歴が溜まるまでは最古の値、その後は 2 ステップ前
    assert outs == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_corruptor_noise_statistics():
    c = ObservationCorruptor(noise_std=0.01, delay_steps=0, seed=0)
    xs = np.array([c.apply((0.0, 0.0, 0.0)) for _ in range(5000)])
    assert np.abs(xs.mean(axis=0)).max() < 0.001
    assert xs.std(axis=0) == pytest.approx([0.01, 0.01, 0.01], rel=0.1)


def test_corruptor_noise_only_xy_option():
    c = ObservationCorruptor(noise_std=0.01, delay_steps=0, seed=0, axes=(0, 1))
    xs = np.array([c.apply((0.0, 0.0, 0.5)) for _ in range(100)])
    assert np.all(xs[:, 2] == 0.5)


def test_corruptor_reset_clears_history():
    c = ObservationCorruptor(noise_std=0.0, delay_steps=3, seed=0)
    for i in range(5):
        c.apply((float(i), 0.0, 0.0))
    c.reset()
    assert c.apply((9.0, 0.0, 0.0))[0] == 9.0


def test_corruptor_bias_is_constant_within_trial_and_resampled_on_reset():
    c = ObservationCorruptor(noise_std=0.0, delay_steps=0, seed=0, bias_std=0.01)
    a = [c.apply((0.0, 0.0, 0.0)) for _ in range(3)]
    assert np.allclose(a[0], a[1]) and np.allclose(a[1], a[2])
    assert np.linalg.norm(a[0]) > 0
    c.reset()
    b = c.apply((0.0, 0.0, 0.0))
    assert not np.allclose(a[0], b)


def test_compose_observation_object_minus_ee():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'],
                          env_state_sources=['object_xyz', 'object_minus_ee'])
    joints = {n: float(i) for i, n in enumerate(ORDER)}          # j1=0, j2=1, j3=2, j4=3
    fk = lambda q: [q[0] + 0.1, q[1], 0.0]                        # 手先 = (0.1, 1.0, 0.0)
    obs = compose_observation(spec, joints, (0.3, 0.2, 0.1), fk=fk)
    assert obs['observation.environment_state'].tolist() == pytest.approx([0.3, 0.2, 0.1, 0.2, -0.8, 0.1])


def test_compose_observation_with_place():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'], env_state_sources=['object_xyz', 'place_xy'])
    joints = {n: 0.0 for n in ORDER}
    obs = compose_observation(spec, joints, (0.1, 0.2, 0.02), place_xy=(0.05, -0.1))
    assert obs['observation.environment_state'].tolist() == pytest.approx([0.1, 0.2, 0.02, 0.05, -0.1])
    # 置き場所が未受信なら観測は揃っていない
    assert compose_observation(spec, joints, (0.1, 0.2, 0.02)) is None


def test_object_latch_holds_while_gripper_closed():
    from pai_training.inference_utils import ObjectLatch
    latch = ObjectLatch(close=0.12, open_=0.0)
    assert latch.update((0.1, 0.0, 0.02), -0.5) == (0.1, 0.0, 0.02)
    assert latch.update((0.2, 0.0, 0.02), 0.3) == (0.1, 0.0, 0.02)     # 閉じた → 直前の値
    assert latch.update((0.3, 0.0, 0.02), 0.05) == (0.1, 0.0, 0.02)    # しきい値の間は閉のまま
    assert latch.update((0.4, 0.0, 0.02), -0.2) == (0.4, 0.0, 0.02)    # 開いた → 追跡に戻る
    latch.reset()
    assert latch.update((0.5, 0.0, 0.02), 0.3) == (0.5, 0.0, 0.02)     # 閉じた状態で始まったら最初の値
    assert latch.update(None, 0.3) == (0.5, 0.0, 0.02)


def test_compose_observation_with_latched_source():
    spec = ConversionSpec(joint_order=ORDER, state_sources=['joints'], env_state_sources=['object_xyz_latched'])
    obs = compose_observation(spec, {n: 0.0 for n in ORDER}, (0.1, 0.2, 0.02))
    assert obs['observation.environment_state'].tolist() == pytest.approx([0.1, 0.2, 0.02])

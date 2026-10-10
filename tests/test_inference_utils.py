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

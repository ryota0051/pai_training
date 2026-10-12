from pathlib import Path

import pytest

from pai_training.fk_loader import make_fk

PKG = Path('/home/ubuntu/final_task/ros2_ws/src/crane_plus_kinematics')


def test_no_kinematics_section_returns_none():
    assert make_fk({}) is None


@pytest.mark.skipif(not PKG.exists(), reason='crane_plus_kinematics がない環境')
def test_make_fk_returns_grasp_center_position():
    fk = make_fk({'kinematics': {'package': str(PKG), 'config': str(PKG / 'config' / 'kinematics.yaml')}})
    # ホーム姿勢付近の関節角（学習データの開始姿勢）→ 手先はホーム (0.10, 0.0, 0.08) 付近
    x, y, z = fk([-0.171, -0.439, -1.487, -1.657])
    assert abs(x - 0.10) < 0.01 and abs(y) < 0.01 and abs(z - 0.08) < 0.01

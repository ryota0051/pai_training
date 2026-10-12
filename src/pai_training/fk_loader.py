"""変換設定の kinematics 節から順運動学（手先位置）を作る（object_minus_ee 用）.

kinematics:
  package: /home/ubuntu/final_task/ros2_ws/src/crane_plus_kinematics   # ROS 非依存の運動学パッケージ
  config: /home/ubuntu/final_task/ros2_ws/src/crane_plus_kinematics/config/kinematics.yaml
"""
from __future__ import annotations

import sys
from typing import Mapping, Optional

from pai_training.conversion import ForwardKinematics


def make_fk(conv_cfg: Mapping) -> Optional[ForwardKinematics]:
    k = conv_cfg.get('kinematics')
    if not k:
        return None
    if k['package'] not in sys.path:
        sys.path.insert(0, k['package'])
    from crane_plus_kinematics.kinematics import load_kinematics
    kin = load_kinematics(k['config'])

    def fk(arm_joints):
        p = kin.forward(list(arm_joints))
        return (p.x, p.y, p.z)
    return fk

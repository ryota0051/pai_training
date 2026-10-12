#!/usr/bin/env python3
"""学習した ACT 方策を ROS 2 ノードとして実行する.

観測の組み立て（関節順・物体位置の有無）は変換設定（conversion_config）と共有し、学習時と一致させる。

Usage:
    pixi run infer --model-path <ckpt>                                       # 演習（お辞儀）互換
    pixi run infer --config configs/pick/infer_objpos.yaml [--model-path <ckpt>]

I/F:
    sub  /joint_states, （物体位置を使う場合）/task/object_pose
    pub  /crane_plus_arm_controller/joint_trajectory, /crane_plus_gripper_controller/joint_trajectory
    pub  /policy/status (std_msgs/String, JSON: enabled, latency_ms, predicted, step)
    srv  /policy/enable (std_srvs/SetBool), /policy/reset (std_srvs/Trigger)
    rerun: 予測チャンク・実関節角・物体位置をストリーミング（rerun.enabled: true のとき）
"""
import argparse
import json
import time

import numpy as np
import rclpy
import torch
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy
from pai_training.conversion import ConversionSpec
from pai_training.fk_loader import make_fk
from pai_training.inference_utils import ObservationCorruptor, TemporalEnsembler, compose_observation

DEFAULTS = {
    'device': 'cpu',
    'rate': 20.0,
    'trajectory_time': 0.1,
    'start_enabled': True,
    'temporal_ensemble_coeff': None,     # None: 毎ステップのチャンク先頭を実行（演習と同じ）
    'conversion_config': 'configs/conversion_config.yaml',
    'topics': {
        'arm_command': '/crane_plus_arm_controller/joint_trajectory',
        'gripper_command': '/crane_plus_gripper_controller/joint_trajectory',
    },
    # 物体位置の観測劣化（認識誤差・遅延への頑健性評価用）。既定は無効
    'object_corruption': {'noise_std': 0.0, 'bias_std': 0.0, 'delay_steps': 0, 'seed': None},
    'rerun': {'enabled': False, 'web_port': 9877, 'grpc_port': 9876, 'app_id': 'crane_plus_policy'},
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    return cfg


def make_trajectory(names, positions, seconds):
    jt = JointTrajectory()
    jt.joint_names = list(names)
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in positions]
    pt.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
    jt.points.append(pt)
    return jt


class InferenceNode(Node):
    def __init__(self, cfg, model_path):
        super().__init__('crane_plus_act_infer')
        with open(cfg['conversion_config']) as f:
            conv_cfg = yaml.safe_load(f)
        self.spec = ConversionSpec.from_dict(conv_cfg)
        self.fk = make_fk(conv_cfg)   # object_minus_ee を観測に使うモデルのみ
        self.device = torch.device(cfg['device'])
        self.traj_time = cfg['trajectory_time']
        self.enabled = bool(cfg['start_enabled'])

        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.policy.to(self.device)
        coeff = cfg['temporal_ensemble_coeff']
        self.ensembler = TemporalEnsembler(coeff, self.policy.config.chunk_size) if coeff is not None else None
        oc = cfg['object_corruption']
        self.corruptor = None
        if oc['noise_std'] > 0 or oc['delay_steps'] > 0 or oc.get('bias_std', 0) > 0:
            self.corruptor = ObservationCorruptor(oc['noise_std'], oc['delay_steps'], oc.get('seed'),
                                                  bias_std=oc.get('bias_std', 0.0))
            self.get_logger().info(f'object observation corruption: {oc}')

        topics = {**DEFAULTS['topics'], **self.spec.topics, **cfg['topics']}
        self.arm_pub = self.create_publisher(JointTrajectory, topics['arm_command'], 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, topics['gripper_command'], 10)
        self.status_pub = self.create_publisher(String, '/policy/status', 10)
        self.create_subscription(JointState, topics['joint_states'], self.on_joint_state, 10)
        if self.spec.needs_object:
            self.create_subscription(PoseStamped, topics['object_pose'], self.on_object, 10)
        self.create_service(SetBool, '/policy/enable', self.on_enable)
        self.create_service(Trigger, '/policy/reset', self.on_reset)

        self.joints = {}
        self.object_xyz = None
        self.observed_object = None
        self.step = 0
        self.last_latency_ms = None
        self.last_pred = None

        self.rr = None
        if cfg['rerun']['enabled']:
            import rerun as rr
            rr.init(cfg['rerun']['app_id'], spawn=False)
            rr.serve_web(open_browser=False, web_port=cfg['rerun']['web_port'],
                         grpc_port=cfg['rerun']['grpc_port'])
            self.rr = rr
            self.get_logger().info(
                f"rerun viewer: http://localhost:{cfg['rerun']['web_port']}"
                f"/?url=rerun%2Bhttp://localhost:{cfg['rerun']['grpc_port']}/proxy")

        self.create_timer(1.0 / cfg['rate'], self.on_timer)
        self.get_logger().info(f'policy loaded from {model_path} on {self.device} (enabled={self.enabled})')

    # ---- callbacks ----
    def on_joint_state(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def on_object(self, msg):
        p = msg.pose.position
        self.object_xyz = (p.x, p.y, p.z)

    def reset_policy(self):
        self.policy.reset()
        if self.ensembler is not None:
            self.ensembler.reset()
        if self.corruptor is not None:
            self.corruptor.reset()
        self.step = 0

    def on_enable(self, req, res):
        if req.data and not self.enabled:
            self.reset_policy()
        self.enabled = bool(req.data)
        res.success, res.message = True, f'enabled={self.enabled}'
        return res

    def on_reset(self, _req, res):
        self.reset_policy()
        res.success, res.message = True, 'reset'
        return res

    @torch.no_grad()
    def predict_chunk(self, obs):
        batch = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        batch = self.policy.normalize_inputs(batch)
        chunk = self.policy.model(batch)[0]
        chunk = self.policy.unnormalize_outputs({'action': chunk})['action']
        return chunk[0].cpu().numpy()  # (chunk_size, action_dim)

    def on_timer(self):
        object_xyz = self.object_xyz
        if object_xyz is not None and self.corruptor is not None and self.enabled:
            object_xyz = tuple(self.corruptor.apply(object_xyz))
        self.observed_object = object_xyz
        obs = compose_observation(self.spec, self.joints, object_xyz, fk=self.fk)
        if obs is None or not self.enabled:
            self.publish_status()
            return
        t0 = time.perf_counter()
        chunk = self.predict_chunk(obs)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        action = self.ensembler.update(chunk) if self.ensembler is not None else chunk[0]
        self.last_pred = action

        self.arm_pub.publish(make_trajectory(self.spec.arm_joints, action[:-1], self.traj_time))
        self.gripper_pub.publish(make_trajectory([self.spec.gripper_joint], action[-1:], self.traj_time))
        self.log_rerun(obs, chunk, action)
        self.step += 1
        self.publish_status()

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps({
            'enabled': self.enabled,
            'ready': compose_observation(self.spec, self.joints, self.object_xyz, fk=self.fk) is not None,
            'step': self.step,
            'latency_ms': None if self.last_latency_ms is None else round(self.last_latency_ms, 2),
            'predicted': None if self.last_pred is None else [round(float(v), 4) for v in self.last_pred],
            'device': str(self.device),
        })))

    def log_rerun(self, obs, chunk, action):
        if self.rr is None:
            return
        rr = self.rr
        rr.set_time('step', sequence=self.step)
        names = self.spec.joint_order
        current = [self.joints[n] for n in names]
        for i, n in enumerate(names):
            short = n.replace('crane_plus_', '')
            rr.log(f'joints/{short}/actual', rr.Scalars(current[i]))
            rr.log(f'joints/{short}/command', rr.Scalars(float(action[i])))
            # 予測チャンク（未来 chunk_size ステップ）を折れ線で表示
            pts = np.stack([np.arange(len(chunk)), chunk[:, i]], axis=1)
            rr.log(f'chunk/{short}', rr.LineStrips2D([pts]))
        if self.object_xyz is not None:
            rr.log('object/xyz', rr.Points3D([self.object_xyz], radii=0.02))
            for axis, v, o in zip('xyz', self.object_xyz, self.observed_object):
                rr.log(f'object/{axis}/true', rr.Scalars(v))
                rr.log(f'object/{axis}/observed', rr.Scalars(float(o)))
        rr.log('latency_ms', rr.Scalars(self.last_latency_ms))


def main():
    parser = argparse.ArgumentParser(description='ACTPolicy inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--config', default=None, help='推論設定 YAML（省略時は演習と同じ既定値）')
    parser.add_argument('--model-path', default=None, help='学習済み方策ディレクトリ（設定の model_path を上書き）')
    parser.add_argument('--device', default=None)
    parser.add_argument('--temporal-ensemble', type=float, default=None,
                        help='temporal ensembling の係数（例: 0.01）。設定ファイルの値を上書き')
    parser.add_argument('--object-noise', type=float, default=None, help='物体位置ノイズ標準偏差 [m]')
    parser.add_argument('--object-bias', type=float, default=None, help='試行ごとの物体位置オフセット標準偏差 [m]')
    parser.add_argument('--object-delay', type=int, default=None, help='物体位置の遅延 [制御ステップ]')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.temporal_ensemble is not None:
        cfg['temporal_ensemble_coeff'] = args.temporal_ensemble
    if args.object_noise is not None:
        cfg['object_corruption']['noise_std'] = args.object_noise
    if args.object_bias is not None:
        cfg['object_corruption']['bias_std'] = args.object_bias
    if args.object_delay is not None:
        cfg['object_corruption']['delay_steps'] = args.object_delay
    model_path = args.model_path or cfg.get('model_path')
    if not model_path:
        parser.error('--model-path か設定の model_path が必要です')

    rclpy.init()
    node = InferenceNode(cfg, model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

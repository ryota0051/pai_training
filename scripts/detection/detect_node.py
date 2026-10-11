"""学習した物体検出器（RT-DETRv2）で俯瞰カメラ画像から物体位置を推定する ROS ノード（ADR 0008）.

source /opt/ros/humble/setup.bash
pixi run --frozen detect-node --config configs/detection/detect_rtdetr.yaml
赤色検出ノード（pai_task_env/object_detector_node）と同じトピックに出力するので、
タスク環境は red_detector:=false で起動し、推論ノードはカメラ推定版の設定（/task/object_pose_est）で動かす。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection


class DetectNode(Node):
    def __init__(self, cfg):
        super().__init__('learned_object_detector')
        sys.path.insert(0, cfg['pai_task_env_path'])
        from pai_task_env.domain_randomization import bbox_center_to_position
        from pai_task_env.vision import CameraModel, EstimateHolder
        self.to_position = bbox_center_to_position
        cam_all = yaml.safe_load(open(cfg['camera_config']))['object_detector_node']['ros__parameters']
        c = cam_all['camera']
        self.cam = CameraModel.from_hfov(c['width'], c['height'], c['hfov'], c['position'], c['rpy'])
        self.size = (c['width'], c['height'])
        self.r, self.h = cfg['object_size'] / 2, cfg['object_size']
        self.zc = cfg['object_center_z']
        self.threshold = cfg['score_threshold']
        self.holder = EstimateHolder(hold_last=cam_all['hold_last'], min_visible_ratio=cam_all['min_visible_ratio'])
        self.device = torch.device(cfg['device'])
        self.processor = RTDetrImageProcessor.from_pretrained(cfg['model_dir'])
        self.model = RTDetrV2ForObjectDetection.from_pretrained(cfg['model_dir']).to(self.device).eval()
        self.episode = None
        self.pub = self.create_publisher(PoseStamped, cfg['output_topic'], 10)
        self.status_pub = self.create_publisher(String, '/task/detector_status', 10)
        self.create_subscription(Image, cfg['image_topic'], self.on_image, 2)
        self.create_subscription(String, '/task/status',
                                 lambda m: setattr(self, 'episode', json.loads(m.data).get('episode')), 10)
        self.get_logger().info(f"learned detector: {cfg['image_topic']} -> {cfg['output_topic']} ({cfg['model_dir']})")

    @torch.no_grad()
    def detect(self, rgb):
        img = PILImage.fromarray(rgb)
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(out, threshold=self.threshold,
                                                           target_sizes=[(img.height, img.width)])[0]
        if not len(res['scores']):
            return None, 0.0
        k = int(res['scores'].argmax())
        return res['boxes'][k].tolist(), float(res['scores'][k])

    def on_image(self, msg: Image):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        if msg.encoding == 'bgr8':
            img = img[..., ::-1]
        box, score = self.detect(np.ascontiguousarray(img))
        detection, area = None, 0
        if box is not None:
            u0, v0, u1, v1 = box
            x, y = self.to_position(self.cam, ((u0 + u1) / 2, (v0 + v1) / 2), self.r, self.h, self.size,
                                    center_z=self.zc)
            detection, area = (x, y, self.zc), int((u1 - u0) * (v1 - v0))
        # 枠の面積を「見えている量」として部分遮蔽の判定に使う（赤色検出の画素数と同じ扱い）
        estimate = self.holder.update(detection, self.episode, pixels=area)
        if estimate is not None:
            ps = PoseStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = 'crane_plus_base'
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = estimate
            ps.pose.orientation.w = 1.0
            self.pub.publish(ps)
        self.status_pub.publish(String(data=json.dumps({
            'detected': box is not None, 'score': round(score, 3), 'pixels': area, 'episode': self.episode,
            'detected_in_episode': self.holder.detected_in_episode,
            'partially_occluded': self.holder.partially_occluded,
            'estimate': None if estimate is None else [round(v, 4) for v in estimate]})))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rclpy.init()
    node = DetectNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

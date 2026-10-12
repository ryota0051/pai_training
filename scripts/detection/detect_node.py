"""学習した物体検出器（RT-DETRv2）で俯瞰カメラ画像から物体位置を推定する ROS ノード（ADR 0008）.

source /opt/ros/humble/setup.bash
pixi run --frozen detect-node --config configs/detection/detect_rtdetr.yaml
赤色検出ノード（pai_task_env/object_detector_node）と同じトピックに出力するので、
タスク環境は red_detector:=false で起動し、推論ノードはカメラ推定版の設定（/task/object_pose_est）で動かす。
設定に place があれば、置き場所（2 クラス目）の位置も推定して place.output_topic へ出す（実験12）。
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

from pai_training.detection.coco import RecentMedian, best_per_class


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
        self.label = cfg.get('object_label', 0)
        # 採用した検出の直近 median_window 回の中央値を出す（一部が隠れた瞬間の外れ値を除く。1 なら従来どおり）
        self.median = RecentMedian(cfg.get('median_window', 1))
        # 置き場所（厚さ 1mm の円板）。腕に隠れても動かないので、最後の推定値を保持する
        self.place = cfg.get('place')
        if self.place:
            self.place_holder = EstimateHolder(hold_last=True, min_visible_ratio=cam_all['min_visible_ratio'])
        self.device = torch.device(cfg['device'])
        self.processor = RTDetrImageProcessor.from_pretrained(cfg['model_dir'])
        self.model = RTDetrV2ForObjectDetection.from_pretrained(cfg['model_dir']).to(self.device).eval()
        self.episode = None
        self.pub = self.create_publisher(PoseStamped, cfg['output_topic'], 10)
        if self.place:
            self.place_pub = self.create_publisher(PoseStamped, self.place['output_topic'], 10)
        self.status_pub = self.create_publisher(String, cfg.get('status_topic', '/task/detector_status'), 10)
        self.create_subscription(Image, cfg['image_topic'], self.on_image, 2)
        self.create_subscription(String, '/task/status',
                                 lambda m: setattr(self, 'episode', json.loads(m.data).get('episode')), 10)
        self.get_logger().info(f"learned detector: {cfg['image_topic']} -> {cfg['output_topic']} ({cfg['model_dir']})")

    @torch.no_grad()
    def detect(self, rgb):
        """クラスごとに最もスコアの高い枠 → {label: (box, score)}."""
        img = PILImage.fromarray(rgb)
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(out, threshold=self.threshold,
                                                           target_sizes=[(img.height, img.width)])[0]
        return best_per_class(res['scores'].tolist(), res['labels'].tolist(), res['boxes'].tolist())

    def publish_pose(self, pub, stamp, xyz):
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = 'crane_plus_base'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.w = 1.0
        pub.publish(ps)

    def update_place(self, found, stamp):
        """置き場所の枠から中心位置を求めて publish する. 推定値（なければ None）を返す."""
        detection, area = None, 0
        if found is not None:
            (u0, v0, u1, v1), _ = found
            pl = self.place
            x, y = self.to_position(self.cam, ((u0 + u1) / 2, (v0 + v1) / 2), pl['radius'], pl['height'],
                                    self.size, center_z=pl['center_z'])
            detection, area = (x, y, pl['center_z']), int((u1 - u0) * (v1 - v0))
        estimate = self.place_holder.update(detection, self.episode, pixels=area)
        if estimate is not None:
            self.publish_pose(self.place_pub, stamp, estimate)
        return estimate

    def on_image(self, msg: Image):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        if msg.encoding == 'bgr8':
            img = img[..., ::-1]
        found = self.detect(np.ascontiguousarray(img))
        box, score = found.get(self.label, (None, 0.0))
        detection, area = None, 0
        if box is not None:
            u0, v0, u1, v1 = box
            x, y = self.to_position(self.cam, ((u0 + u1) / 2, (v0 + v1) / 2), self.r, self.h, self.size,
                                    center_z=self.zc)
            detection, area = (x, y, self.zc), int((u1 - u0) * (v1 - v0))
        # 枠の面積を「見えている量」として部分遮蔽の判定に使う（赤色検出の画素数と同じ扱い）
        estimate = self.holder.update(detection, self.episode, pixels=area)
        accepted = detection if detection is not None and not self.holder.partially_occluded else None
        median = self.median.update(accepted, self.episode)
        if estimate is not None and median is not None:
            estimate = median
        if estimate is not None:
            self.publish_pose(self.pub, msg.header.stamp, estimate)
        status = {
            'detected': box is not None, 'score': round(score, 3), 'pixels': area, 'episode': self.episode,
            'detected_in_episode': self.holder.detected_in_episode,
            'partially_occluded': self.holder.partially_occluded,
            'estimate': None if estimate is None else [round(v, 4) for v in estimate]}
        if self.place:
            place = self.update_place(found.get(self.place['label']), msg.header.stamp)
            # 評価器は detected_in_episode を待ってから方策を始めるので、置き場所も見つけるまで待たせる
            status['detected_in_episode'] = status['detected_in_episode'] and self.place_holder.detected_in_episode
            status['place_detected'] = self.place['label'] in found
            status['place_estimate'] = None if place is None else [round(v, 4) for v in place]
        self.status_pub.publish(String(data=json.dumps(status)))


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

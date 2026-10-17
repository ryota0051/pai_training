"""カメラ画像（sensor_msgs/Image）の変換. 変換（rosbag）と推論ノードで共有する（torch / ROS に依存しない）."""
from __future__ import annotations

import numpy as np

# encoding → (チャンネル数, RGB として取り出す並び)
_ENCODINGS = {
    'rgb8': (3, [0, 1, 2]),
    'bgr8': (3, [2, 1, 0]),
    'rgba8': (4, [0, 1, 2]),
    'bgra8': (4, [2, 1, 0]),
    'mono8': (1, [0, 0, 0]),
}


def decode_image(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """sensor_msgs/Image の生データ → (H, W, 3) uint8 の RGB. 行末の詰め物（step > width * ch）は捨てる."""
    if encoding not in _ENCODINGS:
        raise ValueError(f'unsupported image encoding: {encoding} (choose from {list(_ENCODINGS)})')
    ch, order = _ENCODINGS[encoding]
    buf = np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, step)
    img = buf[:, :width * ch].reshape(height, width, ch)
    return np.ascontiguousarray(img[..., order])


def to_chw_float(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 → (3, H, W) float32 [0, 1]（LeRobotDataset が学習時に返す形と同じ）."""
    return np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1), dtype=np.float32) / 255.0

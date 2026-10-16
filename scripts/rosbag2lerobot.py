#!/usr/bin/env python3
"""rosbag2 → LeRobotDataset v2.1 変換.

変換ロジックは pai_training.conversion（純粋関数・テスト済み）にあり、ここでは
rosbag の読み出しと LeRobotDataset への書き込みのみを行う。

Usage:
    pixi run convert                                   # configs/conversion_config.yaml（演習のお辞儀）
    pixi run convert --config configs/pick/conversion_objpos.yaml [--overwrite]
    pixi run convert --config configs/pick/conversion_image.yaml [--overwrite]   # カメラ画像を観測に使う

カメラ画像（observation.images）を使う場合、画像は大きいので 1 回目の読み出しでは時刻だけを集め、
書き込み時にエピソードごとに読み直す（全エピソードの画像をメモリに載せない）。
"""
import argparse
import importlib.util
import shutil
from pathlib import Path

import torch
import yaml
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pai_training.conversion import IMAGE_FEATURE_PREFIX, IMAGE_INDEX_PREFIX, ConversionSpec, build_episode_arrays
from pai_training.images import decode_image
from pai_training.fk_loader import make_fk
from pai_training.episode_discovery import find_episode_dirs


def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def read_messages(episode_dir: Path, spec: ConversionSpec):
    """必要なトピックを (timestamp_ns, names, values) のリストで返す. カメラ画像は時刻 [ns] のリストのみ."""
    topics = {k: v for k, v in spec.topics.items() if v}
    out = {k: [] for k in topics}
    by_topic = {v: k for k, v in topics.items()}
    with Reader(episode_dir) as reader:
        conns = [c for c in reader.connections if c.topic in by_topic]
        for conn, ts, raw in reader.messages(connections=conns):
            key = by_topic[conn.topic]
            if key in spec.image_sources:   # 画像は展開しない（時刻だけ）
                out[key].append(ts)
                continue
            msg = deserialize_cdr(raw, conn.msgtype)
            if key == 'joint_states':
                out[key].append((ts, list(msg.name), list(msg.position)))
            elif key in ('arm_command', 'gripper_command'):
                if msg.points:
                    out[key].append((ts, list(msg.joint_names), list(msg.points[0].positions)))
            elif key in ('object_pose', 'place_pose'):
                p = msg.pose.position
                out[key].append((ts, None, [p.x, p.y, p.z]))
    return out


def read_images(episode_dir: Path, spec: ConversionSpec):
    """カメラ名 → (H, W, 3) uint8 の RGB 画像のリスト（read_messages の時刻と同じ順）."""
    by_topic = {spec.topics[cam]: cam for cam in spec.image_sources}
    out = {cam: [] for cam in spec.image_sources}
    with Reader(episode_dir) as reader:
        conns = [c for c in reader.connections if c.topic in by_topic]
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = deserialize_cdr(raw, conn.msgtype)
            out[by_topic[conn.topic]].append(decode_image(msg.encoding, msg.height, msg.width, msg.step, msg.data))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/conversion_config.yaml')
    parser.add_argument('--overwrite', action='store_true', help='出力先が存在する場合に確認なしで作り直す')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conv_cfg = yaml.safe_load(f)
    spec = ConversionSpec.from_dict(conv_cfg)
    fk = make_fk(conv_cfg)   # object_minus_ee を使う場合のみ
    features_cfg = load_config_py(conv_cfg['config'])
    bag_dir = conv_cfg['bag_dir']   # 文字列、または複数ディレクトリのリスト（順に連結）
    output_dir = Path(conv_cfg['output_dir']).expanduser().resolve()

    episode_dirs = find_episode_dirs(bag_dir)
    if not episode_dirs:
        raise FileNotFoundError(f'No rosbag2 episode directories found in {bag_dir}')
    print(f'Found {len(episode_dirs)} episodes in {bag_dir}')

    if output_dir.exists():
        if not args.overwrite:
            ans = input(f'{output_dir} exists. Overwrite? (y/N): ').strip().lower()
            if ans != 'y':
                return
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=getattr(features_cfg, 'repo_id', 'teleop_rosbag_dataset'),
        fps=int(spec.target_freq),
        root=output_dir,
        robot_type=getattr(features_cfg, 'robot_type', 'crane_plus'),
        features=features_cfg.features,
        use_videos=True,
    )
    task_name = getattr(features_cfg, 'task_name', 'teleop')

    episodes = []
    for ep_dir in episode_dirs:
        try:
            msgs = read_messages(ep_dir, spec)
            arrays = build_episode_arrays(spec, msgs['joint_states'], msgs['arm_command'],
                                          msgs['gripper_command'], msgs.get('object_pose'), fk=fk,
                                          place_poses=msgs.get('place_pose'),
                                          image_times={cam: msgs[cam] for cam in spec.image_sources})
            episodes.append((ep_dir, arrays))
            print(f'  {ep_dir.name}: {len(arrays["action"])} frames')
        except Exception as e:  # 1 エピソードの不備で全体を止めない
            print(f'  [skip] {ep_dir}: {e}')

    max_frames = max(len(a['action']) for _, a in episodes)
    for idx, (ep_dir, arrays) in enumerate(episodes):
        n = len(arrays['action'])
        total = max_frames if spec.pad_to_longest else n
        images = read_images(ep_dir, spec) if spec.image_sources else {}
        for i in range(total):
            j = min(i, n - 1)  # 演習と同様、短いエピソードは最終フレームを保持して延長
            frame = {
                'observation.state': torch.from_numpy(arrays['observation.state'][j]),
                'action': torch.from_numpy(arrays['action'][j]),
                'task': task_name,
            }
            if spec.has_env_state:
                frame['observation.environment_state'] = torch.from_numpy(arrays['observation.environment_state'][j])
            for cam in spec.image_sources:
                frame[IMAGE_FEATURE_PREFIX + cam] = images[cam][arrays[IMAGE_INDEX_PREFIX + cam][j]]
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f'saved episode {idx} ({ep_dir.name})')
    print(f'Dataset saved to: {dataset.root}')


if __name__ == '__main__':
    main()

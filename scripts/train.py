# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""
Train ACT (Action Chunking with Transformer) policy on the collected dataset.

Usage:
    pixi run train                                         # configs/train_config.yaml（演習）
    pixi run train --config configs/pick/train_objpos.yaml # 把持・配置タスク
    pixi run train --config ... --device cpu               # CPU/GPU 比較用に上書き

追加した設定項目（省略時は演習と同じ挙動）:
    wandb_mode: online | offline | disabled
    seed: int
    episodes: [0, 1, ...]  学習に使うエピソード（null なら全て）
    act: {...}             ACTConfig への追加引数（dim_model, n_encoder_layers, kl_weight など）
学習時間などは <output_directory>/train_info.json に保存する。
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import wandb
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--device', default=None, help='設定ファイルの device を上書き')
    parser.add_argument('--steps', type=int, default=None, help='設定ファイルの training_steps を上書き')
    parser.add_argument('--output', default=None, help='設定ファイルの output_directory を上書き')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(cfg["dataset_root"])
    repo_id = cfg["repo_id"]
    output_directory = Path(args.output or cfg["output_directory"])
    device = torch.device(args.device or cfg["device"])
    training_steps = args.steps or cfg["training_steps"]
    log_freq = cfg["log_freq"]
    episodes = cfg.get("episodes")
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])

    output_directory.mkdir(parents=True, exist_ok=True)
    features = load_config_py(Path(cfg["config_path"])).features

    run = wandb.init(project=cfg.get("wandb_project", "pai_crane_gazebo"),
                     mode=cfg.get("wandb_mode", "online"),
                     config={**cfg, "device": str(device), "training_steps": training_steps})

    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=cfg["chunk_size"],
        n_action_steps=cfg["n_action_steps"],
        **(cfg.get("act") or {}),
    )
    policy = ACTPolicy(policy_cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=episodes,
                             delta_timestamps=cfg["delta_timestamps"])
    print(f"dataset: {dataset.num_episodes} episodes, {len(dataset)} frames")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg.get("weight_decay", 0.0))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    step = 0
    done = False
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            # Remove the sequence_length dimension to adapt to the ACT model implementation
            batch["observation.state"] = batch["observation.state"][:, -1, :]  # (B, D)
            batch["observation.environment_state"] = batch["observation.environment_state"][:, -1, :]  # (B, D)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(), **{k: v for k, v in loss_dict.items()}, "step": step})
            step += 1
            if step >= training_steps:
                done = True
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    policy.save_pretrained(output_directory)
    info = {
        "config": args.config,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_steps": training_steps,
        "batch_size": cfg["batch_size"],
        "train_seconds": elapsed,
        "sec_per_step": elapsed / max(training_steps, 1),
        "final_loss": loss.item(),
        "num_episodes": dataset.num_episodes,
        "num_frames": len(dataset),
    }
    (output_directory / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"policy saved to {output_directory} ({elapsed:.1f}s, {info['sec_per_step'] * 1000:.1f} ms/step)")
    run.finish()


if __name__ == "__main__":
    main()

"""COCO 形式の正解を RT-DETR（transformers）の学習用ラベルに変換する."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def to_target(annotations: Sequence[dict], width: int, height: int) -> Dict[str, torch.Tensor]:
    """[{'bbox': [x, y, w, h], 'category_id': c}] → {'class_labels', 'boxes'（正規化 cx, cy, w, h）}."""
    boxes = [[(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
             for x, y, w, h in (a['bbox'] for a in annotations)]
    return {
        'class_labels': torch.tensor([a['category_id'] for a in annotations], dtype=torch.long),
        'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    }


def flip_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """左右反転した画像に対応するラベル."""
    boxes = target['boxes'].clone()
    boxes[:, 0] = 1.0 - boxes[:, 0]
    return {'class_labels': target['class_labels'].clone(), 'boxes': boxes}


def split_ids(ids: Iterable[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_ratio))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


class RecentMedian:
    """直近 window 回の検出の各軸の中央値（一部が隠れた瞬間の外れ値を除く）. 検出がなければ直前の値を返す.

    エピソードが変わったら履歴を捨てる。
    """

    def __init__(self, window: int):
        self.window = window
        self.history: List[Tuple[float, ...]] = []
        self.episode = None
        self.value = None

    def update(self, detection, episode):
        if episode != self.episode:
            self.episode, self.history, self.value = episode, [], None
        if detection is not None:
            self.history = (self.history + [tuple(detection)])[-self.window:]
            cols = list(zip(*self.history))
            self.value = tuple(float(sorted(c)[len(c) // 2]) for c in cols)
        return self.value


def best_per_class(scores: Sequence[float], labels: Sequence[int],
                   boxes: Sequence[Sequence[float]]) -> Dict[int, Tuple[List[float], float]]:
    """検出結果からクラスごとに最もスコアの高い枠を選ぶ → {label: (box, score)}."""
    best: Dict[int, Tuple[List[float], float]] = {}
    for s, c, b in zip(scores, labels, boxes):
        s, c = float(s), int(c)
        if c not in best or s > best[c][1]:
            best[c] = (list(b), s)
    return best

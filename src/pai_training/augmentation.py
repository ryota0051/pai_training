"""学習時の画像の拡張. 学習設定の image_transforms 節 → LeRobot の ImageTransforms.

設定の例（configs/pick/train_image_color.yaml）:
    image_transforms:
      enable: true
      max_num_transforms: 3       # 1 フレームにかける拡張の最大数（tfs から重みに従って選ぶ）
      random_order: false
      tfs:                        # 省略すると LeRobot の既定（明るさ・コントラスト・彩度・色相・シャープネス）
        brightness: {type: ColorJitter, kwargs: {brightness: [0.5, 1.2]}}
type は LeRobot が扱える Identity / ColorJitter / SharpnessJitter のみ。
"""
from __future__ import annotations

from typing import Mapping, Optional

from lerobot.common.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig

_TYPES = ('Identity', 'ColorJitter', 'SharpnessJitter')


def _transform_config(name: str, cfg: Mapping) -> ImageTransformConfig:
    tf_type = cfg.get('type', 'Identity')
    if tf_type not in _TYPES:
        raise ValueError(f'image transform "{name}": unsupported type {tf_type} (choose from {_TYPES})')
    # yaml のリストは torchvision が受け取れるタプルにする
    kwargs = {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in (cfg.get('kwargs') or {}).items()}
    return ImageTransformConfig(weight=float(cfg.get('weight', 1.0)), type=tf_type, kwargs=kwargs)


def make_image_transforms(cfg: Optional[Mapping]) -> Optional[ImageTransforms]:
    """image_transforms 節から ImageTransforms を作る. 節がない・enable が偽なら None（拡張なし）."""
    if not cfg or not cfg.get('enable', False):
        return None
    tf_cfg = ImageTransformsConfig(enable=True,
                                   max_num_transforms=int(cfg.get('max_num_transforms', 3)),
                                   random_order=bool(cfg.get('random_order', False)))
    if cfg.get('tfs'):
        tf_cfg.tfs = {name: _transform_config(name, t) for name, t in cfg['tfs'].items()}
    return ImageTransforms(tf_cfg)

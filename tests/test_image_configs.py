"""カメラ画像を観測に使う設定（configs/pick/*_image.*）が互いに整合しているか."""
import importlib.util
from pathlib import Path

import yaml

from pai_training.conversion import IMAGE_FEATURE_PREFIX, ConversionSpec

ROOT = Path(__file__).resolve().parents[1]


def _yaml(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))


def _features(rel):
    spec = importlib.util.spec_from_file_location('features', ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_image_configs_are_consistent():
    conv = _yaml('configs/pick/conversion_image.yaml')
    train = _yaml('configs/pick/train_image.yaml')
    infer = _yaml('configs/pick/infer_image.yaml')
    spec = ConversionSpec.from_dict(conv)
    feats = _features(conv['config'])

    # A 案: 関節角 + 画像のみ
    assert spec.state_sources == ['joints'] and not spec.has_env_state
    assert not spec.needs_object and not spec.needs_place
    image_keys = {IMAGE_FEATURE_PREFIX + cam for cam in spec.image_sources}
    assert image_keys and image_keys == {k for k in feats.features if k.startswith(IMAGE_FEATURE_PREFIX)}
    assert 'observation.environment_state' not in feats.features
    # 学習は全ての観測を 1 時刻ずつ読む
    assert set(train['delta_timestamps']) == set(feats.features)
    assert train['config_path'] == conv['config']
    assert train['dataset_root'] == conv['output_dir']
    assert train['repo_id'] == feats.repo_id
    assert infer['conversion_config'] == 'configs/pick/conversion_image.yaml'
    assert infer['model_path'] == train['output_directory']

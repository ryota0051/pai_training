"""学習時の画像の拡張（設定 → LeRobot の ImageTransforms）."""
import pytest

torch = pytest.importorskip('torch')

from pai_training.augmentation import make_image_transforms  # noqa: E402

COLOR = {
    'enable': True,
    'max_num_transforms': 2,
    'tfs': {
        'brightness': {'type': 'ColorJitter', 'kwargs': {'brightness': [0.5, 1.2]}},
        'hue': {'type': 'ColorJitter', 'weight': 0.5, 'kwargs': {'hue': [-0.03, 0.03]}},
    },
}


def test_disabled_or_missing_returns_none():
    assert make_image_transforms(None) is None
    assert make_image_transforms({}) is None
    assert make_image_transforms({'enable': False, 'tfs': COLOR['tfs']}) is None


def test_builds_configured_transforms_only():
    tf = make_image_transforms(COLOR)
    assert list(tf.transforms) == ['brightness', 'hue']
    assert tf.weights == [1.0, 0.5]
    assert tf._cfg.max_num_transforms == 2
    # yaml のリストは torchvision が受け取れるタプルにする
    assert tf._cfg.tfs['brightness'].kwargs['brightness'] == (0.5, 1.2)


def test_without_tfs_uses_lerobot_defaults():
    tf = make_image_transforms({'enable': True})
    assert set(tf.transforms) == {'brightness', 'contrast', 'saturation', 'hue', 'sharpness'}


def test_keeps_shape_and_range():
    torch.manual_seed(0)
    img = torch.rand(3, 24, 32)
    out = make_image_transforms(COLOR)(img)
    assert out.shape == img.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        make_image_transforms({'enable': True, 'tfs': {'x': {'type': 'RandomCrop'}}})

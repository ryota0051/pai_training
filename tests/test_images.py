import numpy as np
import pytest

from pai_training.images import decode_image, to_chw_float


def _raw(img: np.ndarray, pad: int = 0) -> bytes:
    """行末に pad バイトの詰め物を入れた生データ（sensor_msgs/Image の step を模す）."""
    rows = [r.tobytes() + b'\x00' * pad for r in img]
    return b''.join(rows)


def test_decode_rgb8():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = decode_image('rgb8', 2, 3, 9, _raw(img))
    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    assert np.array_equal(out, img)


def test_decode_bgr8_converts_to_rgb():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = decode_image('bgr8', 2, 3, 9, _raw(img))
    assert np.array_equal(out, img[..., ::-1])


def test_decode_rgba8_drops_alpha():
    img = np.arange(2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 4)
    out = decode_image('rgba8', 2, 2, 8, _raw(img))
    assert np.array_equal(out, img[..., :3])


def test_decode_mono8_repeats_channel():
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = decode_image('mono8', 2, 3, 3, _raw(img))
    assert out.shape == (2, 3, 3)
    assert np.array_equal(out[..., 1], img)


def test_decode_handles_row_padding():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = decode_image('rgb8', 2, 3, 12, _raw(img, pad=3))
    assert np.array_equal(out, img)


def test_decode_unknown_encoding_raises():
    with pytest.raises(ValueError):
        decode_image('16UC1', 1, 1, 2, b'\x00\x00')


def test_to_chw_float():
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[..., 0] = 255
    out = to_chw_float(img)
    assert out.dtype == np.float32 and out.shape == (3, 2, 3)
    assert out[0].tolist() == [[1.0] * 3] * 2
    assert out[1].max() == 0.0

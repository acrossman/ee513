import numpy as np

from isp.bilinear import bilinear
from isp.mhc import mhc


def mosaic_of(rgb):
    img = np.zeros(rgb.shape[:2])
    img[0::2, 0::2] = rgb[0::2, 0::2, 0]
    img[0::2, 1::2] = rgb[0::2, 1::2, 1]
    img[1::2, 0::2] = rgb[1::2, 0::2, 1]
    img[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return img


def constant_image():
    rgb = np.zeros((16, 16, 3))
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = 0.2, 0.5, 0.8
    return rgb


def test_bilinear_reproduces_constant_image():
    rgb = constant_image()
    assert np.allclose(bilinear(mosaic_of(rgb)), rgb)


def test_mhc_reproduces_constant_image_away_from_border():
    rgb = constant_image()
    assert np.allclose(mhc(mosaic_of(rgb))[2:-2, 2:-2], rgb[2:-2, 2:-2])


def test_demosaic_keeps_measured_samples():
    rng = np.random.default_rng(0)
    img = mosaic_of(rng.random((16, 16, 3)))
    for method in (bilinear, mhc):
        out = method(img)
        assert np.allclose(out[0::2, 0::2, 0], img[0::2, 0::2])
        assert np.allclose(out[0::2, 1::2, 1], img[0::2, 1::2])
        assert np.allclose(out[1::2, 0::2, 1], img[1::2, 0::2])
        assert np.allclose(out[1::2, 1::2, 2], img[1::2, 1::2])

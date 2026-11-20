import numpy as np

from isp.mosaic import mosaic_rggb, quarter_reference


def test_quarter_reference_takes_red_mean_green_blue():
    mosaic = np.array([[1.0, 2.0, 5.0, 6.0],
                       [4.0, 3.0, 8.0, 7.0]])
    rgb = quarter_reference(mosaic)
    assert rgb.shape == (1, 2, 3)
    assert np.array_equal(rgb[0, 0], [1.0, 3.0, 3.0])
    assert np.array_equal(rgb[0, 1], [5.0, 7.0, 7.0])


def test_mosaic_rggb_keeps_one_color_per_site():
    rgb = np.random.default_rng(6).random((6, 8, 3))
    img = mosaic_rggb(rgb)
    assert np.array_equal(img[0::2, 0::2], rgb[0::2, 0::2, 0])
    assert np.array_equal(img[0::2, 1::2], rgb[0::2, 1::2, 1])
    assert np.array_equal(img[1::2, 0::2], rgb[1::2, 0::2, 1])
    assert np.array_equal(img[1::2, 1::2], rgb[1::2, 1::2, 2])

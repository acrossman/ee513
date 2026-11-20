import numpy as np

from isp.defects import correct_defects, repeatable_hot_sites, stuck_sites
from isp.pipeline import black_level


def test_black_level_clips_to_zero_and_one():
    mosaic = np.array([[60.0, 64.0], [1023.0, 2000.0]])
    out = black_level(mosaic, [64, 64, 64, 64])
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[1, 0] == 1 and out[1, 1] == 1


def test_repeatable_hot_sites_finds_a_planted_hot_pixel_only():
    rng = np.random.default_rng(7)
    frames = 64 + rng.normal(0, 2, (5, 64, 64))
    frames[:, 10, 20] += 40
    frames[3, 30, 30] += 60
    hot = repeatable_hot_sites(frames)
    assert hot[10, 20]
    assert not hot[30, 30]


def test_stuck_sites_finds_a_constant_pixel():
    rng = np.random.default_rng(8)
    frames = np.round(64 + rng.normal(0, 20, (5, 32, 32)))
    frames[:, 5, 7] = 80
    stuck = stuck_sites(frames)
    assert stuck[5, 7] and stuck.sum() == 1


def test_correct_defects_uses_same_color_neighbours():
    mosaic = np.zeros((8, 8))
    mosaic[0::2, 0::2] = 10.0
    mosaic[4, 4] = 999.0
    defective = np.zeros((8, 8), dtype=bool)
    defective[4, 4] = True
    out = correct_defects(mosaic, defective)
    assert out[4, 4] == 10.0
    assert np.array_equal(np.delete(out.ravel(), 4 * 8 + 4), np.delete(mosaic.ravel(), 4 * 8 + 4))


def test_normalized_flat_is_one_at_the_center_and_corrects_itself():
    from isp.pipeline import lens_shading, normalized_flat
    rows, cols = np.mgrid[0:80, 0:120]
    falloff = 1 - 0.3 * (((rows - 40) / 40) ** 2 + ((cols - 60) / 60) ** 2)
    flat = falloff * 0.4
    flat[0::2, 0::2] *= 0.6
    planes = normalized_flat(flat)
    assert np.allclose(planes[:, 18:22, 28:32].mean(axis=(1, 2)), 1, atol=0.01)
    corrected = lens_shading(flat, planes)
    for p in range(4):
        site = [(0, 0), (0, 1), (1, 0), (1, 1)][p]
        values = corrected[site[0]::2, site[1]::2]
        assert np.allclose(values, values[20, 30])


def test_white_balance_mosaic_scales_each_site_and_clips():
    from isp.pipeline import white_balance_mosaic
    mosaic = np.full((4, 4), 0.3)
    out = white_balance_mosaic(mosaic, [2.0, 1.0, 4.0])
    assert np.allclose(out[0::2, 0::2], 0.6)
    assert np.allclose(out[0::2, 1::2], 0.3) and np.allclose(out[1::2, 0::2], 0.3)
    assert np.allclose(out[1::2, 1::2], 1.0)

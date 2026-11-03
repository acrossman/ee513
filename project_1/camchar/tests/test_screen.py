"""camchar.screen: the screen target is drawn as claimed, and its square is found and placed precisely."""
import numpy as np
import pytest
from scipy.special import ndtr

from camchar import screen
from camchar.mtf import mtf50, slanted_edge_mtf
from camchar.screen import find_target_square, refine


def scene(rng, side=200, tilt=5.0, center=(620, 400), sigma=0.7):
    """Bright 'screen' with a tilted gray square, plus a dark distractor that is not square."""
    h, w = 800, 1200
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    th = np.radians(tilt)
    u = (xx - center[0]) * np.cos(th) + (yy - center[1]) * np.sin(th)
    v = -(xx - center[0]) * np.sin(th) + (yy - center[1]) * np.cos(th)
    inside = ndtr((side / 2 - np.abs(u)) / sigma) * ndtr((side / 2 - np.abs(v)) / sigma)
    img = 0.8 - 0.6 * inside
    img[650:760, 60:460] = 0.05                             # a long dark object (keyboard, bezel)
    img += 0.003 * rng.standard_normal(img.shape)
    c, s = np.cos(th), np.sin(th)
    corners = np.array([[center[0] + a * c - b * s, center[1] + a * s + b * c]
                        for a, b in ((-side / 2, -side / 2), (side / 2, -side / 2), (side / 2, side / 2), (-side / 2, side / 2))])
    return img, corners


def test_finds_the_square_not_the_distractor_and_places_corners(rng):
    img, truth = scene(rng)
    quad = find_target_square(img)
    assert quad is not None
    fine = refine(img, quad)
    # each refined corner within 0.5 px of a true corner, and the side length within 0.5%
    d = np.min(np.linalg.norm(fine[:, None, :] - truth[None, :, :], axis=2), axis=1)
    assert d.max() < 0.5
    side = np.mean([np.linalg.norm(fine[i] - fine[(i + 1) % 4]) for i in range(4)])
    assert side == pytest.approx(200, rel=0.005)


def test_returns_none_without_a_square(rng):
    img = 0.8 + 0.003 * rng.standard_normal((600, 800))
    img[400:500, 50:600] = 0.05                             # only a non-square dark object
    assert find_target_square(img) is None


def test_coverage_area_equals_the_square():
    cov = screen.coverage(400, 300, side=150, tilt_deg=5.0)
    assert cov.sum() == pytest.approx(150 ** 2, rel=2e-3)   # area integrates to side^2
    assert cov.min() >= 0 and cov.max() <= 1


def test_drawn_edge_is_an_ideal_pixel_area_edge():
    # Linear-light target levels and a box-filtered edge: MTF50 of a one-pixel box is 0.603 cy/px
    cov = screen.coverage(600, 600, side=420, tilt_deg=5.0)
    lin = screen.LIGHT + (screen.DARK - screen.LIGHT) * cov
    crop = lin[200:400, 90 - 40:90 + 40]                    # around the left edge (x = 300 - 210 at the center row)
    assert mtf50(*slanted_edge_mtf(crop)[:2]) == pytest.approx(0.603, rel=0.1)


def test_srgb_encoding_of_the_two_levels():
    assert float(screen.srgb_encode(screen.LIGHT)) * 255 == pytest.approx(203, abs=1)
    assert float(screen.srgb_encode(screen.DARK)) * 255 == pytest.approx(108, abs=1)
    assert screen.LIGHT / screen.DARK == pytest.approx(4.0)         # ISO 12233 4:1 in linear light

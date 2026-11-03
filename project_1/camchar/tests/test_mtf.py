"""Known-answer tests for camchar.mtf."""
import cv2
import numpy as np
import pytest

from camchar import mtf


def test_gaussian_mtf50_value():
    # sqrt(ln2 / (2 pi^2)) = 0.18739 cycles/px for sigma = 1
    assert mtf.gaussian_mtf50(1.0) == pytest.approx(0.187390625, rel=1e-6)
    assert mtf.gaussian_mtf50(2.0) == pytest.approx(mtf.gaussian_mtf50(1.0) / 2)


def test_mtf50_interpolates_a_linear_mtf():
    freq = np.linspace(0, 1, 11)
    assert mtf.mtf50(freq, 1 - freq) == pytest.approx(0.5)
    assert np.isnan(mtf.mtf50(freq, np.ones_like(freq)))       # never crosses 0.5


@pytest.mark.parametrize("sigma", [0.8, 1.0, 1.25, 1.5, 2.0])
def test_slanted_edge_recovers_gaussian_mtf50(sigma):
    edge = mtf.synthetic_edge(256, 128, angle_deg=5.0, sigma=sigma)
    measured = mtf.mtf50(*mtf.slanted_edge_mtf(edge)[:2])
    assert measured == pytest.approx(mtf.gaussian_mtf50(sigma), rel=0.05)


def test_slanted_edge_with_noise(rng):
    # Shot-noise-like noise at a modest signal level should not bias MTF50 beyond 5%
    edge = mtf.synthetic_edge(270, 128, sigma=1.0, lo=0.02, hi=0.12)
    electrons = rng.poisson(edge * 9000) / 9000
    noisy = electrons + 0.0005 * rng.standard_normal(edge.shape)
    assert mtf.mtf50(*mtf.slanted_edge_mtf(noisy)[:2]) == pytest.approx(mtf.gaussian_mtf50(1.0), rel=0.05)


def test_slanted_edge_is_independent_of_polarity_and_scale():
    edge = mtf.synthetic_edge(256, 128, sigma=1.2)
    ref = mtf.mtf50(*mtf.slanted_edge_mtf(edge)[:2])
    assert mtf.mtf50(*mtf.slanted_edge_mtf(np.fliplr(edge))[:2]) == pytest.approx(ref, rel=0.01)
    assert mtf.mtf50(*mtf.slanted_edge_mtf(0.1 + 0.3 * edge)[:2]) == pytest.approx(ref, rel=1e-6)


def test_edge_centroids_recover_the_tilt():
    edge = mtf.synthetic_edge(256, 128, angle_deg=5.0, sigma=1.0)
    rows = np.arange(256)
    slope = np.polyfit(rows, mtf.edge_centroids(edge), 1)[0]
    # x increases with y at tan(5 deg) for this edge
    assert slope == pytest.approx(np.tan(np.deg2rad(5.0)), rel=0.02)
    assert mtf.slanted_edge_mtf(edge)[4] == pytest.approx(np.tan(np.deg2rad(5.0)), rel=0.02)


def test_mtf_models():
    nu_c = 1 / (0.55 * 1.9)
    assert mtf.mtf_diffraction(0.0, 0.55, 1.9) == pytest.approx(1.0)
    assert mtf.mtf_diffraction(nu_c, 0.55, 1.9) == pytest.approx(0.0, abs=1e-12)
    assert mtf.mtf_diffraction(2 * nu_c, 0.55, 1.9) == pytest.approx(0.0, abs=1e-12)
    assert mtf.mtf_diffraction(nu_c / 2, 0.55, 1.9) == pytest.approx(0.3910, abs=1e-4)   # tabulated half-cutoff value
    assert mtf.mtf_pixel(0.0, 1.5) == pytest.approx(1.0)
    assert mtf.mtf_pixel(1 / 1.5, 1.5) == pytest.approx(0.0, abs=1e-12)                 # first zero at 1/p
    assert mtf.mtf_pixel(1 / (4 * 1.5), 1.5) == pytest.approx(np.sinc(0.25))             # sinc(1/4) at green Nyquist


def five_square_target(h=900, w=1200):
    """Bright page with a large center square and four corner squares, each tilted 5 degrees."""
    img = np.full((h, w), 0.8, dtype=np.float32)
    specs = {"center": (w / 2, h / 2, 300), "top_left": (200, 170, 150), "top_right": (w - 200, 170, 150),
             "bottom_left": (200, h - 170, 150), "bottom_right": (w - 200, h - 170, 150)}
    for cx, cy, size in specs.values():
        box = cv2.boxPoints(((cx, cy), (size, size), 5.0)).astype(np.int32)
        cv2.fillPoly(img, [box], 0.1)
    return img, specs


def test_find_squares_labels_and_locates_all_five():
    img, specs = five_square_target()
    squares = mtf.find_squares(img)
    assert set(squares) == set(specs)
    for name, (cx, cy, size) in specs.items():
        center = squares[name].mean(axis=0)
        assert np.hypot(center[0] - cx, center[1] - cy) < 3
        side = np.linalg.norm(squares[name][0] - squares[name][1])
        assert side == pytest.approx(size, rel=0.03)


def test_edge_crop_is_centered_on_the_requested_side():
    img, specs = five_square_target()
    quad = mtf.find_squares(img)["center"]
    for side in ("left", "right"):
        crop, (y0, x0, h, w) = mtf.edge_crop(img, quad, side, width=64)
        assert crop.shape == (h, 64)
        expected_x = quad[:, 0].min() if side == "left" else quad[:, 0].max()
        # the edge lies inside the crop, within the tilt of the square
        assert abs((x0 + w / 2) - expected_x) < 0.1 * 300
        assert np.nanmin(mtf.edge_centroids(crop)) > 0


def test_clean_rows_removes_a_mark_and_keeps_clean_crops_whole():
    edge = mtf.synthetic_edge(300, 128, sigma=1.0, lo=0.05, hi=0.8)    # bright on the right
    crop, first, dropped = mtf.clean_rows(edge)
    assert dropped == 0 and crop.shape == edge.shape

    marked = edge.copy()
    marked[200:230, 100:110] = 0.05                                     # "handwriting" on the light side
    crop, first, dropped = mtf.clean_rows(marked, pad=8)
    kept = np.arange(first, first + crop.shape[0])
    assert not np.intersect1d(kept, np.arange(200, 230)).size          # the mark is gone
    assert np.array_equal(kept, np.arange(0, 192))                      # longest clean block, minus the padding
    assert dropped == 300 - 192

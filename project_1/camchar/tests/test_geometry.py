"""Known-answer tests for camchar.geometry, on a synthetic camera with known intrinsics."""
import cv2
import numpy as np
import pytest

from camchar import geometry

TRUE_K = np.array([[3477.0, 0, 2238.0], [0, 3474.0, 1736.0], [0, 0, 1]])
TRUE_DIST = np.array([0.054, -0.27, 0.0, 0.0, 0.35])
SIZE = (4624, 3472)


def synthetic_views(rng, n=15, noise_px=0.0, square_mm=19.2):
    """Project a 9x6 board through the known camera from varied, tilted poses."""
    obj = geometry.board_points((9, 6), square_mm)
    views = []
    while len(views) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-150, 0), rng.uniform(-100, 0), rng.uniform(250, 600)])
        img, _ = cv2.projectPoints(obj, rvec, tvec, TRUE_K, TRUE_DIST)
        img = img.reshape(-1, 2)
        if (img.min(axis=0) > 0).all() and (img[:, 0].max() < SIZE[0]) and (img[:, 1].max() < SIZE[1]):
            views.append((img + noise_px * rng.standard_normal(img.shape)).astype(np.float32).reshape(-1, 1, 2))
    return views


def test_board_points_grid_and_scale():
    pts = geometry.board_points((9, 6), 19.2)
    assert pts.shape == (54, 3)
    assert np.allclose(pts[1] - pts[0], [19.2, 0, 0])
    assert np.allclose(pts[9] - pts[0], [0, 19.2, 0])
    assert np.all(pts[:, 2] == 0)


def test_calibrate_recovers_known_intrinsics(rng):
    cal = geometry.calibrate(synthetic_views(rng, noise_px=0.2), SIZE, square_mm=19.2)
    assert cal.rms == pytest.approx(0.2 * np.sqrt(2), rel=0.3)        # 2-D noise of 0.2 px per axis
    for name, truth in (("fx", 3477.0), ("fy", 3474.0), ("cx", 2238.0), ("cy", 1736.0)):
        value, std = cal.params[name]
        assert value == pytest.approx(truth, rel=0.01)
        assert abs(value - truth) < 4 * std                            # the reported std covers the truth
    assert cal.params["k1"][0] == pytest.approx(0.054, abs=0.02)
    assert cal.f_px[0] == pytest.approx((3477.0 + 3474.0) / 2, rel=0.01)
    assert len(cal.per_view_rms) == 15


def test_square_size_scales_distances_not_intrinsics(rng):
    views = synthetic_views(rng, noise_px=0.1)
    a = geometry.calibrate(views, SIZE, square_mm=19.2)
    b = geometry.calibrate(views, SIZE, square_mm=19.0)
    assert a.params["fx"][0] == pytest.approx(b.params["fx"][0], rel=1e-6)
    assert np.allclose(a.board_distances() / b.board_distances(), 19.2 / 19.0, rtol=1e-4)


def test_find_corners_on_a_rendered_board():
    # 10 x 7 squares of 60 px give 9 x 6 inner corners at known positions
    sq, x0, y0 = 60, 100, 80
    img = np.full((600, 800), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                img[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    corners = geometry.find_corners(img).reshape(-1, 2)
    truth = np.array([(x0 + (c + 1) * sq - 0.5, y0 + (r + 1) * sq - 0.5) for r in range(6) for c in range(9)])
    # OpenCV may order the corners from either end
    err = min(np.abs(corners - truth).max(), np.abs(corners[::-1] - truth).max())
    assert err < 0.25
    assert geometry.find_corners(np.full((600, 800), 128, np.uint8)) is None


def test_max_line_deviation():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    assert geometry.max_line_deviation(pts) == pytest.approx(0.0, abs=1e-12)
    bent = np.array([[0, 0], [1, 1], [2, 0]], float)        # TLS line is y = 1/3; farthest point 2/3 away
    assert geometry.max_line_deviation(bent) == pytest.approx(2 / 3)


def test_pixel_pitch_routes_and_field_of_view():
    assert geometry.pixel_pitch_from_focal(5.43, 3475.58) == pytest.approx(1.56233, rel=1e-4)
    # a 7.398 mm wide sensor at f = 5.43 mm has a width-based 35 mm equivalent of 5.43 * 36 / 7.398
    f35 = 5.43 * 36 / 7.3984
    assert geometry.pixel_pitch_from_35mm(5.43, f35, 4624) == pytest.approx(1.6, rel=1e-4)
    assert geometry.field_of_view(100.0, 200.0) == pytest.approx(90.0)
    assert geometry.focus_breathing(5.0, 505.0) == pytest.approx(0.01)

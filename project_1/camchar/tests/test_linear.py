"""Known-answer tests for camchar.linear, including an independent sRGB reference."""
import numpy as np
import pytest

from camchar import linear, raw


def test_srgb_known_values():
    assert linear.srgb_encode(0.0) == 0.0
    assert linear.srgb_encode(1.0) == pytest.approx(1.0)
    assert linear.srgb_encode(0.5) == pytest.approx(0.735357, abs=1e-6)      # half the light
    assert linear.srgb_decode(0.5) == pytest.approx(0.214041, abs=1e-6)      # an encoded 0.5
    assert linear.srgb_encode(0.002) == pytest.approx(12.92 * 0.002)          # linear segment near black


def test_srgb_matches_colour_science():
    colour = pytest.importorskip("colour")
    x = np.linspace(0, 1, 1001)
    assert np.allclose(linear.srgb_encode(x), colour.cctf_encoding(x, function="sRGB"), atol=1e-9)
    assert np.allclose(linear.srgb_decode(x), colour.cctf_decoding(x, function="sRGB"), atol=1e-9)


def test_srgb_round_trip_and_clipping():
    x = np.linspace(0, 1, 2001)
    assert np.allclose(linear.srgb_decode(linear.srgb_encode(x)), x, atol=1e-12)
    assert linear.srgb_encode(-0.5) == 0.0 and linear.srgb_encode(2.0) == pytest.approx(1.0)


def test_checkerboard_average_in_linear_light():
    board = (np.indices((256, 256)).sum(axis=0) % 2).astype(float)
    assert board.mean() == 0.5                                                 # averaging encoded values
    assert linear.srgb_encode(linear.srgb_decode(board).mean()) == pytest.approx(0.735357, abs=1e-6)
    assert linear.reduce_in_linear_light(board, 8) == pytest.approx(np.full((32, 32), 0.735357), abs=1e-6)


def test_box_reduce_exact_and_drops_partial_blocks():
    img = np.arange(16, dtype=float).reshape(4, 4)
    assert np.array_equal(linear.box_reduce(img, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert linear.box_reduce(np.ones((9, 10)), 4).shape == (2, 2)


def test_linear_light_never_darker_than_encoded_average(rng):
    enc = rng.uniform(0, 1, (64, 64))
    assert np.all(linear.reduce_in_linear_light(enc, 8) >= linear.box_reduce(enc, 8) - 1e-12)
    flat = np.full((16, 16), 0.3)
    assert np.allclose(linear.reduce_in_linear_light(flat, 8), linear.box_reduce(flat, 8))   # no mixing, no error


def test_center_patch():
    img = np.arange(100 * 120).reshape(100, 120)
    patch = raw.center_patch(img, 10)
    assert patch.shape == (10, 10)
    assert patch[0, 0] == img[45, 55]

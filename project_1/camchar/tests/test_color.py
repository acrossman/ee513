"""camchar.color: white balance, the color correction matrix and CIEDE2000 on cases with known answers."""
import numpy as np
import pytest

from camchar import color


@pytest.fixture
def ref():
    return color.reference()[0]


def test_reference_set_is_the_2014_formulation_in_order():
    xyz, names = color.reference()
    assert xyz.shape == (24, 3)
    assert names[0] == "dark skin" and names[18].startswith("white") and names[23].startswith("black")


def test_white_balance_makes_mid_gray_neutral_and_scales_white(ref, rng):
    rgb = np.abs(ref * [0.6, 1.0, 0.8] + 0.001 * rng.standard_normal(ref.shape))   # a color cast
    gains = color.white_balance_gains(rgb)
    assert gains[1] == 1.0
    balanced = color.balance(rgb, ref)
    gray = balanced[color.MID_GRAY]
    assert gray[0] == pytest.approx(gray[1]) and gray[2] == pytest.approx(gray[1])
    assert balanced[color.WHITE, 1] == pytest.approx(ref[color.WHITE, 1])


def test_ccm_recovers_a_known_matrix_exactly(ref):
    # A camera whose filters are a linear combination of the CIE functions (Luther condition):
    # its RGB is an exact linear map of XYZ, so the fitted matrix must undo it and dE must be ~0.
    M_true = np.array([[0.6, 0.3, 0.05], [0.25, 0.8, -0.05], [0.0, -0.1, 1.1]])
    camera = ref @ np.linalg.inv(M_true).T
    M = color.fit_ccm(camera, ref)
    assert M == pytest.approx(M_true, abs=1e-9)
    assert color.delta_e_2000(camera @ M.T, ref).max() < 1e-6


def test_delta_e_is_zero_for_identical_colors_and_positive_otherwise(ref):
    assert color.delta_e_2000(ref, ref).max() == pytest.approx(0, abs=1e-9)
    shifted = ref.copy(); shifted[0] *= [1.1, 1.0, 0.9]
    de = color.delta_e_2000(shifted, ref)
    assert de[0] > 1 and de[1:].max() == pytest.approx(0, abs=1e-9)


def test_lab_known_values():
    white = np.array([0.9505, 1.0, 1.089])
    assert color.xyz_to_lab(white, white) == pytest.approx([100, 0, 0], abs=1e-9)
    gray = color.xyz_to_lab(0.18 * white, white)                       # 18% gray: L* = 116 * 0.18^(1/3) - 16
    assert gray == pytest.approx([116 * 0.18 ** (1 / 3) - 16, 0, 0], abs=1e-9)


def test_numbering_check(ref):
    assert color.numbering_ok(ref)                    # reference order: gray row falls, dark skin redder than blue
    assert not color.numbering_ok(ref[::-1])          # an upside-down chart fails


def test_angle_between():
    assert color.angle_between([1, 0, 0], [0, 1, 0]) == pytest.approx(90)
    assert color.angle_between([1, 1, 0], [2, 2, 0]) == pytest.approx(0, abs=1e-5)   # arccos near 1 rounds to ~1e-6 deg

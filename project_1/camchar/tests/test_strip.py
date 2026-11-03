"""camchar.strip: the strip slanted-edge method behind the screen-target MTF50."""
import numpy as np
import pytest
from synthetic import blurred_edge, true_mtf50

from camchar.strip import strip_curve, strip_mtf


@pytest.mark.parametrize("sigma", [0.35, 0.45, 0.6, 1.0])
@pytest.mark.parametrize("tilt", [5, 10, 15, 85])        # the screen target is at 5; 85 = 5 from horizontal
def test_recovers_known_mtf50(sigma, tilt, rng):
    # Gold standard: Gaussian x square-pixel aperture along the edge normal, known analytically.
    # 5% is the pass threshold set for MTF50.
    img, p, q = blurred_edge(150, tilt, sigma, rng)
    assert strip_mtf(img, p, q, across=20) == pytest.approx(true_mtf50(sigma, tilt), rel=0.05)


def test_independent_of_polarity_and_contrast(rng):
    img, p, q = blurred_edge(120, 8, 0.6, rng, noise=0.0)
    a = strip_mtf(img, p, q, across=20)
    assert strip_mtf(1 - img, p, q, across=20) == pytest.approx(a, rel=1e-9)        # dark-to-light flipped
    assert strip_mtf(0.1 + 3 * img * 0.3, p, q, across=20) == pytest.approx(a, rel=1e-9)  # scaled and offset


def test_clipped_strip_is_rejected(rng):
    img, p, q = blurred_edge(120, 8, 0.6, rng, hi=1.2)     # light side above saturation (1.0)
    img = np.minimum(img, 1.0)
    assert np.isnan(strip_mtf(img, p, q, across=20))
    assert strip_curve(img, p, q, across=20) is None


def test_curve_is_one_at_zero_frequency_and_falls(rng):
    img, p, q = blurred_edge(120, 8, 0.6, rng, noise=0.0)
    f, m = strip_curve(img, p, q, across=20)
    assert m[0] == pytest.approx(1.0)
    assert m[np.argmin(np.abs(f - 0.5))] < m[np.argmin(np.abs(f - 0.1))]


@pytest.mark.xfail(strict=True, reason="Known limitation, recorded on purpose: at 20 degrees (tan = 0.364, near "
                   "4/11) the pixel centers line up along the edge, sub-pixel coverage of the ESF degrades, and a very "
                   "sharp edge (MTF50 ~0.41) reads 6-8% low. 45 degrees is fully degenerate. The screen target's edges "
                   "are at 5 degrees.")
def test_twenty_degrees_reads_low_on_sharp_edges(rng):
    vals = [strip_mtf(*blurred_edge(150, 20, 0.35, rng), across=20) for _ in range(3)]
    assert np.median(vals) == pytest.approx(true_mtf50(0.35, 20), rel=0.05)


@pytest.mark.xfail(strict=True, reason="Known limitation, recorded on purpose: on a point-sampled edge, a short "
                   "strip at a shallow tilt moves the edge by under 1.2 px, the 0.25-px ESF bins fill unevenly, and "
                   "MTF50 reads ~26% high (noise in thinly filled bins). Use edges that move several px along the strip.")
def test_short_shallow_strip_aliases_when_point_sampled(rng):
    # 0.6 * 36 * tan(2 deg) = 0.75 px of shift
    vals = [strip_mtf(*blurred_edge(36, 2, 1.0, rng, ss=1, noise=0.004), across=14.4) for _ in range(9)]
    assert np.median(vals) == pytest.approx(true_mtf50(1.0, box=False), rel=0.05)


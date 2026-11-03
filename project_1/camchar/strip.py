"""Slanted-edge MTF from a strip of pixels that follows an edge given by two points on it.

Pixels within ``across`` px of the line, between 20% and 80% of the way from p to q, are projected onto
their signed distance from the line and binned at 0.25 px into an edge spread function (ESF); the rest
(gradient to the line spread function, baseline outside +/-LSF_HALF, Hann window, |FFT|, MTF50) is as in
:func:`camchar.mtf.slanted_edge_mtf`. Nothing is resampled, so oblique edges add no blur. Frequencies are
cycles per pixel along the edge normal.

Validated on synthetic edges blurred by a Gaussian and integrated over each pixel's area
(``camchar/tests/test_strip.py``): within 5% of the analytic MTF50 for MTF50 0.18-0.41 at 5-15 degrees
of tilt. Known limits, recorded as expected test failures: 6-8% low on very sharp edges near 20 degrees
(pixel centers line up along the edge), useless at 45 degrees, and aliasing high on short, shallow,
noisy strips.
"""
import numpy as np

from .mtf import mtf50

LSF_HALF = 8.0       # px; the default 16 is wider than the smallest checkerboard squares


def strip_mtf(g, p, q, across):
    """MTF50 of the edge through corners p, q (see strip_curve). NaN if the strip has clipped pixels."""
    curve = strip_curve(g, p, q, across)
    return np.nan if curve is None else mtf50(*curve)


def strip_curve(g, p, q, across):
    """MTF curve (freq, mtf) of the edge through corners p, q from the pixels in a strip that follows it.

    Pixels between 20% and 80% of the way from p to q, and within `across` px of the line, are projected
    onto their signed distance from the line, binned at 0.25 px (ESF), then differentiated, windowed and
    Fourier transformed exactly as camchar.mtf.slanted_edge_mtf does. Frequencies: cycles per pixel
    along the edge normal. Returns None if the strip has clipped pixels.
    """
    u = (q - p) / np.linalg.norm(q - p); n = np.array([-u[1], u[0]]); L = np.linalg.norm(q - p)
    x0, x1 = int(min(p[0], q[0]) - across - 2), int(max(p[0], q[0]) + across + 3)
    y0, y1 = int(min(p[1], q[1]) - across - 2), int(max(p[1], q[1]) + across + 3)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rel = np.stack([xx - p[0], yy - p[1]], -1).astype(float)
    t, dist = rel @ u, rel @ n
    sel = (t > 0.2 * L) & (t < 0.8 * L) & (np.abs(dist) <= across)
    vals, dist = g[yy[sel], xx[sel]], dist[sel]
    if vals.max() >= 0.98:
        return None
    if vals[dist < 0].mean() > vals[dist > 0].mean():          # same polarity for every edge
        dist = -dist
    edges = np.arange(-across, across + 0.25, 0.25)
    idx = np.clip(np.digitize(dist, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=vals, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    keep = count > 0
    esf, ex = total[keep] / count[keep], (edges[:-1] + 0.125)[keep]
    lsf = np.gradient(esf, ex)
    inside = np.abs(ex) <= LSF_HALF
    lsf = (lsf - lsf[~inside].mean())[inside] * np.hanning(inside.sum())
    spec = np.abs(np.fft.rfft(lsf))
    return np.fft.rfftfreq(len(lsf), d=0.25), spec / spec[0]


def line_distance(a, b, point):
    """Perpendicular distance of point from the line through a and b."""
    u = (b - a) / np.linalg.norm(b - a)
    r = point - a
    return abs(r[0] * u[1] - r[1] * u[0])



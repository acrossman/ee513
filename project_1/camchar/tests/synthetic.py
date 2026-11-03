"""Synthetic edges with a known MTF50, shared by the tests in this folder."""
import numpy as np
from scipy.special import ndtr


def blurred_edge(L, tilt_deg, sigma, rng, noise=0.002, ss=8, lo=0.05, hi=0.30):
    """One straight edge of length L px through the image center, tilt_deg from vertical, blurred by a
    Gaussian of sigma px and integrated over each pixel's area (ss x ss samples), as a sensor does.

    Returns (image, p, q): the image and two points on the edge, L px apart.
    """
    n = int(2 * L) + 60
    th = np.radians(tilt_deg)
    u = np.array([np.sin(th), np.cos(th)]); nrm = np.array([-u[1], u[0]]); c = np.array([n / 2, n / 2])
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    acc = np.zeros((n, n))
    for dy in sub:
        for dx in sub:
            acc += ndtr(((xx + dx - c[0]) * nrm[0] + (yy + dy - c[1]) * nrm[1]) / sigma)
    img = lo + (hi - lo) * acc / ss ** 2 + noise * rng.standard_normal((n, n))
    return img, c - u * L / 2, c + u * L / 2


def true_mtf50(sigma, tilt_deg=0.0, box=True):
    """MTF50 along the normal of an edge tilted tilt_deg from vertical: Gaussian blur times the square
    pixel's aperture seen from that direction, exp(-2 pi^2 s^2 f^2) |sinc(f cos t) sinc(f sin t)|.
    box=False: point sampling, Gaussian only."""
    f = np.linspace(0, 1, 200001)
    t = np.radians(tilt_deg)
    m = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
    if box:
        m = m * np.abs(np.sinc(f * np.cos(t)) * np.sinc(f * np.sin(t)))
    return float(f[np.argmax(m < 0.5)])

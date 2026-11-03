"""Spatial resolution: slanted-edge MTF, MTF50, target-square location and MTF models.

Frequencies are in cycles per sample of whatever image is passed in; on a green plane
that is cycles per green px, with Nyquist at 0.5.
"""
import cv2
import numpy as np
from scipy.special import ndtr


def edge_centroids(roi, window=8.0, smooth=1.5):
    """Sub-pixel position of a near-vertical edge in every row.

    The coarse position is the largest horizontal gradient of a lightly smoothed copy;
    the fine position is the centroid of the absolute derivative within ``window`` px of it.

    Parameters
    ----------
    roi : ndarray, shape (H, W)
        Crop containing one near-vertical edge.
    window : float, optional
        Half-width of the centroid window, px. Default 8.
    smooth : float, optional
        Horizontal Gaussian sigma for the coarse search, px. Default 1.5.

    Returns
    -------
    ndarray of float, shape (H,)
        Edge position per row in column coordinates; NaN where a row has no gradient.
    """
    roi = np.asarray(roi, dtype=float)
    smoothed = cv2.GaussianBlur(roi, (0, 0), sigmaX=smooth, sigmaY=0.01)
    coarse = np.argmax(np.abs(np.diff(smoothed, axis=1)), axis=1) + 0.5
    deriv = np.abs(np.diff(roi, axis=1))
    x = np.arange(deriv.shape[1]) + 0.5
    weight = np.where(np.abs(x[None, :] - coarse[:, None]) <= window, deriv, 0.0)
    total = weight.sum(axis=1)
    good = total > 1e-9
    centers = np.full(roi.shape[0], np.nan)
    centers[good] = (weight[good] * x).sum(axis=1) / total[good]
    return centers


def slanted_edge_mtf(roi, bin_width=0.25, window=8.0, lsf_half_width=16.0):
    """MTF of a near-vertical slanted edge (ISO 12233-style).

    Fits a straight line to the per-row edge positions, projects every pixel onto its
    distance from that line, bins the projection into an oversampled edge spread function
    (ESF), differentiates it into the line spread function (LSF), windows the LSF with a
    fixed-width Hann window, and takes the normalized magnitude of its Fourier transform.

    Parameters
    ----------
    roi : ndarray, shape (H, W)
        Linear-light crop with one near-vertical edge, tilted a few degrees. Either polarity.
    bin_width : float, optional
        ESF bin width, px. 0.25 gives four-times oversampling. Default 0.25.
    window : float, optional
        Centroid window for :func:`edge_centroids`, px. Default 8.
    lsf_half_width : float, optional
        Half-width of the LSF window, px; the baseline is taken from outside it. Default 16.

    Returns
    -------
    freq : ndarray
        Spatial frequency, cycles per px of ``roi``.
    mtf : ndarray
        MTF, 1 at zero frequency.
    esf_x : ndarray
        ESF sample positions, px from the edge.
    esf : ndarray
        Edge spread function.
    slope : float
        Edge slope, columns per row (``degrees(arctan(slope))`` is the tilt).

    Notes
    -----
    Validated on Gaussian-blurred synthetic edges (``camchar/tests/test_mtf.py``):
    within 5% of the analytic MTF50 for sigma = 0.8 to 2.0 px.
    """
    roi = np.asarray(roi, dtype=float)
    rows = np.arange(roi.shape[0])
    centers = edge_centroids(roi, window=window)
    ok = np.isfinite(centers)
    slope, intercept = np.polyfit(rows[ok], centers[ok], 1)
    fitted = slope * rows + intercept

    cols = np.arange(roi.shape[1])
    distance = (cols[None, :] - fitted[:, None]).ravel()
    edges = np.arange(distance.min(), distance.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(distance, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=roi.ravel(), minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    keep = count > 0
    esf = total[keep] / count[keep]
    esf_x = (edges[:-1] + bin_width / 2)[keep]

    lsf = np.gradient(esf, esf_x)
    inside = np.abs(esf_x) <= lsf_half_width
    if (~inside).any():
        lsf = lsf - np.mean(lsf[~inside])
    lsf = lsf[inside] * np.hanning(inside.sum())

    spectrum = np.abs(np.fft.rfft(lsf))
    return np.fft.rfftfreq(len(lsf), d=bin_width), spectrum / spectrum[0], esf_x, esf, slope


def mtf50(freq, mtf):
    """Frequency where an MTF first falls below 0.5, by linear interpolation.

    Parameters
    ----------
    freq : ndarray
        Spatial frequency.
    mtf : ndarray
        MTF values at ``freq``.

    Returns
    -------
    float
        MTF50 in the units of ``freq``, or NaN if the MTF never crosses 0.5.
    """
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(freq[i - 1] + (mtf[i - 1] - 0.5) * (freq[i] - freq[i - 1]) / (mtf[i - 1] - mtf[i]))


def find_squares(green, n=5):
    """Locate the dark squares of a slanted-edge target and label them by position.

    Thresholds the image against a heavily blurred copy of itself, which adapts to lens
    shading, and keeps the ``n`` largest contours.

    Parameters
    ----------
    green : ndarray, shape (H, W)
        Normalized green plane (full scale 1).
    n : int, optional
        Number of squares to keep. Default 5.

    Returns
    -------
    dict
        ``"center"``, ``"top_left"``, ``"top_right"``, ``"bottom_left"``, ``"bottom_right"``
        to a (4, 2) array of corner points (x, y) from ``cv2.minAreaRect``.
    """
    small = cv2.GaussianBlur(green, (0, 0), 3)
    local = cv2.GaussianBlur(small, (0, 0), 150)
    mask = (small < 0.6 * local).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]

    H, W = green.shape
    squares = {}
    for c in contours:
        quad = cv2.boxPoints(cv2.minAreaRect(c))
        cx, cy = quad.mean(axis=0)
        if abs(cx - W / 2) < W / 6 and abs(cy - H / 2) < H / 6:
            name = "center"
        else:
            name = ("top" if cy < H / 2 else "bottom") + "_" + ("left" if cx < W / 2 else "right")
        squares[name] = quad
    return squares


def edge_crop(green, quad, side="left", width=128, height_frac=0.6):
    """Crop centered on one near-vertical side of a target square.

    Parameters
    ----------
    green : ndarray, shape (H, W)
        Green plane.
    quad : ndarray, shape (4, 2)
        Square corners from :func:`find_squares`.
    side : {"left", "right"}, optional
        Which near-vertical side. Default "left".
    width : int, optional
        Crop width, px. Default 128.
    height_frac : float, optional
        Crop height as a fraction of the side's length. Default 0.6.

    Returns
    -------
    crop : ndarray
        The crop (a view of ``green``).
    box : tuple of int
        ``(y0, x0, height, width)`` of the crop in ``green``.
    """
    order = np.argsort(quad[:, 0])
    pts = quad[order[:2]] if side == "left" else quad[order[2:]]
    top, bot = pts[np.argsort(pts[:, 1])]
    length = np.hypot(*(bot - top))
    mx, my = (top + bot) / 2
    h = int(height_frac * length)
    y0, x0 = int(round(my - h / 2)), int(round(mx - width / 2))
    return green[y0:y0 + h, x0:x0 + width], (y0, x0, h, width)


def clean_rows(crop, margin=10, pad=8):
    """Largest contiguous block of rows with no marks (e.g. handwriting) on the light side.

    A row is marked if, more than ``margin`` px from the edge on the light side, its 3-px
    running mean falls below halfway between the light and dark levels. Marked rows are
    padded by ``pad`` rows on each side. The edge position comes from a straight line fit
    to the per-row centroids with outlier rejection, so a mark with more contrast than the
    edge cannot be mistaken for it.

    Parameters
    ----------
    crop : ndarray, shape (H, W)
        Edge crop, e.g. from :func:`edge_crop`.
    margin : float, optional
        Distance from the edge excluded from the check, px. Default 10.
    pad : int, optional
        Rows rejected on each side of a marked row. Default 8.

    Returns
    -------
    clean : ndarray
        The clean sub-crop.
    first : int
        Index of its first row within ``crop``.
    dropped : int
        Number of rows removed.
    """
    raw_centers = edge_centroids(crop)
    rows = np.arange(crop.shape[0])
    ok = np.isfinite(raw_centers)
    for _ in range(3):   # fit, drop rows more than 3 px off the line, refit
        line = np.polyfit(rows[ok], raw_centers[ok], 1)
        ok &= np.abs(raw_centers - np.polyval(line, rows)) < 3
    centers = np.polyval(line, rows)
    x = np.arange(crop.shape[1])[None, :]
    light_on_left = crop[:, :20].mean() > crop[:, -20:].mean()
    light = (x < centers[:, None] - margin) if light_on_left else (x > centers[:, None] + margin)
    dark = (x > centers[:, None] + margin) if light_on_left else (x < centers[:, None] - margin)
    L, D = np.median(crop[light]), np.median(crop[dark])
    smooth = cv2.blur(crop, (3, 1))
    marked = np.array([(smooth[i][light[i]] < (L + D) / 2).any() for i in range(crop.shape[0])])
    marked = np.convolve(marked, np.ones(2 * pad + 1), mode="same") > 0
    best, start = (0, 0), None
    for i, m in enumerate(np.append(marked, True)):
        if not m and start is None:
            start = i
        elif m and start is not None:
            best = max(best, (i - start, start))
            start = None
    n, first = best
    return crop[first:first + n], first, crop.shape[0] - n


def gaussian_mtf50(sigma):
    """Analytic MTF50 of a Gaussian point spread function.

    Parameters
    ----------
    sigma : float
        PSF standard deviation, px.

    Returns
    -------
    float
        MTF50 in cycles/px, ``sqrt(ln 2 / (2 pi^2 sigma^2))`` (0.187 for sigma = 1).
    """
    return float(np.sqrt(np.log(2) / (2 * np.pi ** 2 * sigma ** 2)))


def synthetic_edge(h=200, w=200, angle_deg=5.0, sigma=1.0, lo=0.0, hi=1.0):
    """Closed-form slanted edge blurred by a Gaussian PSF and point-sampled at pixel centers.

    A straight step convolved with a 2-D Gaussian is the 1-D Gaussian CDF of the signed
    distance from the edge, so its MTF is exactly Gaussian. Noise-free.

    Parameters
    ----------
    h, w : int, optional
        Image size, px. Default 200 x 200.
    angle_deg : float, optional
        Edge tilt from vertical. Default 5.
    sigma : float, optional
        PSF standard deviation, px. Default 1.
    lo, hi : float, optional
        Dark and light levels. Default 0 and 1.

    Returns
    -------
    ndarray, shape (h, w)
        Edge image, bright on the right.
    """
    theta = np.deg2rad(angle_deg)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    y, x = np.mgrid[0:h, 0:w].astype(float)
    d = (x - cx) * np.cos(theta) - (y - cy) * np.sin(theta)
    return lo + (hi - lo) * ndtr(d / sigma)


def mtf_diffraction(nu, wavelength_um, f_number):
    """Diffraction MTF of an aberration-free circular aperture.

    Parameters
    ----------
    nu : float or ndarray
        Spatial frequency, cycles/um.
    wavelength_um : float
        Wavelength, um (0.55 for green).
    f_number : float
        Lens f-number.

    Returns
    -------
    float or ndarray
        ``(2/pi)(arccos s - s sqrt(1 - s^2))`` with ``s = nu / nu_c`` and cutoff
        ``nu_c = 1 / (wavelength * N)``; zero beyond the cutoff.
    """
    s = np.clip(np.asarray(nu, dtype=float) * wavelength_um * f_number, 0, 1)
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s ** 2))


def mtf_pixel(nu, pitch_um):
    """MTF of a square photodiode aperture of width equal to the pitch.

    Parameters
    ----------
    nu : float or ndarray
        Spatial frequency, cycles/um.
    pitch_um : float
        Pixel pitch, um.

    Returns
    -------
    float or ndarray
        ``|sinc(nu * p)|`` with ``sinc(x) = sin(pi x) / (pi x)``.
    """
    return np.abs(np.sinc(np.asarray(nu, dtype=float) * pitch_um))

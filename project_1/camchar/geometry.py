"""Geometry: checkerboard calibration, distortion, pixel pitch and field of view.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FULL_FRAME_MM = (36.0, 24.0)   # 35 mm format, width x height


def board_points(pattern=(9, 6), square_mm=1.0):
    """3-D coordinates of a checkerboard's inner corners on the Z = 0 plane.

    Parameters
    ----------
    pattern : tuple of int, optional
        Inner corners as (columns, rows). Default (9, 6).
    square_mm : float, optional
        Square side length. Default 1.0 (units of squares). Intrinsics in pixels and
        distortion do not depend on it; board poses (distances) scale with it.

    Returns
    -------
    ndarray of float32, shape (columns * rows, 3)
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return pts


def find_corners(gray, pattern=(9, 6), max_dim=None):
    """Find checkerboard inner corners and refine them to sub-pixel accuracy.

    Parameters
    ----------
    gray : ndarray of uint8, shape (H, W)
        Grayscale image, e.g. from :func:`camchar.raw.load_gray`.
    pattern : tuple of int, optional
        Inner corners as (columns, rows). Default (9, 6).
    max_dim : int, optional
        If given, detect on a copy downscaled so its longer side is ``max_dim`` pixels,
        then refine at full resolution (much faster on 16 MP frames). Default None
        detects at full resolution.

    Returns
    -------
    ndarray of float32, shape (N, 1, 2) or None
        Corner positions in full-resolution pixels, or None if the board was not found.
    """
    if max_dim is None:
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
    else:
        scale = min(1.0, max_dim / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(small, pattern, flags)
        if found:
            corners = (corners / scale).astype(np.float32)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


@dataclass
class Calibration:
    """Result of :func:`calibrate`. Every intrinsic carries a 1-sigma standard deviation.

    Attributes
    ----------
    rms : float
        RMS reprojection error, px.
    K : ndarray, shape (3, 3)
        Camera matrix.
    dist : ndarray
        Distortion coefficients in OpenCV order (k1, k2, p1, p2, k3, ...).
    params : dict
        ``name -> (value, std)`` for fx, fy, cx, cy (px) and k1, k2, p1, p2, k3.
    per_view_rms : ndarray
        RMS reprojection error of each view, px.
    rvecs, tvecs : list of ndarray
        Board pose of each view; ``tvecs`` are in the units of ``square_mm``.
    image_size : tuple of int
        (width, height) in pixels.
    """
    rms: float
    K: np.ndarray
    dist: np.ndarray
    params: dict
    per_view_rms: np.ndarray
    rvecs: list = field(repr=False)
    tvecs: list = field(repr=False)
    image_size: tuple = (0, 0)

    @property
    def f_px(self):
        """Mean of fx and fy, and its standard deviation (treating them as independent)."""
        (fx, sx), (fy, sy) = self.params["fx"], self.params["fy"]
        return (fx + fy) / 2, np.hypot(sx, sy) / 2

    def board_distances(self):
        """Distance from the camera to each board's origin corner, in the units of ``square_mm``."""
        return np.array([np.linalg.norm(t) for t in self.tvecs])


def calibrate(image_points, image_size, pattern=(9, 6), square_mm=1.0):
    """Calibrate intrinsics and distortion with ``cv2.calibrateCameraExtended``.

    Parameters
    ----------
    image_points : list of ndarray
        Corner positions per view, from :func:`find_corners`.
    image_size : tuple of int
        (width, height) in pixels.
    pattern : tuple of int, optional
        Inner corners as (columns, rows). Default (9, 6).
    square_mm : float, optional
        Square side length, for the board poses. Default 1.0.

    Returns
    -------
    Calibration

    Examples
    --------
    >>> pts = [find_corners(load_gray(f)) for f in files]
    >>> cal = calibrate([p for p in pts if p is not None], (4624, 3472), square_mm=19.2)
    >>> cal.params["fx"]
    (3476.99, 8.34)
    """
    obj = [board_points(pattern, square_mm)] * len(image_points)
    rms, K, dist, rvecs, tvecs, std_intr, _, per_view = cv2.calibrateCameraExtended(
        obj, image_points, tuple(image_size), None, None)
    std_intr, dist = std_intr.ravel(), dist.ravel()
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    values = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist[:5]]
    params = {n: (float(v), float(s)) for n, v, s in zip(names, values, std_intr[:9])}
    return Calibration(float(rms), K, dist, params, per_view.ravel(), list(rvecs), list(tvecs), tuple(image_size))


def max_line_deviation(pts):
    """Largest perpendicular distance of 2-D points from their total-least-squares line.

    Used on one row of checkerboard corners to quantify how straight it is, before and
    after undistortion.

    Parameters
    ----------
    pts : ndarray, shape (N, 2)
        Point coordinates, px.

    Returns
    -------
    float
        Maximum perpendicular deviation, px.
    """
    pts = np.asarray(pts, dtype=float)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    return float(np.abs(centered @ vt[1]).max())


def pixel_pitch_from_focal(f_mm, f_px):
    """Pixel pitch from a focal length in millimetres and the calibrated focal length in pixels.

    Parameters
    ----------
    f_mm : float
        Focal length, mm (e.g. from EXIF).
    f_px : float
        Focal length, px (e.g. ``Calibration.f_px[0]``).

    Returns
    -------
    float
        Pitch in micrometres, ``1000 * f_mm / f_px``.

    Notes
    -----
    A calibration at close range measures the focused image distance, which exceeds
    the infinity focal length (see :func:`focus_breathing`), so this underestimates the pitch.
    """
    return 1000.0 * f_mm / f_px


def pixel_pitch_from_35mm(f_mm, f35_mm, width_px):
    """Pixel pitch from a 35 mm-equivalent focal length taken across the sensor width.

    The equivalent focal length in the Camera FV5 device database is ``f * 36 mm / sensor
    width``, so the crop factor here is width-based. Combining it with the 43.3 mm diagonal
    instead mixes conventions and gives a pitch about 4% low.

    Parameters
    ----------
    f_mm : float
        Actual focal length, mm.
    f35_mm : float
        35 mm-equivalent focal length, mm, width-based.
    width_px : int
        Image width, px.

    Returns
    -------
    float
        Pitch in micrometres.

    Notes
    -----
    This returns whatever sensor size the source assumed; it is not independent of the
    manufacturer's figures.
    """
    crop = f35_mm / f_mm
    return 1000.0 * FULL_FRAME_MM[0] / crop / width_px


def field_of_view(f_px, size_px):
    """Full angle of view across a dimension of the image.

    Parameters
    ----------
    f_px : float
        Focal length, px.
    size_px : float
        Image extent in px: width, height, or diagonal (``np.hypot(w, h)``).

    Returns
    -------
    float
        Angle in degrees, ``2 * atan(size / (2 f))``.
    """
    return float(np.degrees(2 * np.arctan(size_px / (2 * f_px))))


def focus_breathing(f_mm, distance_mm):
    """Fractional increase of the image distance when focused at a finite distance (thin lens).

    Parameters
    ----------
    f_mm : float
        Focal length at infinity focus, mm.
    distance_mm : float or ndarray
        Object distance, mm.

    Returns
    -------
    float or ndarray
        ``f / (u - f)``: the image distance is ``f * (1 + this)``.
    """
    return f_mm / (np.asarray(distance_mm, dtype=float) - f_mm)

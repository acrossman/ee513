"""Reading raw captures: metadata, camera identity, green planes, and pairing captures.
"""
from pathlib import Path

import cv2
import numpy as np
import rawpy
import tifffile


def exif(path):
    """Return the EXIF fields of a DNG as a dict.

    Pixel DNGs put the EXIF fields directly in IFD0 instead of an Exif sub-IFD; both
    layouts are handled.

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    dict
        Tag name to value, e.g. ``"FocalLength"`` to a ``(numerator, denominator)`` tuple.
    """
    with tifffile.TiffFile(str(path)) as t:
        p = t.pages[0]
        if "ExifTag" in p.tags:
            return dict(p.tags["ExifTag"].value)
        return {tag.name: tag.value for tag in p.tags.values()}


def lens_id(path):
    """Identify the physical camera that made a DNG (model, focal length, 35 mm equivalent, image size).

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    tuple
        ``(lens model or phone model, focal length in mm rounded to 0.001,
        35 mm-equivalent focal length or None, (image width, image height))``.

    Examples
    --------
    >>> lens_id("IMG_20261019_190358.dng")
    ('Pixel 7a', 5.43, None, (4624, 3472))
    """
    with tifffile.TiffFile(str(path)) as t:
        p = t.pages[0]
        width, height = p.tags["ImageWidth"].value, p.tags["ImageLength"].value
    e = exif(path)
    fl = e["FocalLength"]
    return (e.get("LensModel", e.get("Model", "unknown")),
            round(fl[0] / fl[1], 3),
            e.get("FocalLengthIn35mmFilm"),
            (width, height))


def iso(path):
    """ISO sensitivity recorded in a DNG.

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    int
        ISO from ``ISOSpeedRatings`` or ``PhotographicSensitivity``.
    """
    e = exif(path)
    v = e.get("ISOSpeedRatings", e.get("PhotographicSensitivity"))
    return v[0] if isinstance(v, (tuple, list)) else v


def exposure_time(path):
    """Exposure time recorded in a DNG.

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    float
        Exposure time in seconds.
    """
    num, den = exif(path)["ExposureTime"]
    return num / den


def assert_one_camera(files, check_iso=False):
    """Assert that every capture in a series came from one camera (and optionally one ISO).

    Run this over every measurement series before fitting anything.

    Parameters
    ----------
    files : iterable of str or Path
        DNG files in the series.
    check_iso : bool, optional
        Also require a single ISO across the series. Default False.

    Returns
    -------
    tuple
        The single :func:`lens_id` value of the series.

    Raises
    ------
    AssertionError
        If the series is empty, or mixes cameras (or ISOs when ``check_iso`` is True).
    """
    files = list(files)
    assert files, "no files in this series"
    ids = {lens_id(f) for f in files}
    assert len(ids) == 1, f"more than one camera in this series: {ids}"
    if check_iso:
        isos = {iso(f) for f in files}
        assert len(isos) == 1, f"more than one ISO in this series: {isos}"
    return ids.pop()


def pairs_by_exposure(files, ndigits=4):
    """Group captures into pairs that share an exposure time.

    Parameters
    ----------
    files : iterable of str or Path
        DNG files, two per exposure level.
    ndigits : int, optional
        Exposure times are rounded to this many decimal places before grouping. Default 4.

    Returns
    -------
    list of tuple
        ``(frame 1, frame 2)`` path pairs sorted by exposure time, each pair in filename order.

    Raises
    ------
    AssertionError
        If any exposure time does not have exactly two captures.
    """
    groups = {}
    for f in sorted(files):
        groups.setdefault(round(exposure_time(f), ndigits), []).append(f)
    assert all(len(g) == 2 for g in groups.values()), {t: len(g) for t, g in groups.items()}
    return [tuple(groups[t]) for t in sorted(groups)]


def raw_levels(path):
    """Black level per channel and white level recorded in a DNG.

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    black : list of int
        Black level per CFA channel, in DN.
    white : int
        White (saturation) level, in DN. The bit depth is ``log2(white + 1)``.
    """
    with rawpy.imread(str(path)) as r:
        return list(r.black_level_per_channel), int(r.white_level)


def cfa_layout(path):
    """The 2x2 color filter layout as four letters, e.g. ``"RGGB"``.

    Parameters
    ----------
    path : str or Path
        DNG file.

    Returns
    -------
    str
        CFA letters in reading order of the 2x2 tile.
    """
    with rawpy.imread(str(path)) as r:
        return "".join(chr(r.color_desc[c]) for c in r.raw_pattern.flatten())


def green_plane(path, black=None, normalize=False):
    """Extract the G1 green plane (row 0, column 1 of each RGGB tile) from a DNG.

    No demosaicing, white balance or gamma: the result is linear in light.

    Parameters
    ----------
    path : str or Path
        DNG file with an RGGB mosaic.
    black : float or {"file"}, optional
        Black level to subtract, in DN. ``"file"`` uses the mean of the file's
        per-channel black levels. Default None subtracts nothing.
    normalize : bool, optional
        Divide by (white level - black level) so full scale is 1. Requires ``black``.
        Default False.

    Returns
    -------
    ndarray of float, shape (H/2, W/2)
        The green plane, in DN (or normalized to full scale).

    Raises
    ------
    AssertionError
        If the file is not RGGB, or ``normalize`` is set without ``black``.
    """
    with rawpy.imread(str(path)) as r:
        assert r.raw_pattern.tolist() == [[0, 1], [3, 2]] and r.color_desc == b"RGBG", "expected RGGB"
        g = r.raw_image_visible[0::2, 1::2].astype(np.float64)
        file_black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
    if black == "file":
        black = file_black
    if black is not None:
        g = g - black
    if normalize:
        assert black is not None, "normalize needs a black level"
        g = g / (white - black)
    return g


def center_patch(img, size=128):
    """Square patch at the center of an image.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Image, e.g. a green plane.
    size : int, optional
        Side length in pixels. Default 128, the patch used for the photon transfer curve.

    Returns
    -------
    ndarray, shape (size, size)
        A view of the central patch.
    """
    h, w = img.shape[:2]
    return img[h // 2 - size // 2:h // 2 + size // 2, w // 2 - size // 2:w // 2 + size // 2]


def load_gray(path, bps=8):
    """Demosaic a DNG and return a grayscale image in the sensor's native orientation.

    ``user_flip=0`` ignores the EXIF rotation, so portrait and landscape captures share
    one pixel coordinate system, which calibration needs. Used for corner detection only;
    measurements use :func:`green_plane`.

    Parameters
    ----------
    path : str or Path
        DNG file.
    bps : {8, 16}, optional
        Output bits per sample. Default 8.

    Returns
    -------
    ndarray of uint8 or uint16, shape (H, W)
        Grayscale image.
    """
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(user_flip=0, output_bps=bps)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def find_raw_files(folder, suffixes=(".dng",)):
    """Sorted raw files under a folder, searched recursively.

    Parameters
    ----------
    folder : str or Path
        Folder to search.
    suffixes : tuple of str, optional
        Lower-case file extensions to accept. Default ``(".dng",)``.

    Returns
    -------
    list of Path
    """
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

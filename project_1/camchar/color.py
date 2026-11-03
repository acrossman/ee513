"""Color accuracy from ColorChecker photos: linear camera RGB, white balance, a 3 x 3 color correction
matrix (CCM) and CIEDE2000 against published reference values.

Steps, per chart photo:
  1. :func:`quarter_rgb`: linear R, G, B at half the mosaic's width and height, without demosaicing
     (red site, mean of the two green sites, blue site), black subtracted, 1 = saturation.
  2. :func:`normalized_flat`: a white-sheet photo divided by its own center, per channel; dividing a
     chart photo by it removes lens shading (vignetting).
  3. :func:`find_patches`: OpenCV's ColorChecker detector (``cv2.mcc``) gives each patch's outline, in
     reference order, on the photo before shading correction; :func:`middle_of` shrinks each outline to
     about half the patch width and :func:`patch_means` averages the corrected image inside it.
  4. :func:`balance`: white-balance gains from mid-gray patch 21 (green gain 1), then a single scale so
     white's green equals the published white Y.
  5. :func:`fit_ccm`: the 3 x 3 matrix M minimizing sum |M rgb - xyz|^2 over the 24 patches.
  6. :func:`delta_e_2000`: CIEDE2000 per patch, each side converted to CIELAB against its own white patch.

Reference values: ``colour-science`` "ColorChecker24 - After November 2014" (:func:`reference`).
Tested in ``camchar/tests/test_color.py``.
"""
import colour
import cv2
import numpy as np
import rawpy

WHITE, MID_GRAY = 18, 20         # patch 19 (white) and patch 21 (neutral 6.5), as Python indexes
SHRINK = 0.78                    # the detector's outline is 0.59-0.64 of the patch width; 0.78 x 0.64 = 0.50

# Linear sRGB (D65) to XYZ: the "white balance only" comparison treats balanced camera RGB as sRGB
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])


def quarter_rgb(path, saturation=1023):
    """Linear (R, G, B) image at quarter resolution, 0 = black and 1 = saturation, and a map of the
    2 x 2 blocks in which any site reached saturation. The color order is read from the file."""
    with rawpy.imread(str(path)) as r:
        mosaic = r.raw_image_visible.astype(np.float64)
        pattern = r.raw_pattern.copy()
        letters = r.color_desc.decode()
        black = list(r.black_level_per_channel)
    sites = {"R": [], "G": [], "B": []}
    clipped = np.zeros((mosaic.shape[0] // 2, mosaic.shape[1] // 2), dtype=bool)
    for row in (0, 1):
        for col in (0, 1):
            index = pattern[row, col]
            site = mosaic[row::2, col::2]
            clipped |= site >= saturation
            sites[letters[index]].append((site - black[index]) / (saturation - black[index]))
    return np.dstack([sites["R"][0], (sites["G"][0] + sites["G"][1]) / 2, sites["B"][0]]), clipped


def normalized_flat(path, saturation=1023):
    """A white-sheet photo (quarter resolution) divided by the mean of its central 10% box, per channel."""
    flat, _ = quarter_rgb(path, saturation)
    h, w, _ = flat.shape
    center = flat[h // 2 - h // 20:h // 2 + h // 20, w // 2 - w // 20:w // 2 + w // 20]
    return flat / center.mean(axis=(0, 1))


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def find_patches(image):
    """Outline (4 corners, x and y) of each of the 24 patches, in reference order 1..24, from
    OpenCV's ColorChecker detector run on an 8-bit, sRGB-encoded copy of a linear image."""
    stretched = np.clip(image / np.percentile(image, 99.5, axis=(0, 1)), 0, 1)
    bgr8 = (srgb_encode(stretched)[..., ::-1] * 255).astype(np.uint8)
    detector = cv2.mcc.CCheckerDetector_create()
    detector.setColorChartType(cv2.mcc.MCC24)
    if not detector.process(bgr8, 1):
        raise ValueError("no ColorChecker found")
    return np.array(detector.getBestColorChecker().getColorCharts()).reshape(24, 4, 2)


def middle_of(outline):
    """The outline shrunk about its own center by SHRINK."""
    center = outline.mean(axis=0)
    return center + SHRINK * (outline - center)


def patch_means(image, clipped, samples):
    """Mean R, G, B inside each sampled region, and the number of clipped pixels inside them."""
    means, n_clipped = [], 0
    for sample in samples:
        inside = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(inside, [sample.round().astype(np.int32)], 1)
        means.append(image[inside == 1].mean(axis=0))
        n_clipped += int(clipped[inside == 1].sum())
    return np.array(means), n_clipped


def numbering_ok(rgb):
    """True if patches 19-24 (white to black) fall in brightness and patch 1 (dark skin) is redder than blue."""
    return bool(np.all(np.diff(rgb[18:24, 1]) < 0) and rgb[0, 0] > rgb[0, 2])


def reference():
    """Published XYZ (24 x 3) and names of the ColorChecker24 - After November 2014 patches."""
    data = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    return colour.xyY_to_XYZ(np.array(list(data.data.values()))), list(data.data.keys())


def white_balance_gains(rgb):
    """Gains that make mid-gray patch 21 neutral, with green's gain = 1."""
    gray = rgb[MID_GRAY]
    return gray[1] / gray


def balance(rgb, xyz_ref):
    """White balance, then scale so white's green equals the published white Y (removes exposure only)."""
    balanced = rgb * white_balance_gains(rgb)
    return balanced * xyz_ref[WHITE, 1] / balanced[WHITE, 1]


def fit_ccm(rgb, xyz):
    """3 x 3 M making M @ rgb closest to xyz over all patches, by least squares."""
    Mt, *_ = np.linalg.lstsq(rgb, xyz, rcond=None)
    return Mt.T


def xyz_to_lab(xyz, white):
    """CIE 1976 L*a*b* against a given white."""
    t = np.asarray(xyz, float) / np.asarray(white, float)
    delta = 6 / 29
    f = np.where(t > delta ** 3, np.cbrt(np.clip(t, 1e-12, None)), t / (3 * delta ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e_2000(xyz, xyz_ref):
    """CIEDE2000 per patch, each side in CIELAB against its own white patch (19)."""
    return colour.delta_E(xyz_to_lab(xyz, xyz[WHITE]), xyz_to_lab(xyz_ref, xyz_ref[WHITE]), method="CIE 2000")


def angle_between(u, v):
    """Angle between two vectors, degrees."""
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

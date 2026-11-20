import numpy as np


def quarter_reference(mosaic):
    red = mosaic[0::2, 0::2]
    green = (mosaic[0::2, 1::2] + mosaic[1::2, 0::2]) / 2
    blue = mosaic[1::2, 1::2]
    return np.dstack([red, green, blue])


def mosaic_rggb(rgb):
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=rgb.dtype)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out

import numpy as np

from isp.burst import merge_planes, split_planes
from isp.rawio import SATURATION, normalize_per_channel


def black_level(mosaic, black):
    return np.clip(normalize_per_channel(mosaic, black), 0, 1)


def normalized_flat(flat_mosaic):
    planes = split_planes(flat_mosaic)
    H, W = planes.shape[1:]
    center = planes[:, H // 2 - H // 20:H // 2 + H // 20, W // 2 - W // 20:W // 2 + W // 20]
    return planes / center.mean(axis=(1, 2))[:, None, None]


def lens_shading(mosaic, flat_planes):
    return merge_planes(split_planes(mosaic) / flat_planes)


def white_balance_mosaic(mosaic, gains):
    out = mosaic.astype(np.float64).copy()
    out[0::2, 0::2] *= gains[0]
    out[0::2, 1::2] *= gains[1]
    out[1::2, 0::2] *= gains[1]
    out[1::2, 1::2] *= gains[2]
    return np.clip(out, 0, 1)

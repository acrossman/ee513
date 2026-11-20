import numpy as np

from isp.noise import sigma_mad

SITES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def hot_sites(dark_average, k=6):
    hot = np.zeros(dark_average.shape, dtype=bool)
    for row, col in SITES:
        plane = dark_average[row::2, col::2]
        center = np.median(plane)
        sigma = sigma_mad((plane - center).ravel())
        hot[row::2, col::2] = plane > center + k * sigma
    return hot


def repeatable_hot_sites(dark_frames, k=6):
    first = hot_sites(dark_frames[:2].mean(axis=0), k)
    rest = hot_sites(dark_frames[2:].mean(axis=0), k)
    return first & rest


def stuck_sites(dark_frames):
    return (dark_frames == dark_frames[0]).all(axis=0)


def correct_defects(mosaic, defective):
    out = mosaic.astype(np.float64).copy()
    H, W = mosaic.shape
    for row, col in zip(*np.nonzero(defective)):
        values = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            r, c = row + dr, col + dc
            if 0 <= r < H and 0 <= c < W and not defective[r, c]:
                values.append(mosaic[r, c])
        if values:
            out[row, col] = np.mean(values)
    return out

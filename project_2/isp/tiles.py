import cv2
import numpy as np

from isp.burst import shift_of

TILE = 256
PAD = 32


def tile_grid(H, W, tile=TILE, pad=PAD):
    n_rows = (H - 2 * pad) // tile
    n_cols = (W - 2 * pad) // tile
    row0 = (H - n_rows * tile) // 2
    col0 = (W - n_cols * tile) // 2
    return [(row0 + i * tile, col0 + j * tile) for i in range(n_rows) for j in range(n_cols)]


def tile_shifts(greens, grid, global_shifts, tile=TILE, max_jump=3.0):
    window = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    shifts = np.zeros((len(greens), len(grid), 2))
    fell_back = np.zeros((len(greens), len(grid)), dtype=bool)
    for t, (r0, c0) in enumerate(grid):
        ref = np.ascontiguousarray(greens[0, r0:r0 + tile, c0:c0 + tile])
        for k in range(1, len(greens)):
            frame = np.ascontiguousarray(greens[k, r0:r0 + tile, c0:c0 + tile])
            dx, dy, _ = shift_of(ref, frame, window)
            if np.hypot(dx - global_shifts[k, 0], dy - global_shifts[k, 1]) > max_jump:
                dx, dy = global_shifts[k]
                fell_back[k, t] = True
            shifts[k, t] = dx, dy
    return shifts, fell_back


def align_tiles(frames, grid, shifts, tile=TILE, pad=PAD):
    aligned = np.full_like(frames, np.nan)
    for t, (r0, c0) in enumerate(grid):
        for k in range(len(frames)):
            dx, dy = shifts[k, t]
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            for p in range(frames.shape[1]):
                window = frames[k, p, r0 - pad:r0 + tile + pad, c0 - pad:c0 + tile + pad]
                moved = cv2.warpAffine(window, M, (tile + 2 * pad, tile + 2 * pad),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                aligned[k, p, r0:r0 + tile, c0:c0 + tile] = moved[pad:pad + tile, pad:pad + tile]
    return aligned

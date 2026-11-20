import numpy as np
from scipy.ndimage import gaussian_filter, shift

from isp.tiles import align_tiles, tile_grid, tile_shifts


def test_tile_grid_keeps_padding_inside_the_image():
    grid = tile_grid(1736, 2312)
    assert len(grid) == 6 * 8
    for r0, c0 in grid:
        assert r0 - 32 >= 0 and c0 - 32 >= 0
        assert r0 + 256 + 32 <= 1736 and c0 + 256 + 32 <= 2312


def test_each_tile_gets_its_own_shift():
    rng = np.random.default_rng(4)
    scene = gaussian_filter(rng.random((640, 1152)), 2).astype(np.float32)
    moved = scene.copy()
    moved[:, :576] = shift(scene, (0.0, 3.0), order=1, mode="reflect")[:, :576]
    moved[:, 576:] = shift(scene, (0.0, -2.0), order=1, mode="reflect")[:, 576:]
    greens = np.stack([scene, moved])
    grid = tile_grid(*scene.shape)
    shifts, fell_back = tile_shifts(greens, grid, np.zeros((2, 2)), max_jump=5)
    for t, (r0, c0) in enumerate(grid):
        expected = 3.0 if c0 + 256 <= 576 else -2.0 if c0 >= 576 else None
        if expected is not None:
            assert abs(shifts[1, t, 0] - expected) < 0.2
    assert not fell_back.any()


def test_align_tiles_undoes_the_shift_and_leaves_gaps_outside_tiles():
    rng = np.random.default_rng(5)
    scene = gaussian_filter(rng.random((384, 384)), 2).astype(np.float32)
    moved = shift(scene, (2.0, 1.0), order=1, mode="reflect").astype(np.float32)
    frames = np.stack([scene, moved])[:, None]
    grid = tile_grid(*scene.shape)
    shifts, _ = tile_shifts(frames[:, 0], grid, np.zeros((2, 2)))
    aligned = align_tiles(frames, grid, shifts)
    r0, c0 = grid[0]
    inside = np.s_[r0:r0 + 256, c0:c0 + 256]
    assert np.allclose(aligned[1, 0][inside], scene[inside], atol=0.02)
    assert np.isnan(aligned[1, 0, 0, 0])

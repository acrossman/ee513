import numpy as np
from scipy.ndimage import gaussian_filter, shift

from isp.burst import align


def test_align_recovers_known_shift():
    rng = np.random.default_rng(1)
    scene = gaussian_filter(rng.random((128, 128)), 2).astype(np.float32)
    moved = shift(scene, (3.0, -5.0), order=1, mode="reflect").astype(np.float32)
    _, shifts = align(np.stack([scene, moved]))
    dx, dy = shifts[1]
    assert abs(dx - (-5.0)) < 0.2
    assert abs(dy - 3.0) < 0.2


def test_identical_frames_need_no_shift():
    rng = np.random.default_rng(2)
    scene = gaussian_filter(rng.random((64, 64)), 2).astype(np.float32)
    aligned, shifts = align(np.stack([scene, scene]))
    assert np.allclose(shifts, 0, atol=1e-3)
    assert np.allclose(aligned[1], scene, atol=1e-5)


def test_split_and_merge_planes_round_trip():
    from isp.burst import merge_planes, split_planes
    mosaic = np.arange(48, dtype=np.float32).reshape(6, 8)
    planes = split_planes(mosaic)
    assert planes.shape == (4, 3, 4)
    assert planes[0, 0, 0] == mosaic[0, 0] and planes[3, 0, 0] == mosaic[1, 1]
    assert np.array_equal(merge_planes(planes), mosaic)


def test_shift_plane_undoes_known_whole_pixel_shift():
    from isp.burst import measure_shifts, shift_plane
    rng = np.random.default_rng(3)
    scene = gaussian_filter(rng.random((96, 96)), 2).astype(np.float32)
    moved = shift(scene, (2.0, -3.0), order=1, mode="reflect").astype(np.float32)
    dx, dy = measure_shifts(np.stack([scene, moved]))[1]
    back = shift_plane(moved, dx, dy)
    assert np.allclose(back[8:-8, 8:-8], scene[8:-8, 8:-8], atol=0.02)

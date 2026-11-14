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

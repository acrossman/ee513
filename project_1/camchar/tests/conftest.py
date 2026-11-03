"""Known-answer tests for camchar on synthetic data with analytic or constructed answers.

Run from project_1/:  python -m pytest camchar/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project_1/, so `import camchar` works


@pytest.fixture
def rng():
    return np.random.default_rng(513)


def simulate_flat_pair(electrons, K, sigma_r, rng, shape=(128, 128), black=64.0, prnu=None):
    """Two frames of a flat field from the Poisson-Gaussian sensor model, in DN.

    ``prnu`` is a fixed per-pixel gain map shared by both frames (fixed-pattern noise).
    No clipping or quantization, so the model holds exactly.
    """
    gain_map = np.ones(shape) if prnu is None else prnu
    frames = []
    for _ in range(2):
        e = rng.poisson(electrons * gain_map).astype(float)
        frames.append(black + K * e + K * sigma_r * rng.standard_normal(shape))
    return frames

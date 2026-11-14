import numpy as np

from isp.metrics import psnr
from isp.noise import fit_noise_model, frame_difference, sigma_mad, sigma_std


def test_both_estimates_match_gaussian_noise():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 200_000)
    b = rng.normal(0, 0.01, 200_000)
    d = frame_difference(a, b)
    assert abs(sigma_std(d) - 0.01) < 2e-4
    assert abs(sigma_mad(d) - 0.01) < 2e-4


def test_noise_model_recovers_read_and_shot_terms():
    rng = np.random.default_rng(4)
    clean = rng.uniform(0, 1, 400_000)
    read_var, k_shot = 1e-5, 4e-4
    sigma = np.sqrt(read_var + k_shot * clean)
    d = frame_difference(clean + rng.normal(0, 1, clean.size) * sigma,
                         clean + rng.normal(0, 1, clean.size) * sigma)
    fit_read, fit_k = fit_noise_model(clean, d)
    assert abs(fit_k - k_shot) / k_shot < 0.05
    assert abs(fit_read - read_var) < 5e-6


def test_psnr_of_known_error():
    ref = np.zeros((10, 10))
    assert np.isclose(psnr(ref + 0.1, ref), 20.0)

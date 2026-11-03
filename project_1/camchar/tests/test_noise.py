"""Known-answer tests for camchar.noise, on a simulated Poisson-Gaussian sensor."""
import numpy as np
import pytest
from conftest import simulate_flat_pair

from camchar import noise

TRUE_K = 0.4266       # DN/e-, simulated sensor
TRUE_READ = 2.8       # e-
FULL_WELL = 9000.0    # e-


def test_ptc_point_exact_mean_and_cancels_fixed_pattern(rng):
    pattern = 5.0 * rng.standard_normal((128, 128))          # identical in both frames
    noise_a, noise_b = rng.standard_normal((2, 128, 128))
    a = 100.0 + pattern + 2.0 * noise_a
    b = 100.0 + pattern + 2.0 * noise_b
    mean, var = noise.ptc_point(a, b, black=64.0)
    assert mean == pytest.approx((a.mean() + b.mean()) / 2 - 64.0)
    assert var == pytest.approx(4.0, rel=0.05)                # random variance 2^2; the pattern's 25 is gone


def test_fit_ptc_exact_on_a_noiseless_line():
    m = np.linspace(10, 900, 12)
    v = 0.07 * m + 0.35
    fit = noise.fit_ptc(m, v, full_scale=959)
    assert fit.K == pytest.approx(0.07)
    assert fit.intercept == pytest.approx(0.35)
    assert fit.sigma_r == pytest.approx(np.sqrt(0.35) / 0.07)
    assert fit.used.sum() == np.sum(m < 0.7 * 959)


LEVELS = np.logspace(np.log10(5), np.log10(0.65 * FULL_WELL), 22)   # 22 exposure levels


def simulated_fit(seed, **kwargs):
    rng = np.random.default_rng(seed)
    pts = np.array([noise.ptc_point(*simulate_flat_pair(e, TRUE_K, TRUE_READ, rng), black=64.0) for e in LEVELS])
    return noise.fit_ptc(pts[:, 0], pts[:, 1], full_scale=TRUE_K * FULL_WELL, **kwargs)


def test_fit_ptc_recovers_a_simulated_sensor():
    # Over 40 seeds the fitted gain scatters by 0.67% with no bias (+0.03%), so 2.5% is a ~3.7-sigma bound
    fit = simulated_fit(513)
    assert fit.K == pytest.approx(TRUE_K, rel=0.025)
    assert fit.sigma_r == pytest.approx(TRUE_READ, rel=0.35)
    assert abs(fit.sigma_r - TRUE_READ) < 3 * fit.s_sigma_r


def test_fit_ptc_is_unbiased_over_many_sensors():
    fits = [simulated_fit(seed) for seed in range(40)]
    assert np.mean([f.K for f in fits]) == pytest.approx(TRUE_K, rel=0.003)
    assert np.mean([f.sigma_r for f in fits]) == pytest.approx(TRUE_READ, rel=0.08)


@pytest.mark.xfail(strict=True, reason="Known issue: the unweighted fit's covariance assumes equal scatter at every "
                   "level, but a variance estimate's scatter grows with the variance, so sK is about 3.6x too small. "
                   "A fit weighted by 1/(var*sqrt(2/(N-1))) gives correct coverage.")
def test_fit_ptc_gain_uncertainty_has_correct_coverage():
    z = np.array([(f.K - TRUE_K) / f.sK for f in (simulated_fit(seed) for seed in range(60))])
    assert 0.7 < z.std() < 1.4


def test_single_frame_variance_overestimates_gain_with_fixed_pattern(rng):
    # 1.2% PRNU: pair differences stay right, single frames do not
    prnu = 1 + 0.012 * rng.standard_normal((128, 128))
    levels = np.linspace(200, 0.65 * FULL_WELL, 12)
    pair, single = [], []
    for e in levels:
        a, b = simulate_flat_pair(e, TRUE_K, TRUE_READ, rng, prnu=prnu)
        pair.append(noise.ptc_point(a, b, 64.0))
        single.append((pair[-1][0], a.var()))
    pair, single = np.array(pair), np.array(single)
    k_pair = noise.fit_ptc(pair[:, 0], pair[:, 1], TRUE_K * FULL_WELL).K
    k_single = noise.fit_ptc(single[:, 0], single[:, 1], TRUE_K * FULL_WELL).K
    assert k_pair == pytest.approx(TRUE_K, rel=0.02)
    assert k_single > 1.3 * TRUE_K


def test_full_well_and_dynamic_range():
    fw, s_fw = noise.full_well(959.0, 0.0691, 0.0002)
    assert fw == pytest.approx(959.0 / 0.0691)
    assert s_fw == pytest.approx(fw * 0.0002 / 0.0691)
    dr, s_dr = noise.dynamic_range_stops(1024.0, 1.0)
    assert dr == pytest.approx(10.0)                          # 2^10
    assert s_dr == 0.0
    dr1, _ = noise.dynamic_range_stops(1024.0, 8.0, convention="snr1")
    assert dr1 == pytest.approx(np.log2(1024.0 / noise.snr_one_signal(8.0)))
    with pytest.raises(ValueError):
        noise.dynamic_range_stops(1024.0, 8.0, convention="nope")


def test_snr_model_limits_and_knee():
    sr = 8.75
    assert noise.snr_model(sr ** 2, sr) == pytest.approx(sr / np.sqrt(2))      # at the knee
    assert noise.snr_model(1e8, sr) == pytest.approx(1e4, rel=1e-6)            # photon limit sqrt(mu)
    assert noise.snr_model(1e-3, sr) == pytest.approx(1e-3 / sr, rel=1e-3)     # read-noise limit mu / sigma_r
    assert noise.snr_model(noise.snr_one_signal(sr), sr) == pytest.approx(1.0)


def test_noise_profile_round_trip():
    K, sr, span = 0.0691, 8.75, 959.0
    S, O = K / span, (K * sr) ** 2 / span ** 2               # what a vendor tag would hold for this sensor
    K2, sr2, read_var = noise.noise_profile_to_gain(S, O, span)
    assert (K2, sr2, read_var) == pytest.approx((K, sr, (K * sr) ** 2))


def test_dark_current_recovers_a_constructed_slope(rng):
    t = np.repeat([0.025, 0.1, 1, 2, 16], 5)
    level = 64.011 + 0.0517 * t + 0.001 * rng.standard_normal(t.size)
    d = noise.dark_current(t, level, K=0.0691, sK=0.0002)
    assert d["slope"] == pytest.approx(0.0517, rel=0.01)
    assert d["offset"] == pytest.approx(64.011, abs=0.002)
    assert d["e_per_s"] == pytest.approx(0.0517 / 0.0691, rel=0.01)
    assert d["s_e_per_s"] > 0


def test_weighted_fit_is_unbiased_with_correct_coverage():
    fits = [simulated_fit(seed, weighted=True, n_pixels=128 * 128) for seed in range(60)]
    assert np.mean([f.K for f in fits]) == pytest.approx(TRUE_K, rel=0.003)
    z_K = np.array([(f.K - TRUE_K) / f.sK for f in fits])
    z_c = np.array([(f.intercept - (TRUE_K * TRUE_READ) ** 2) / f.s_intercept for f in fits])
    assert 0.7 < z_K.std() < 1.4 and 0.7 < z_c.std() < 1.4
    assert 0.5 < np.mean([f.chi2_dof for f in fits]) < 1.5          # pure noise: chi2/dof near 1
    assert all(f.method == "weighted" for f in fits)


def test_weighted_fit_inflates_for_extra_scatter():
    m = np.linspace(20, 600, 10)
    v = 0.07 * m + 0.3
    v_noisy = v * (1 + 0.05 * np.array([1, -1] * 5))                 # 5% scatter, far above sqrt(2/N) = 1.1%
    plain = noise.fit_ptc(m, v_noisy, 959, weighted=True, n_pixels=16384, inflate=False)
    infl = noise.fit_ptc(m, v_noisy, 959, weighted=True, n_pixels=16384)
    assert plain.chi2_dof > 1
    assert infl.sK == pytest.approx(plain.sK * np.sqrt(plain.chi2_dof))
    assert infl.K == plain.K
    with pytest.raises(ValueError):
        noise.fit_ptc(m, v, 959, weighted=True)

"""OP3, noise and dynamic range at ISO 45, on the G1 green plane.

Photon transfer: for each flat-field pair, the mean of a central 128 x 128 patch above the measured black
level and half the variance of the pair's difference (which cancels fixed pattern). A line fit below 70%
of saturation gives the conversion gain K (slope) and K^2 sigma_d^2 (intercept); the weighted fit, with
its covariance inflated by chi^2/dof, is reported. The black level is the all-pixel mean of the shortest
dark frames; dark current is the slope of the G1-plane mean of 25 lens-capped frames over exposure time.
Also: the single-frame gain (why pairs are needed), the vendor's DNG NoiseProfile, dark-frame SNR points,
EMVA 1288 quantities, and K repeated on 25 patches with frame-to-frame illumination changes removed.
Writes outputs/op3_noise.json and outputs/op3_points.npz.   Usage: python -m pipeline.op3_noise
"""
import json

import numpy as np
import rawpy

from camchar import noise, raw
from .paths import images, out

PATCH = 128
FIT = "weighted"


def frame_mean(path):
    with rawpy.imread(str(path)) as r:
        return float(r.raw_image_visible.astype(np.float64).mean())


def patch_grid_gain(planes, black, full_scale):
    """K on a 5 x 5 grid of patches, before and after removing row bands and smooth gradients
    (row means + x, x^2, x*y) from each pair's difference image; residual variance scaled by N/(N - p)."""
    y, x = np.mgrid[0:PATCH, 0:PATCH].astype(float)
    xs, yc = (x - x.mean()) / PATCH, (y - y.mean()) / PATCH
    rows = np.zeros((PATCH * PATCH, PATCH)); rows[np.arange(PATCH * PATCH), y.ravel().astype(int)] = 1
    A = np.column_stack([rows, xs.ravel(), xs.ravel() ** 2, (xs * yc).ravel()])
    N, P = A.shape
    H, W = planes[0][0].shape
    K_raw, K_clean = [], []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y0, x0 = H // 2 - PATCH // 2 + dy * PATCH, W // 2 - PATCH // 2 + dx * PATCH
            r_pts, c_pts = [], []
            for a, b in planes:
                pa, pb = a[y0:y0 + PATCH, x0:x0 + PATCH], b[y0:y0 + PATCH, x0:x0 + PATCH]
                m = (pa.mean() + pb.mean()) / 2 - black
                d = (pa - pb).ravel()
                coef, *_ = np.linalg.lstsq(A, d, rcond=None)
                r_pts.append((m, np.var(d) / 2)); c_pts.append((m, np.mean((d - A @ coef) ** 2) * N / (N - P) / 2))
            for pts, dest in ((np.array(r_pts), K_raw), (np.array(c_pts), K_clean)):
                dest.append(noise.fit_ptc(pts[:, 0], pts[:, 1], full_scale, weighted=True, n_pixels=PATCH * PATCH).K)
    K_raw, K_clean = np.array(K_raw), np.array(K_clean)
    return dict(as_measured=dict(mean=float(K_raw.mean()), sd=float(K_raw.std(ddof=1)), min=float(K_raw.min()), max=float(K_raw.max())),
                illumination_removed=dict(mean=float(K_clean.mean()), sd=float(K_clean.std(ddof=1)), center=float(K_clean[12])))


def main():
    flats, darks = images("flat_pair"), images("dark_frame")
    pairs = raw.pairs_by_exposure(flats)
    print("flat:", raw.assert_one_camera(flats, check_iso=True), f"{len(pairs)} pairs")
    print("dark:", raw.assert_one_camera(darks, check_iso=True), f"{len(darks)} frames")

    dark_t = np.array([raw.exposure_time(f) for f in darks])
    dark_level = np.array([frame_mean(f) for f in darks])
    shortest = dark_t == dark_t.min()
    black, s_black = float(dark_level[shortest].mean()), float(dark_level[shortest].std(ddof=1))
    white = raw.raw_levels(flats[0])[1]
    full_scale = white - black

    planes = [(raw.green_plane(a).astype(float), raw.green_plane(b).astype(float)) for a, b in pairs]
    patches = [(raw.center_patch(a, PATCH), raw.center_patch(b, PATCH)) for a, b in planes]
    pts = np.array([noise.ptc_point(a, b, black) for a, b in patches])
    single = np.array([pa.var() for pa, _ in patches])

    dpatch = [raw.center_patch(raw.green_plane(f), PATCH) for f in np.array(darks)[shortest]]
    dark_read_var = float(np.mean([np.var(a.astype(float) - b) / 2 for a, b in zip(dpatch[:-1], dpatch[1:])]))

    S, O = np.array(raw.exif(flats[0])["NoiseProfile"]).reshape(-1, 2)[1]       # green plane
    K_v, sr_v, rv_v = noise.noise_profile_to_gain(S, O, white - raw.raw_levels(flats[0])[0][1])

    dark_level_g1 = np.array([raw.green_plane(f).astype(np.float64).mean() for f in darks])

    result = dict(black_dn=[black, s_black], white_dn=white, n_pairs=len(pairs), dark_read_var_dn2=dark_read_var,
                  vendor=dict(K=K_v, sigma_r=sr_v, read_var_dn2=rv_v))
    for method, kwargs in (("unweighted", {}), ("weighted", {"weighted": True, "n_pixels": PATCH * PATCH})):
        fit = noise.fit_ptc(pts[:, 0], pts[:, 1], full_scale, **kwargs)
        fw, s_fw = noise.full_well(full_scale, fit.K, fit.sK)
        dr, s_dr = noise.dynamic_range_stops(fw, fit.sigma_r, s_full_well=s_fw, s_sigma_r=fit.s_sigma_r)
        dc = noise.dark_current(dark_t, dark_level_g1, fit.K, fit.sK)
        # EMVA 1288: absolute sensitivity threshold (SNR = 1) and dynamic range over it
        mu_min = noise.snr_one_signal(fit.sigma_r)
        s_mu_min = fit.sigma_r / np.sqrt(fit.sigma_r ** 2 + 0.25) * fit.s_sigma_r
        ratio = fw / mu_min
        s_ratio = ratio * np.hypot(s_fw / fw, s_mu_min / mu_min)
        snr_meas = pts[fit.used, 0] / np.sqrt(pts[fit.used, 1])
        snr_pred = noise.snr_model(pts[fit.used, 0] / fit.K, fit.sigma_r)
        result[method] = dict(
            K=[fit.K, fit.sK], inv_K=[1 / fit.K, fit.sK / fit.K ** 2], intercept_dn2=[fit.intercept, fit.s_intercept],
            sigma_d=[fit.sigma_r, fit.s_sigma_r], correlation=fit.correlation, chi2_dof=fit.chi2_dof, n_fit=int(fit.used.sum()),
            sat_capacity_e=[fw, s_fw], mu_e_min=[mu_min, float(s_mu_min)],
            dr_emva=dict(ratio=[ratio, s_ratio], bit=[np.log2(ratio), s_ratio / ratio / np.log(2)],
                         dB=[20 * np.log10(ratio), 20 / np.log(10) * s_ratio / ratio]),
            dr_read_noise_floor_bit=[dr, s_dr],
            snr_max=[np.sqrt(fw), 0.5 * s_fw / np.sqrt(fw)],
            knee_e=[fit.sigma_r ** 2, 2 * fit.sigma_r * fit.s_sigma_r],
            dark_current_e_per_s=[dc["e_per_s"], dc["s_e_per_s"]], dark_current_dn_per_s=[dc["slope"], dc["s_slope"]],
            snr_median_rel_error=float(np.median(np.abs(snr_meas - snr_pred) / snr_pred)),
            vendor_K_gap_pct=100 * (K_v / fit.K - 1), vendor_K_gap_sigma=(K_v - fit.K) / fit.sK)
    result["single_frame_K"] = noise.fit_ptc(pts[:, 0], single, full_scale).K
    result["sigma_y_dark_dn"] = float(np.sqrt(dark_read_var))
    result["patch_grid_K"] = patch_grid_gain(planes, black, full_scale)

    # Dark-frame SNR points: adjacent lens-capped frames, whole G1 plane
    dark_snr = []
    for t in np.unique(dark_t):
        frames = [raw.green_plane(f).astype(float) for f, tt in zip(darks, dark_t) if tt == t]
        for a, b in zip(frames[:-1], frames[1:]):
            m, v = noise.ptc_point(a, b, black)
            dark_snr.append((t, m, v))
    out("op3_noise.json").write_text(json.dumps(result, indent=2, default=float))
    np.savez(out("op3_points.npz"), pts=pts, used=noise.fit_ptc(pts[:, 0], pts[:, 1], full_scale).used,
             single=single, dark_snr=np.array(dark_snr), full_scale=full_scale)
    w = result[FIT]
    print(f"K {w['K'][0]:.4f} +/- {w['K'][1]:.4f} DN/e-, sigma_d {w['sigma_d'][0]:.2f} +/- {w['sigma_d'][1]:.2f} e-, "
          f"DR {w['dr_emva']['bit'][0]:.2f} +/- {w['dr_emva']['bit'][1]:.2f} bit, dark {w['dark_current_e_per_s'][0]:.3f} e-/s")
    g = result["patch_grid_K"]["illumination_removed"]
    print(f"single-frame K {result['single_frame_K']:.4f}; 25 patches, illumination removed: {g['mean']:.4f} +/- {g['sd']:.4f}")


if __name__ == "__main__":
    main()

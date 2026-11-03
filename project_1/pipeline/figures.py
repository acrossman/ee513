"""The datasheet's figures, from the measurement outputs:
  geometry_panel.png   original | undistorted with shift arrows | heat map of the shift (OP1)
  mtf_screen_vs_print.png   mean MTF of the screen and printed targets (OP2)
  ptc.png, snr.png     photon transfer curve; SNR with dark-frame pairs (OP3)
  delta_e_per_patch.png    per-patch CIEDE2000 before and after the matrix, incandescent chart 1 (OP4)
Usage: python -m pipeline.figures
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from matplotlib.colors import LogNorm

from .paths import images, out

W, H = 4624, 3472


def save(fig, name):
    fig.savefig(out("figures/" + name), dpi=150)
    plt.close(fig)


def geometry_panel(g):
    K, D = np.array(g["K"]), np.array(g["dist"])
    k1, k2 = g["params"]["k1"][0], g["params"]["k2"][0]
    R = np.hypot(W / 2, H / 2); f = (K[0, 0] + K[1, 1]) / 2
    und = lambda p: cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), K, D, P=K).reshape(p.shape)
    field = lambda p: np.hypot(p[..., 0] - K[0, 2], p[..., 1] - K[1, 2]) / R
    f12, fcal = np.sqrt(abs(k1 / k2)) * f / R, g["max_corner_field"]

    def ring(ax, fld, **kw):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(K[0, 2] + fld * R * np.cos(t), K[1, 2] + fld * R * np.sin(t), **kw)

    with rawpy.imread(str(images("undistortion_example")[0])) as r:
        rgb = r.postprocess(use_camera_wb=True, output_bps=8, user_flip=0)
    small = lambda im: cv2.resize(im, (1400, round(1400 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    orig, undi = small(rgb), small(cv2.undistort(rgb, K, D))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.07]})
    ax[0].imshow(orig, extent=(0, W, H, 0)); ax[0].set_title("Original", fontsize=10)
    xs, ys = np.meshgrid(np.linspace(200, W - 200, 15), np.linspace(200, H - 200, 11)); G = np.stack([xs, ys], -1)
    d = (und(G) - G) * 15; inside = field(G) <= fcal
    ax[1].imshow(undi, extent=(0, W, H, 0))
    for m, col, al in ((inside, "red", 1), (~inside, "orange", 0.6)):
        ax[1].quiver(G[m][:, 0], G[m][:, 1], d[m][:, 0], d[m][:, 1], angles="xy", scale_units="xy", scale=1,
                     color=col, alpha=al, width=0.004, headwidth=4)
    ax[1].set_title("Undistorted with Shift Directions", fontsize=10)
    xs, ys = np.meshgrid(np.linspace(0, W, 463), np.linspace(0, H, 348)); G = np.stack([xs, ys], -1)
    v = und(G) - G; rad = G - [K[0, 2], K[1, 2]]
    inward = -(v * rad).sum(-1) / np.maximum(np.linalg.norm(rad, axis=-1), 1)
    im = ax[2].imshow(np.clip(inward, 0.25, None), extent=(0, W, H, 0), cmap="viridis", norm=LogNorm(0.25, 80))
    cb = fig.colorbar(im, ax=ax[2], fraction=0.04, pad=0.02, ticks=[0.25, 1, 4, 16, 64])
    cb.ax.set_yticklabels(["0.25", "1", "4", "16", "64"], fontsize=7); cb.set_label("inward shift (px)", fontsize=8)
    ax[2].set_title("Shift magnitude (all inward)", fontsize=10)
    for a in ax[1:]:
        ring(a, f12, color="lime", lw=1.5); ring(a, fcal, color="w", lw=1.2, ls="--")
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    fig.tight_layout(); save(fig, "geometry_panel.png")


def mtf_curves():
    c = np.load(out("op2_curves.npz"))
    grid = c["grid"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, label, col, ls in (("screen_center", "screen, center, 1.10 m", "C0", "-"),
                                ("screen_corners", "screen, corners (in focus), 0.76-0.94 m", "C0", "--"),
                                ("print_center", "printed target, center, 179 mm", "C3", "-"),
                                ("print_corners", "printed target, corners, 179 mm", "C3", "--")):
        cur = c[key]; m, s = cur.mean(0), cur.std(0, ddof=1)
        ax.plot(grid, m, color=col, ls=ls, lw=1.6, label=f"{label} ({len(cur)} edges)")
        ax.fill_between(grid, m - s, m + s, color=col, alpha=0.12)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8); ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, 0.75); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_xlabel("cycles per green pixel"); ax.set_ylabel("MTF"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "mtf_screen_vs_print.png")


def noise_figures(n):
    p = np.load(out("op3_points.npz"))
    pts, used, full_scale = p["pts"], p["used"], float(p["full_scale"])
    w = n["weighted"]; K, c = w["K"][0], w["intercept_dn2"][0]; sr = w["sigma_d"][0]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(pts[used, 0], pts[used, 1], "o", color="C0", label="flat-field pairs (fitted)")
    ax.plot(pts[~used, 0], pts[~used, 1], "o", mfc="none", color="C0", label="above 70% of saturation (excluded)")
    x = np.linspace(0, pts[:, 0].max() * 1.05, 100)
    ax.plot(x, K * x + c, "k-", lw=1, label=f"weighted fit: K = {K:.4f} DN/e$^-$, intercept {c:.2f} DN$^2$")
    ax.axvline(0.7 * full_scale, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("mean above black (DN)"); ax.set_ylabel("variance (DN$^2$)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "ptc.png")

    e, snr = pts[:, 0] / K, pts[:, 0] / np.sqrt(pts[:, 1])
    ds = p["dark_snr"]; shown = ds[:, 1] > 0
    mu = np.logspace(-2.5, np.log10(w["sat_capacity_e"][0]), 400)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.loglog(mu, mu / np.sqrt(mu + sr ** 2), "-", color="C2", lw=2, label="model $\\mu/\\sqrt{\\mu+\\sigma_d^2}$")
    ax.loglog(mu, np.sqrt(mu), "--", color="C2", lw=1, label="photon limit $\\sqrt{\\mu}$")
    ax.loglog(mu, mu / sr, ":", color="C1", lw=1, label="read-noise limit $\\mu/\\sigma_d$")
    ax.loglog(e[used], snr[used], "o", color="C0", label="flat-field pairs")
    ax.loglog(ds[shown, 1] / K, ds[shown, 1] / np.sqrt(ds[shown, 2]), "s", color="k", ms=5, label="lens-capped pairs")
    ax.axvline(sr ** 2, color="#999", lw=1); ax.text(sr ** 2 * 1.15, 0.003, f"knee\n{sr ** 2:.0f} e$^-$", fontsize=8)
    ax.set_ylim(1e-3, None); ax.set_xlabel("signal (electrons)"); ax.set_ylabel("SNR"); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "snr.png")


def delta_e_figure(col):
    d = col["chart1_per_patch"]; x = np.arange(24)
    b, a = np.array(d["before"]), np.array(d["after"])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(x - 0.2, b, 0.4, color="#c0703a", label=f"white balance only (mean {b.mean():.2f})")
    ax.bar(x + 0.2, a, 0.4, color="#2b7bb9", label=f"plus 3x3 matrix (mean {a.mean():.2f})")
    ax.set_xticks(x); ax.set_xticklabels([f"{i + 1} {nm}" for i, nm in enumerate(d["names"])], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("$\\Delta E_{00}$"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); save(fig, "delta_e_per_patch.png")


def main():
    geometry_panel(json.loads(out("op1_geometry.json").read_text()))
    mtf_curves()
    noise_figures(json.loads(out("op3_noise.json").read_text()))
    delta_e_figure(json.loads(out("op4_color.json").read_text()))
    print("wrote outputs/figures/")


if __name__ == "__main__":
    main()

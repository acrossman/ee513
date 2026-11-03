"""Condensed report (outputs/report_condensed.pdf, no LaTeX): the results, a match table against the values
printed in the published datasheet, the test summary, and the figures.

A value matches if, rounded to the number of decimals the datasheet prints, it equals the datasheet value;
the same for its uncertainty. Usage: python -m pipeline.report
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .paths import out

# (section, quantity, unit, function of the outputs -> (value, uncertainty or None), datasheet value,
#  datasheet uncertainty or None, decimals printed)
CHECKS = [
    ("OP1", "focal length f_x", "px", lambda o: o["g"]["params"]["fx"], 3477.0, 8.3, 1),
    ("OP1", "focal length f_y", "px", lambda o: o["g"]["params"]["fy"], 3474.2, 8.5, 1),
    ("OP1", "focal length, mean", "px", lambda o: o["g"]["f_px"], 3475.6, 5.9, 1),
    ("OP1", "principal point c_x", "px", lambda o: o["g"]["params"]["cx"], 2238.3, 6.9, 1),
    ("OP1", "principal point c_y", "px", lambda o: o["g"]["params"]["cy"], 1736.2, 3.8, 1),
    ("OP1", "radial k1", "", lambda o: o["g"]["params"]["k1"], 0.054, 0.005, 3),
    ("OP1", "radial k2", "", lambda o: o["g"]["params"]["k2"], -0.27, 0.03, 2),
    ("OP1", "radial k3", "", lambda o: o["g"]["params"]["k3"], 0.35, 0.07, 2),
    ("OP1", "tangential p1", "", lambda o: o["g"]["params"]["p1"], -0.0003, 0.0003, 4),
    ("OP1", "tangential p2", "", lambda o: o["g"]["params"]["p2"], 0.0001, 0.0004, 4),
    ("OP1", "RMS reprojection", "px", lambda o: o["g"]["rms_px"], 1.06, 0.30, 2),
    ("OP1", "views used", "", lambda o: (o["g"]["n_views"], None), 17, None, 0),
    ("OP1", "output pixel pitch (bound)", "um", lambda o: o["g"]["pitch_um_bound"], 1.56, 0.04, 2),
    ("OP1", "diagonal field of view", "deg", lambda o: o["g"]["fov_diag_deg"], 79.5, 0.1, 1),
    ("OP2", "MTF50 screen, center", "cy/px", lambda o: o["m"]["screen"]["center"]["mtf50"], 0.473, 0.017, 3),
    ("OP2", "MTF50 screen, bottom right", "cy/px", lambda o: (o["m"]["screen"]["corners"]["bottom-right"]["mtf50"], o["m"]["screen"]["corner_bound"]), 0.25, 0.09, 2),
    ("OP2", "MTF50 screen, top right", "cy/px", lambda o: (o["m"]["screen"]["corners"]["top-right"]["mtf50"], o["m"]["screen"]["corner_bound"]), 0.27, 0.09, 2),
    ("OP2", "MTF50 screen, top left (defocused)", "cy/px", lambda o: (o["m"]["screen"]["corners"]["top-left"]["mtf50"], o["m"]["screen"]["corner_bound"]), 0.09, 0.09, 2),
    ("OP2", "MTF50 screen, bottom left", "cy/px", lambda o: (o["m"]["screen"]["corners"]["bottom-left"]["mtf50"], o["m"]["screen"]["corner_bound"]), 0.20, 0.09, 2),
    ("OP2", "MTF50 screen, corner mean", "cy/px", lambda o: o["m"]["screen"]["corner_mean"], 0.22, 0.07, 2),
    ("OP2", "corner mean without top left", "cy/px", lambda o: o["m"]["screen"]["without_defocused"]["corner_mean"], 0.25, 0.04, 2),
    ("OP2", "worst corner without top left", "cy/px", lambda o: (o["m"]["screen"]["without_defocused"]["corner_worst"]["mtf50"], o["m"]["screen"]["without_defocused"]["corner_bound"]), 0.20, 0.04, 2),
    ("OP2", "center MTF50 / Nyquist", "%", lambda o: (100 * o["m"]["nyquist_fraction_center"], None), 95, None, 0),
    ("OP2", "modulation at Nyquist, screen", "", lambda o: o["m"]["screen"]["center"]["modulation_at_nyquist"], 0.47, 0.02, 2),
    ("OP2", "MTF50 printed target, center", "cy/px", lambda o: o["m"]["printed"]["center"], 0.189, 0.005, 3),
    ("OP2", "MTF50 printed target, corner mean", "cy/px", lambda o: o["m"]["printed"]["corner_mean"], 0.182, 0.008, 3),
    ("OP2", "modulation at Nyquist, print", "", lambda o: o["m"]["printed"]["modulation_at_nyquist"], 0.13, 0.03, 2),
    ("OP3", "conversion gain K", "DN/e-", lambda o: o["n"]["weighted"]["K"], 0.0702, 0.0011, 4),
    ("OP3", "read noise sigma_d", "e-", lambda o: o["n"]["weighted"]["sigma_d"], 7.1, 1.1, 1),
    ("OP3", "sigma_y.dark", "DN", lambda o: (o["n"]["sigma_y_dark_dn"], None), 0.543, None, 3),
    ("OP3", "SNR_max", "", lambda o: o["n"]["weighted"]["snr_max"], 116.8, 0.9, 1),
    ("OP3", "abs. sensitivity threshold", "e-", lambda o: o["n"]["weighted"]["mu_e_min"], 7.6, 1.1, 1),
    ("OP3", "saturation capacity", "e-", lambda o: o["n"]["weighted"]["sat_capacity_e"], 13650, 210, -1),
    ("OP3", "dynamic range (EMVA 1288)", "bit", lambda o: o["n"]["weighted"]["dr_emva"]["bit"], 10.8, 0.2, 1),
    ("OP3", "dynamic range", "dB", lambda o: (o["n"]["weighted"]["dr_emva"]["dB"][0], None), 65.1, None, 1),
    ("OP3", "photon/read-noise knee", "e-", lambda o: o["n"]["weighted"]["knee_e"], 50, 16, 0),
    ("OP3", "dark current (G1)", "e-/s", lambda o: o["n"]["weighted"]["dark_current_e_per_s"], 0.957, 0.016, 3),
    ("OP3", "vendor NoiseProfile gain", "DN/e-", lambda o: (o["n"]["vendor"]["K"], None), 0.0788, None, 4),
    ("OP3", "K, ordinary least squares", "DN/e-", lambda o: o["n"]["unweighted"]["K"], 0.0691, 0.0002, 4),
    ("OP3", "sigma_d, ordinary least squares", "e-", lambda o: o["n"]["unweighted"]["sigma_d"], 8.75, 0.64, 2),
    ("OP3", "K, 25 patches, illumination removed", "DN/e-", lambda o: (o["n"]["patch_grid_K"]["illumination_removed"]["mean"], o["n"]["patch_grid_K"]["illumination_removed"]["sd"]), 0.0699, 0.0005, 4),
    ("OP3", "single-frame K (wrong method)", "DN/e-", lambda o: (o["n"]["single_frame_K"], None), 0.1178, None, 4),
    ("OP4", "mean dE00, white balance only", "", lambda o: o["c"]["before"]["mean"], 10.64, 0.24, 2),
    ("OP4", "worst dE00, white balance only", "", lambda o: o["c"]["before"]["worst"], 18.05, 0.12, 2),
    ("OP4", "mean dE00, 3x3 CCM", "", lambda o: o["c"]["after"]["mean"], 3.77, 0.51, 2),
    ("OP4", "worst dE00, 3x3 CCM", "", lambda o: o["c"]["after"]["worst"], 10.71, 0.40, 2),
    ("OP4", "same CCM, LED light (mean)", "", lambda o: o["c"]["transfer"]["led"]["mean"], 4.23, 0.22, 2),
    ("OP4", "same CCM, outdoor light (mean)", "", lambda o: o["c"]["transfer"]["outdoors"]["mean"], 8.17, 0.47, 2),
    ("OP4", "matrix columns agree within", "deg", lambda o: (o["c"]["max_column_angle_deg"], None), 2.65, None, 2),
]


def fmt(v, d):
    return f"{v:.{max(d, 0)}f}" if d >= 0 else f"{round(v, d):.0f}"


def evaluate(o):
    rows, n_fail = [], 0
    for sec, name, unit, get, exp_v, exp_u, d in CHECKS:
        v, u = get(o)
        ok = round(v, d) == round(exp_v, d) and (exp_u is None or round(u, d) == round(exp_u, d))
        n_fail += not ok
        got = fmt(v, d) + ("" if exp_u is None else " ± " + fmt(u, d))
        want = fmt(exp_v, d) + ("" if exp_u is None else " ± " + fmt(exp_u, d))
        rows.append([sec, name, unit, got, want, "PASS" if ok else "FAIL"])
    return rows, n_fail


def text_page(pdf, title, lines, size=9):
    fig = plt.figure(figsize=(8.5, 11)); fig.text(0.07, 0.95, title, fontsize=15, weight="bold")
    y = 0.91
    for ln in lines:
        fig.text(0.07, y, ln, fontsize=size, family="monospace"); y -= 0.0165 * size / 9
    pdf.savefig(fig); plt.close(fig)


def table_pages(pdf, title, header, rows, per_page=46):
    for start in range(0, len(rows), per_page):
        chunk = rows[start:start + per_page]
        fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis("off")
        ax.set_title(title + ("" if start == 0 else " (continued)"), fontsize=13, weight="bold", loc="left")
        t = ax.table(cellText=chunk, colLabels=header, loc="upper center", cellLoc="left", colLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.25)
        for i, r in enumerate(chunk, start=1):
            t[i, len(header) - 1].set_text_props(color="green" if r[-1] == "PASS" else "red", weight="bold")
        pdf.savefig(fig); plt.close(fig)


def figure_page(pdf, title, png, caption):
    img = plt.imread(out("figures/" + png))
    fig = plt.figure(figsize=(8.5, 11)); fig.text(0.07, 0.95, title, fontsize=13, weight="bold")
    h = 0.84 * min(1.0, img.shape[0] / img.shape[1] * 8.5 / 11 / 0.84 * 0.86)
    ax = fig.add_axes([0.07, 0.93 - h, 0.86, h]); ax.imshow(img); ax.axis("off")
    fig.text(0.07, 0.9 - h, caption, fontsize=8.5, wrap=True, va="top")
    pdf.savefig(fig); plt.close(fig)


def main():
    o = {k: json.loads(out(f).read_text()) for k, f in
         (("g", "op1_geometry.json"), ("m", "op2_mtf.json"), ("n", "op3_noise.json"), ("c", "op4_color.json"))}
    rows, n_fail = evaluate(o)
    tests = out("tests.txt").read_text().strip() if out("tests.txt").exists() else "(tests not run)"
    r = json.loads(out("results.json").read_text())
    summary = [
        "Google Pixel 7a, main (1x) camera, from raw DNG files.",
        "",
        f"Match with the published datasheet: {len(rows) - n_fail} of {len(rows)} values match "
        f"({'ALL MATCH' if n_fail == 0 else str(n_fail) + ' DIFFER'}).",
        f"Unit tests on synthetic data with known answers: {tests}",
        "",
        "results.json (value, uncertainty):",
    ] + [f"  {k:34s} {v}" for k, v in r.items() if k != "_README"]
    with PdfPages(out("report_condensed.pdf")) as pdf:
        text_page(pdf, "Pixel 7a Main Camera: Measured Datasheet (condensed)", summary, size=8)
        table_pages(pdf, "Computed values against the published datasheet",
                    ["OP", "quantity", "unit", "computed", "datasheet", "match"], rows)
        figure_page(pdf, "OP1 Geometry", "geometry_panel.png",
                    "Left: original raw capture. Center: undistorted, with arrows (x15) from each grid point's original "
                    "position to its undistorted one; orange arrows lie beyond the calibrated field. Right: size of the "
                    "inward shift. Green circle: field where the k1 and k2 terms are equal; dashed: the farthest "
                    "calibration corner, beyond which the model is extrapolated.")
        figure_page(pdf, "OP2 Spatial resolution", "mtf_screen_vs_print.png",
                    "Mean MTF (band: +/-1 std. dev. over edges), same +/-8 px LSF window for all. Blue: slanted edge on a "
                    "laptop screen (the camera without a print). Red: printed target at 179 mm.")
        figure_page(pdf, "OP3 Noise: photon transfer curve", "ptc.png",
                    "Half the pair-difference variance against the mean above black, central 128 x 128 G1 patch; open "
                    "points (above 70% of saturation) excluded. Slope K, intercept K^2 sigma_d^2.")
        figure_page(pdf, "OP3 Noise: signal-to-noise ratio", "snr.png",
                    "SNR against signal with the model from K and sigma_d; squares: lens-capped pairs (signal = dark "
                    "current), in the read-noise regime below the knee.")
        figure_page(pdf, "OP4 Color", "delta_e_per_patch.png",
                    "Per-patch CIEDE2000 before and after the 3x3 matrix, incandescent light, chart 1 of 3 (legend means "
                    "are this chart's; the datasheet values are the mean over three charts).")
    print(f"wrote outputs/report_condensed.pdf: {len(rows) - n_fail}/{len(rows)} values match the datasheet")
    for row in rows:
        if row[-1] == "FAIL":
            print("  differs:", row[1], row[3], "vs", row[4])
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

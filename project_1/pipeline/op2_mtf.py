"""OP2, spatial resolution: slanted-edge MTF50 on the G1 green plane, in cycles per green sample (Nyquist 0.5).

1. Screen target (the camera without a print): a 95.9 mm square tilted 5 degrees, shown on a laptop screen.
   Its four sides are measured with the strip method (camchar.strip) in 9 captures at 1.1 m near the
   center and in pairs of captures with the square in each corner. Distance from the square's size.
   Corners: each is one setup, so each corner value carries a stated bound, half the range between the
   corner means; the top-left pair was out of focus (focus on a near object) and is reported, and the
   values without it are given too.
2. Printed target (comparison): 10 captures of the printed five-square target at 179 mm; the center
   square on both near-vertical edges, each corner square on its outer edge, rows crossed by handwriting
   dropped (camchar.mtf).
3. MTF curves of both with the same +/-8 px LSF window, for the figure and the modulation at Nyquist.
Writes outputs/op2_mtf.json and outputs/op2_curves.npz.   Usage: python -m pipeline.op2_mtf
"""
import json

import numpy as np

from camchar import mtf, raw, screen, strip
from .paths import images, out

SQUARE_MM = 95.9                 # the screen square's side (capture_tools/make_screen_target.py, 0.1615 mm pixels)
DEFOCUSED = {"IMG_20261102_124214.dng", "IMG_20261102_124215.dng"}   # top-left pair, focused on a near object
GRID = np.linspace(0, 0.75, 151)
REGIONS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]


def mean_sd(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1))]


def green(path):
    return raw.green_plane(path, black="file", normalize=True)


def screen_edges(files, f_green):
    """Per capture: position, field, distance, and the strip MTF50 and curve of each unclipped side."""
    rows = []
    for f in files:
        g = green(f)
        H, W = g.shape
        quad = screen.find_target_square(g)
        assert quad is not None, f"{f.name}: target square not found"
        quad = screen.refine(g, quad)
        side = np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
        c = quad.mean(0)
        where = ("top" if c[1] < H / 3 else "bottom" if c[1] > 2 * H / 3 else "middle") + "-" + \
                ("left" if c[0] < W / 3 else "right" if c[0] > 2 * W / 3 else "center")
        edges, curves = [], []
        for i in range(4):
            cur = strip.strip_curve(g, quad[i], quad[(i + 1) % 4], min(0.3 * side, 24.0))
            if cur is not None:                                     # None: clipped pixels in the strip
                edges.append(mtf.mtf50(*cur)); curves.append(np.interp(GRID, *cur))
        rows.append(dict(file=f.name, position=where, distance_mm=float(f_green * SQUARE_MM / side),
                         field=float(np.hypot(c[0] - W / 2, c[1] - H / 2) / np.hypot(W / 2, H / 2)),
                         mtf50=edges, curves=curves))
    return rows


def printed_target(files):
    """MTF50 per region and capture (default slanted_edge_mtf), and +/-8 px curves for center and corners."""
    per_region = {r: [] for r in REGIONS}
    curves = {"center": [], "corners": []}
    for f in files:
        g = green(f)
        squares = mtf.find_squares(g)
        assert set(squares) == set(REGIONS), f"{f.name}: found {sorted(squares)}"
        for r in REGIONS:
            sides = ("left", "right") if r == "center" else ("left" if r.endswith("left") else "right",)
            vals = []
            for side in sides:
                crop, _ = mtf.edge_crop(g, squares[r], side, width=128, height_frac=0.6)
                crop, _, _ = mtf.clean_rows(crop)
                assert crop.shape[0] >= 100, f"{f.name} {r} {side}: too few clean rows"
                vals.append(mtf.mtf50(*mtf.slanted_edge_mtf(crop)[:2]))
                fr, m = mtf.slanted_edge_mtf(crop, window=4.0, lsf_half_width=strip.LSF_HALF)[:2]
                curves["center" if r == "center" else "corners"].append(np.interp(GRID, fr, m))
            per_region[r].append(float(np.mean(vals)))
    return per_region, curves


def main():
    geo = json.loads(out("op1_geometry.json").read_text())
    f_green = geo["f_px"][0] / 2

    center = screen_edges(images("screen_center"), f_green)
    corners = screen_edges(images("screen_corners"), f_green)
    c_vals = [v for r in center for v in r["mtf50"]]
    by_corner = {}
    for r in corners:
        by_corner.setdefault(r["position"], []).extend(r["mtf50"])
    means = {k: float(np.mean(v)) for k, v in by_corner.items()}
    in_focus = {k: v for k, v in by_corner.items() if not any(r["file"] in DEFOCUSED and r["position"] == k for r in corners)}
    means_in = {k: means[k] for k in in_focus}
    all_edges = [v for vs in by_corner.values() for v in vs]
    in_edges = [v for vs in in_focus.values() for v in vs]
    worst, worst_in = min(means, key=means.get), min(means_in, key=means_in.get)

    print_regions, print_curves = printed_target(images("printed_target"))
    pc = np.array(print_regions["center"])
    p_corner_mean = np.mean([print_regions[r] for r in REGIONS[1:]], axis=0)

    sc_curves = np.array([c for r in center for c in r["curves"]])
    so_curves = np.array([c for r in corners if r["file"] not in DEFOCUSED for c in r["curves"]])
    nyq = int(np.argmin(np.abs(GRID - 0.5)))
    result = dict(
        screen=dict(
            center=dict(mtf50=mean_sd(c_vals), n_edges=len(c_vals), n_captures=len(center),
                        distance_mm=mean_sd([r["distance_mm"] for r in center]),
                        field=float(np.mean([r["field"] for r in center])),
                        per_capture=[float(np.mean(r["mtf50"])) for r in center],
                        modulation_at_nyquist=mean_sd(sc_curves[:, nyq])),
            corners={k: dict(mtf50=means[k], n_edges=len(v), sd_edges=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                             distance_mm=float(np.mean([r["distance_mm"] for r in corners if r["position"] == k])),
                             field=float(np.mean([r["field"] for r in corners if r["position"] == k])),
                             defocused=k not in in_focus)
                     for k, v in by_corner.items()},
            corner_bound=(max(means.values()) - min(means.values())) / 2,
            corner_mean=mean_sd(all_edges), corner_worst=dict(position=worst, mtf50=means[worst]),
            without_defocused=dict(corner_bound=(max(means_in.values()) - min(means_in.values())) / 2,
                                   corner_mean=mean_sd(in_edges), corner_worst=dict(position=worst_in, mtf50=means_in[worst_in])),
            corner_modulation_at_nyquist=mean_sd(so_curves[:, nyq])),
        printed=dict(
            regions={r: mean_sd(v) for r, v in print_regions.items()},
            center=mean_sd(pc), corner_mean=mean_sd(p_corner_mean),
            modulation_at_nyquist=mean_sd(np.array(print_curves["center"])[:, nyq])),
        nyquist_fraction_center=float(np.mean(c_vals) / 0.5))
    out("op2_mtf.json").write_text(json.dumps(result, indent=2))
    np.savez(out("op2_curves.npz"), grid=GRID, screen_center=sc_curves, screen_corners=so_curves,
             print_center=np.array(print_curves["center"]), print_corners=np.array(print_curves["corners"]))
    s = result["screen"]
    print(f"screen center MTF50 {s['center']['mtf50'][0]:.3f} +/- {s['center']['mtf50'][1]:.3f} ({len(c_vals)} edges)")
    print("screen corners:", {k: round(v["mtf50"], 3) for k, v in s["corners"].items()}, f"bound +/-{s['corner_bound']:.3f}")
    print(f"printed center {result['printed']['center'][0]:.4f} +/- {result['printed']['center'][1]:.4f}, "
          f"corner mean {result['printed']['corner_mean'][0]:.4f} +/- {result['printed']['corner_mean'][1]:.4f}")


if __name__ == "__main__":
    main()

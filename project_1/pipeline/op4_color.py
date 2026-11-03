"""OP4, color: CIEDE2000 of three ColorChecker Classic Mini charts under incandescent light (ISO 399),
before and after a 3 x 3 color correction matrix, and the same matrix carried to LED and outdoor light.

For each light: the white-sheet photo gives the shading correction; each chart photo is reduced to linear
quarter-resolution RGB, its 24 patches found (cv2.mcc) and averaged over their middle half (camchar.color).
Under incandescent light each chart gets its own white balance (mid-gray patch 21) and matrix; "before" treats
the balanced RGB as linear sRGB. Under LED and outdoor light each chart is re-balanced but keeps its
incandescent matrix ("transfer"), and a matrix refitted under that light is given for comparison.
The uncertainty is the standard deviation over the three charts.
Writes outputs/op4_color.json.   Usage: python -m pipeline.op4_color
"""
import json

import numpy as np

from camchar import color, raw
from .paths import notes, out

LIGHTS = {"incandescent": "color_incandescent", "led": "color_led", "outdoors": "color_outdoors"}
PRIMARY = "incandescent"


def mean_sd(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1))]


def charts_under(role):
    """{chart number: 24 x 3 patch means} for one light, shading corrected."""
    files = notes(role)
    white = [f for f, n in files.items() if n == "white sheet"][0]
    flat = color.normalized_flat(white)
    cam = {}
    for f, n in sorted(files.items(), key=lambda kv: kv[1]):
        if not n.startswith("chart"):
            continue
        image, clipped = color.quarter_rgb(f)
        samples = np.array([color.middle_of(o) for o in color.find_patches(image)])   # found before correction
        with np.errstate(divide="ignore", invalid="ignore"):   # a few dark border pixels of the flat are 0; never sampled
            corrected = image / flat
        means, n_clipped = color.patch_means(corrected, clipped, samples)
        assert n_clipped == 0, f"{f.name}: {n_clipped} clipped pixels sampled"
        assert color.numbering_ok(means), f"{f.name}: patch numbering check failed"
        cam[int(n.split()[1])] = means
    return cam


def main():
    all_files = [f for role in LIGHTS.values() for f in notes(role)]
    print("lens_id:", raw.assert_one_camera(all_files))
    xyz_ref, names = color.reference()
    cam = {light: charts_under(role) for light, role in LIGHTS.items()}

    fits = {}
    for k, rgb in cam[PRIMARY].items():
        balanced = color.balance(rgb, xyz_ref)
        M = color.fit_ccm(balanced, xyz_ref)
        fits[k] = dict(M=M, before=color.delta_e_2000(balanced @ color.SRGB_TO_XYZ.T, xyz_ref),
                       after=color.delta_e_2000(balanced @ M.T, xyz_ref))
    angles = [color.angle_between(fits[a]["M"][:, j], fits[b]["M"][:, j]) for a, b in ((1, 2), (1, 3), (2, 3)) for j in range(3)]

    transfer = {}
    for light in ("led", "outdoors"):
        moved, refit = [], []
        for k, rgb in cam[light].items():
            balanced = color.balance(rgb, xyz_ref)
            moved.append(color.delta_e_2000(balanced @ fits[k]["M"].T, xyz_ref))
            refit.append(color.delta_e_2000(balanced @ color.fit_ccm(balanced, xyz_ref).T, xyz_ref).mean())
        transfer[light] = dict(mean=mean_sd([d.mean() for d in moved]), worst=mean_sd([d.max() for d in moved]),
                               refit_mean=mean_sd(refit))

    after_avg = np.mean([f["after"] for f in fits.values()], axis=0)
    before_avg = np.mean([f["before"] for f in fits.values()], axis=0)
    result = dict(
        metric="CIEDE2000", reference="ColorChecker24 - After November 2014",
        before=dict(mean=mean_sd([f["before"].mean() for f in fits.values()]),
                    worst=mean_sd([f["before"].max() for f in fits.values()]), worst_patch=names[int(before_avg.argmax())]),
        after=dict(mean=mean_sd([f["after"].mean() for f in fits.values()]),
                   worst=mean_sd([f["after"].max() for f in fits.values()]), worst_patch=names[int(after_avg.argmax())]),
        per_chart={k: dict(before_mean=float(f["before"].mean()), after_mean=float(f["after"].mean()),
                           after_worst=float(f["after"].max())) for k, f in fits.items()},
        max_column_angle_deg=max(angles), transfer=transfer,
        chart1_per_patch=dict(names=names, before=fits[1]["before"].tolist(), after=fits[1]["after"].tolist()))
    out("op4_color.json").write_text(json.dumps(result, indent=2))
    print(f"mean dE00 before {result['before']['mean'][0]:.2f} +/- {result['before']['mean'][1]:.2f}, "
          f"after {result['after']['mean'][0]:.2f} +/- {result['after']['mean'][1]:.2f}, "
          f"worst after {result['after']['worst'][0]:.2f} ({result['after']['worst_patch']})")
    print(f"LED {transfer['led']['mean'][0]:.2f}, outdoors {transfer['outdoors']['mean'][0]:.2f} "
          f"(refit {transfer['outdoors']['refit_mean'][0]:.2f}); matrices within {max(angles):.2f} deg")


if __name__ == "__main__":
    main()

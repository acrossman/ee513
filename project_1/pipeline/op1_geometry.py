"""OP1, geometry: intrinsics, distortion and derived quantities from 21 raw checkerboard views.

Each raw file is demosaiced (rawpy, native orientation) and converted to gray; corners of the 9 x 6 inner
grid are found and refined to sub-pixel precision; cv2.calibrateCameraExtended returns the intrinsics,
distortion and a standard deviation for each. The board's squares are 19.2 mm (the target printed at 96%,
read as 96 mm on its 100 mm scale bar); the square size scales distances, not the intrinsics.
Writes outputs/op1_geometry.json.   Usage: python -m pipeline.op1_geometry
"""
import json

import numpy as np

from camchar import geometry, raw
from .paths import images, out

PATTERN = (9, 6)
SQUARE_MM = 0.96 * 20.0
F_MM = 5.43                  # EXIF focal length (checked below against the files)
SPEC_PITCH_UM = 1.6          # Google: 0.8 um photodiodes, 2 x 2 binned
SPEC_FOV_DEG = 80.0


def main():
    files = images("checkerboard")
    print("lens_id:", raw.assert_one_camera(files))
    exif_f = {round(raw.lens_id(f)[1], 3) for f in files}
    assert exif_f == {F_MM}, exif_f

    points, used = [], []
    for f in files:
        corners = geometry.find_corners(raw.load_gray(f), PATTERN)
        if corners is None:
            print("  board not detected:", f.name)
        else:
            points.append(corners); used.append(f.name)
    print(f"{len(points)} of {len(files)} views detected")
    W, H = 4624, 3472
    cal = geometry.calibrate(points, (W, H), PATTERN, SQUARE_MM)

    f_px, s_f_px = cal.f_px
    rms_views = cal.per_view_rms
    s_rms = float(np.sqrt(np.sum((rms_views - cal.rms) ** 2) / (len(rms_views) - 1)))   # spread about the overall RMS
    pitch = 1000 * F_MM / f_px
    s_pitch_stat = pitch * np.hypot(s_f_px / f_px, 0.005 / F_MM)
    diag = np.hypot(W, H)
    fov = geometry.field_of_view(f_px, diag)
    s_fov = float(np.degrees(diag / f_px ** 2 / (1 + (diag / 2 / f_px) ** 2)) * s_f_px)
    d = cal.board_distances()
    breathing = geometry.focus_breathing(F_MM, d)
    k1, k2, k3 = (cal.params[k][0] for k in ("k1", "k2", "k3"))
    R = np.hypot(W / 2, H / 2)
    r = 0.72 * R / f_px
    terms = [f_px * r * k * r ** (2 * i) for i, k in enumerate((k1, k2, k3), start=1)]

    result = dict(
        n_views=len(points), n_files=len(files), views=used, square_mm=SQUARE_MM,
        params={k: list(v) for k, v in cal.params.items()},
        f_px=[f_px, s_f_px], f_mm=[F_MM, 0.005], rms_px=[cal.rms, s_rms],
        per_view_rms_px=rms_views.tolist(),
        pitch_um=[pitch, s_pitch_stat], pitch_um_bound=[pitch, abs(SPEC_PITCH_UM - pitch)],
        pitch_gap_to_spec_pct=100 * (pitch / SPEC_PITCH_UM - 1),
        fov_diag_deg=[fov, s_fov], fov_gap_to_spec_pct=100 * (fov / SPEC_FOV_DEG - 1),
        board_distance_mm=[float(d.min()), float(d.mean()), float(d.max())],
        focus_breathing_mean_pct=float(100 * breathing.mean()),
        radial_terms_at_field_072_px=dict(k1=terms[0], k2=terms[1], k3=terms[2], net=sum(terms)),
        field_k1_equals_k2=float(np.sqrt(abs(k1 / k2)) * f_px / R),
        max_corner_field=float(max(np.hypot(*(np.reshape(c, (-1, 2)) - [W / 2, H / 2]).T).max() for c in points) / R),
        K=cal.K.tolist(), dist=cal.dist.tolist())
    out("op1_geometry.json").write_text(json.dumps(result, indent=2))
    p = cal.params
    print(f"fx {p['fx'][0]:.1f} +/- {p['fx'][1]:.1f}  fy {p['fy'][0]:.1f} +/- {p['fy'][1]:.1f}  RMS {cal.rms:.3f} px")
    print(f"pitch {pitch:.3f} um ({result['pitch_gap_to_spec_pct']:+.1f}% vs spec), FOV {fov:.2f} +/- {s_fov:.2f} deg")


if __name__ == "__main__":
    main()

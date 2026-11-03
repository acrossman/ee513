"""Write outputs/results.json: every reported quantity as [value, uncertainty] in the units named by its key,
read from the measurement outputs (nothing retyped).   Usage: python -m pipeline.make_results
"""
import json

from .paths import out


def pair(v, digits):
    return [round(float(v[0]), digits), round(float(v[1]), digits)]


def main():
    g = json.loads(out("op1_geometry.json").read_text())
    m = json.loads(out("op2_mtf.json").read_text())
    n = json.loads(out("op3_noise.json").read_text())
    c = json.loads(out("op4_color.json").read_text())
    p, w, s = g["params"], n["weighted"], m["screen"]
    results = {
        "_README": "Each entry is [value, uncertainty] in the units named by the key; null if not measured.",
        "camera": "Pixel 7a",
        "capture_app": "Open Camera (Camera2 API, JPEG and DNG)",
        "cfa_pattern": "RGGB",
        "raw_width_px": 4624,
        "raw_height_px": 3472,
        "binned_from_higher_resolution_sensor": True,      # 64 MP Quad Bayer read out as 16 MP
        "focal_length_px": pair(g["f_px"], 2),
        "focal_length_mm": g["f_mm"],                     # EXIF, +/- half its last stored digit
        "principal_point_x_px": pair(p["cx"], 2),
        "principal_point_y_px": pair(p["cy"], 2),
        "distortion_k1": pair(p["k1"], 5),
        "distortion_k2": pair(p["k2"], 5),
        "distortion_p1": pair(p["p1"], 6),
        "distortion_p2": pair(p["p2"], 6),
        "reprojection_rms_px": pair(g["rms_px"], 3),
        "n_calibration_images_used": g["n_views"],
        "checkerboard_square_mm_measured": [g["square_mm"], 0.1],   # 96 mm read on the 100 mm bar, +/- 0.5 mm
        # Screen target; corners carry the stated bound (half the range between corner means), all four corners
        "mtf50_center_cy_per_px": pair(s["center"]["mtf50"], 4),
        "mtf50_corner_mean_cy_per_px": pair(s["corner_mean"], 4),
        "mtf50_corner_worst_cy_per_px": pair([s["corner_worst"]["mtf50"], s["corner_bound"]], 4),
        "conversion_gain_dn_per_e": pair(w["K"], 5),
        "read_noise_e": pair(w["sigma_d"], 3),
        "dark_current_e_per_s": pair(w["dark_current_e_per_s"], 4),
        "iso_of_noise_measurement": 45,
        "black_level_dn": pair(n["black_dn"], 4),
        "white_level_dn": n["white_dn"],
        "vendor_noiseprofile_present": True,
        "dynamic_range_stops": pair(w["dr_emva"]["bit"], 2),
        "dynamic_range_criterion": "EMVA 1288: saturation capacity (white level, 1023 DN) over the absolute "
                                   "sensitivity threshold sqrt(sigma_d^2 + 1/4) + 1/2, the signal where SNR = 1, "
                                   "in electrons; log2",
        "delta_e_metric": c["metric"],
        "colorchecker_reference_set": c["reference"],
        "delta_e_mean_before_ccm": pair(c["before"]["mean"], 2),
        "delta_e_mean_after_ccm": pair(c["after"]["mean"], 2),
        "delta_e_worst_after_ccm": pair(c["after"]["worst"], 2),
    }
    out("results.json").write_text(json.dumps(results, indent=2))
    print("wrote outputs/results.json")


if __name__ == "__main__":
    main()

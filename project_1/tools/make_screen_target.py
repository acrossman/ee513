"""Make a slanted-edge target to show on a laptop screen, at the screen's exact pixel size.

A screen edge has no toner spread, so it removes the print from the MTF measurement. The image holds:
  - one dark square tilted 5 degrees (ISO 12233 style), its edges anti-aliased by pixel area
    (16 x 16 samples per screen pixel, mixed in linear light, then sRGB-encoded for display);
  - a 4:1 contrast in linear light (0.60 light, 0.15 dark), as ISO 12233 recommends, so neither side
    clips or sits in the noise;
  - a 1-pixel stripe strip at the top to check the viewer is not scaling the image: shown 1:1 it looks
    like smooth gray from arm's length; scaled, it shows bands (moire);
  - a scale bar of known physical length, if the pixel pitch is given, so the distance can be
    measured from the photo the way the 44 mm squares were used for the printed target.

It prints the minimum shooting distance at which one screen pixel covers at most a quarter of a
green-plane pixel of the Pixel 7a (green sample spacing 2 x 1.562 um, f = 5.43 mm), so the camera
cannot resolve the screen's pixel grid.

Usage (from the repository folder):
  python tools/make_screen_target.py --width 1920 --height 1080 --pitch-mm 0.1615
      (the laptop used: 1920 x 1080 over 310 mm, so 0.1615 mm per pixel)
  python tools/make_screen_target.py --width 2560 --height 1600 --diagonal-in 13.3
Writes tools/screen_target_<width>x<height>.png and a .json with the square's size in mm.

Display it full screen at 100% zoom with display scaling that does not resample it (e.g. on Linux
`feh --fullscreen --zoom 100 <png>`), in a dim room, with brightness low enough that the light side
does not clip in the raw file. Shoot raw, main camera at 1x, fixed ISO, 1/60 s or longer (screens
dim and refresh in bands), from at least the printed distance, about 10 captures.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camchar.screen import DARK, LIGHT, TILT_DEG, coverage, srgb_encode

HERE = Path(__file__).resolve().parent
F_MM = 5.43                       # EXIF focal length
GREEN_PITCH_MM = 2 * 1.562e-3     # green-plane sample spacing (output pixel pitch x 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, required=True, help="screen width, px (native resolution)")
    ap.add_argument("--height", type=int, required=True, help="screen height, px")
    ap.add_argument("--pitch-mm", type=float, help="screen pixel pitch, mm")
    ap.add_argument("--diagonal-in", type=float, help="screen diagonal, inches (used if --pitch-mm is not given)")
    ap.add_argument("--square-frac", type=float, default=0.55, help="square side as a fraction of the screen height")
    args = ap.parse_args()

    pitch = args.pitch_mm
    if pitch is None and args.diagonal_in:
        pitch = 25.4 * args.diagonal_in / np.hypot(args.width, args.height)
    side = round(args.square_frac * args.height)

    lin = LIGHT + (DARK - LIGHT) * coverage(args.width, args.height, side, TILT_DEG)
    img = np.round(srgb_encode(lin) * 255).astype(np.uint8)
    dark8, light8 = (int(round(255 * float(srgb_encode(v)))) for v in (DARK, LIGHT))
    # 1-px stripe check strip along the top: alternating columns at the light and dark levels
    strip = np.where(np.arange(args.width) % 2 == 0, LIGHT, DARK)
    img[8:40] = np.round(srgb_encode(strip) * 255).astype(np.uint8)[None, :]
    info = dict(screen_px=[args.width, args.height], square_side_px=side, tilt_deg=TILT_DEG,
                contrast_linear=[LIGHT, DARK])
    if pitch:
        bar_mm = 50.0
        bar_px = round(bar_mm / pitch)
        y0 = args.height - 60
        img[y0:y0 + 6, 40:40 + bar_px] = dark8
        cv2.putText(img, f"{bar_mm:.0f} mm", (40, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, dark8, 2)
        d_min = pitch * F_MM / (0.25 * GREEN_PITCH_MM)
        info.update(pitch_mm=pitch, square_side_mm=side * pitch, scale_bar_mm=bar_mm, scale_bar_px=bar_px,
                    min_distance_mm=d_min)
        print(f"pixel pitch {pitch:.4f} mm; square {side} px = {side * pitch:.1f} mm on the screen")
        print(f"shoot from at least {d_min / 1000:.2f} m (one screen pixel <= 1/4 green pixel); "
              f"there the square is about {side * pitch * F_MM / d_min / GREEN_PITCH_MM:.0f} green px wide")
    else:
        print("no pixel pitch given: pass --pitch-mm or --diagonal-in to get the minimum distance and a scale bar")

    out = HERE / f"screen_target_{args.width}x{args.height}.png"
    cv2.imwrite(str(out), img)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()

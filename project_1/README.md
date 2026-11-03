# Pixel 7a Main Camera: A Measured Datasheet

A phone's spec sheet gives minimal information that tells little about its image quality. This project measures one camera, the **Google Pixel 7a main (1x) rear camera**, from its raw DNG files and publishes the result as a datasheet with measurements and uncertainties. The layout is inspired by the EMVA 1288 datasheet template but is not intended to represent compliance.

Everything here can be reproduced from the raw images with one command (see below), which also checks each computed value against the published datasheet.

| | |
|---|---|
| Camera | Pixel 7a, main rear camera at 1x; 64 MP Quad Bayer, read out 2 x 2 binned as 4624 x 3472 |
| Raw | DNG, Bayer RGGB, 10 bit (white level 1023), black level 64 DN (measured 64.009 ± 0.001) |
| App | Open Camera (Camera2 API), raw |
| Identity check | every series is checked to come from one camera: `('Pixel 7a', 5.43, None, (4624, 3472))` |

## Reproduce

Linux or macOS, with **Python 3.12** (check with `python3 --version`; on Ubuntu or Debian the `python3.12-venv` package is also needed). Windows is not supported directly; WSL works as Linux.

1. **Download the images** from the shared folder: **https://drive.google.com/drive/folders/1JX_4ep4-dyyncCy3exBf2hd51mXZa5CN?usp=sharing**
   - Put them in a folder named **`images/`** next to `reproduce.sh`, all in one folder (no subfolders).
   - Only the 110 files listed below (and in `images_manifest.csv`) are needed, 3.5 GB. The shared folder holds extra files; anything not listed is ignored, and a duplicate saved as `name (1).dng` does no harm.
   - Downloading the whole folder from Drive arrives as several zip files: unzip them all into `images/`.
2. **Run**, from this folder:
   ```bash
   bash reproduce.sh
   ```
   It creates a virtual environment (`.venv/`) with pinned library versions (`requirements.txt`), checks that every required image is present and unchanged (SHA-256 in `images_manifest.csv`), runs the unit tests and stops if any fails, runs the four measurements, and writes:
   - `outputs/results.json`: every reported value as `[value, uncertainty]`;
   - `outputs/report_condensed.pdf`: the results, a value-by-value **match table against the published datasheet** (each value rounded as the datasheet prints it), the test summary, and the figures;
   - `outputs/figures/` and one JSON file per measurement.

   About 10 minutes on a laptop, plus the library install. To choose the interpreter: `PYTHON=/path/to/python3.12 bash reproduce.sh`.

### Images needed, by operation point

<details><summary><b>OP1 Geometry</b>: 22 files</summary>

- **checkerboard** (21): checkerboard views for calibration (17 of the 21 are detected and used).  
  `IMG_20261002_155544.dng`, `IMG_20261002_155555.dng`, `IMG_20261002_155559.dng`, `IMG_20261002_155603.dng`, `IMG_20261002_155610.dng`, `IMG_20261002_155618.dng`, `IMG_20261002_155624.dng`, `IMG_20261002_155630.dng`, `IMG_20261002_155634.dng`, `IMG_20261002_155639.dng`, `IMG_20261002_155647.dng`, `IMG_20261002_155655.dng`, `IMG_20261002_155705.dng`, `IMG_20261002_155711.dng`, `IMG_20261002_155719.dng`, `IMG_20261008_121440.dng`, `IMG_20261008_121445.dng`, `IMG_20261008_121449.dng`, `IMG_20261008_121452.dng`, `IMG_20261008_121457.dng`, `IMG_20261008_121501.dng`
- **undistortion_example** (1): frame-filling frontal checkerboard, the undistortion example (not a calibration view).  
  `IMG_20261012_122925.dng`

</details>

<details><summary><b>OP2 Spatial resolution (MTF)</b>: 27 files</summary>

- **screen_center** (9): slanted-edge square on a laptop screen, near the frame center, 1.1 m.  
  `IMG_20261102_125314.dng`, `IMG_20261102_125315.dng`, `IMG_20261102_125317.dng`, `IMG_20261102_125318.dng`, `IMG_20261102_125319.dng`, `IMG_20261102_125320.dng`, `IMG_20261102_125321.dng`, `IMG_20261102_125323.dng`, `IMG_20261102_125324.dng`
- **screen_corners** (8): the same target in the frame corners, 0.66-0.94 m (the top-left pair is out of focus and is reported as such).  
  `IMG_20261102_124206.dng`, `IMG_20261102_124207.dng`, `IMG_20261102_124214.dng`, `IMG_20261102_124215.dng`, `IMG_20261102_124221.dng`, `IMG_20261102_124222.dng`, `IMG_20261102_124225.dng`, `IMG_20261102_124226.dng`
- **printed_target** (10): printed five-square slanted-edge target at 179 mm (comparison).  
  `IMG_20261013_123553.dng`, `IMG_20261013_123554.dng`, `IMG_20261013_123555.dng`, `IMG_20261013_123556.dng`, `IMG_20261013_123557.dng`, `IMG_20261013_123558.dng`, `IMG_20261013_123559.dng`, `IMG_20261013_123600.dng`, `IMG_20261013_123601.dng`, `IMG_20261013_123602.dng`

</details>

<details><summary><b>OP3 Noise and dynamic range</b>: 49 files</summary>

- **flat_pair** (24): flat-field pairs, ISO 45, 1/50-1/2 s (12 pairs).  
  `IMG_20261019_190358.dng`, `IMG_20261019_190359.dng`, `IMG_20261019_190411.dng`, `IMG_20261019_190413.dng`, `IMG_20261019_190419.dng`, `IMG_20261019_190421.dng`, `IMG_20261019_190428.dng`, `IMG_20261019_190430.dng`, `IMG_20261019_190449.dng`, `IMG_20261019_190451.dng`, `IMG_20261019_190508.dng`, `IMG_20261019_190510.dng`, `IMG_20261019_190515.dng`, `IMG_20261019_190516.dng`, `IMG_20261019_190527.dng`, `IMG_20261019_190529.dng`, `IMG_20261019_190547.dng`, `IMG_20261019_190550.dng`, `IMG_20261019_190559.dng`, `IMG_20261019_190601.dng`, `IMG_20261019_190610.dng`, `IMG_20261019_190613.dng`, `IMG_20261019_190619.dng`, `IMG_20261019_190622.dng`
- **dark_frame** (25): lens-capped frames, ISO 45, 1/40-16 s (5 at each of 5 exposures).  
  `IMG_20261019_172604.dng`, `IMG_20261019_172608.dng`, `IMG_20261019_172611.dng`, `IMG_20261019_172615.dng`, `IMG_20261019_172618.dng`, `IMG_20261019_172706.dng`, `IMG_20261019_172725.dng`, `IMG_20261019_172744.dng`, `IMG_20261019_172805.dng`, `IMG_20261019_172826.dng`, `IMG_20261019_172519.dng`, `IMG_20261019_172521.dng`, `IMG_20261019_172523.dng`, `IMG_20261019_172525.dng`, `IMG_20261019_172527.dng`, `IMG_20261019_172501.dng`, `IMG_20261019_172504.dng`, `IMG_20261019_172506.dng`, `IMG_20261019_172508.dng`, `IMG_20261019_172510.dng`, `IMG_20261019_172629.dng`, `IMG_20261019_172633.dng`, `IMG_20261019_172637.dng`, `IMG_20261019_172642.dng`, `IMG_20261019_172646.dng`

</details>

<details><summary><b>OP4 Color</b>: 12 files</summary>

- **color_incandescent** (4): 3 ColorChecker charts + white sheet, incandescent light, ISO 399.  
  `IMG_20261020_174844.dng` (chart 1), `IMG_20261020_174848.dng` (chart 2), `IMG_20261020_174852.dng` (chart 3), `IMG_20261020_174856.dng` (white sheet)
- **color_led** (4): 3 ColorChecker charts + white sheet, LED light, ISO 399.  
  `IMG_20261020_174233.dng` (chart 1), `IMG_20261020_174237.dng` (chart 2), `IMG_20261020_174241.dng` (chart 3), `IMG_20261020_174245.dng` (white sheet)
- **color_outdoors** (4): 3 ColorChecker charts + white sheet, outdoors near dusk, ISO 399.  
  `IMG_20261020_180143.dng` (chart 1), `IMG_20261020_180148.dng` (chart 2), `IMG_20261020_180153.dng` (chart 3), `IMG_20261020_180158.dng` (white sheet)

</details>

## Method

Measurements use linear raw data, black subtracted, without demosaicing, on the G1 green plane unless noted. Each uncertainty is a standard deviation over repeated captures, a fit covariance, or a stated bound with its argument.

| | Method | Code |
|---|---|---|
| **OP1 Geometry** | `cv2.calibrateCameraExtended` on the raw checkerboard views (9 x 6 inner corners, 19.2 mm squares: the print is at 96%); Brown-Conrady k1, k2, k3, p1, p2 | `pipeline/op1_geometry.py`, `camchar/geometry.py` |
| **OP2 Spatial resolution** | Slanted-edge MTF. The camera is measured on a square tilted 5° shown on a laptop screen (no toner spread), from a strip of pixels that follows each edge; the printed target, shot at only 179 mm, is kept as a comparison | `pipeline/op2_mtf.py`, `camchar/strip.py`, `camchar/screen.py`, `camchar/mtf.py` |
| **OP3 Noise** | Photon transfer from 10 flat-field pairs (half the variance of each pair's difference against its mean), central 128 x 128 patch, weighted least squares; dark current from 25 lens-capped frames; dynamic range by the EMVA 1288 criterion (SNR = 1) | `pipeline/op3_noise.py`, `camchar/noise.py` |
| **OP4 Color** | White balance from gray patch 21, then a 3 x 3 least-squares matrix to XYZ; CIEDE2000 against ColorChecker24 "After November 2014" (`colour-science`); 3 charts under incandescent light, and the same matrix under LED and outdoor light | `pipeline/op4_color.py`, `camchar/color.py` |

## Results

MTF50 in cycles per green-plane sample (Nyquist 0.5). As in `outputs/results.json`.

| Quantity | Measured | Uncertainty from | Manufacturer |
|---|---|---|---|
| Focal length | 3475.6 ± 5.9 px; 5.43 ± 0.005 mm | OpenCV fit std. dev.; EXIF rounding | 5.43 mm |
| Principal point | (2238.3 ± 6.9, 1736.2 ± 3.8) px | OpenCV fit std. dev. | — |
| Radial distortion k1, k2, k3 | 0.054 ± 0.005, −0.27 ± 0.03, 0.35 ± 0.07 | OpenCV fit std. dev. | — |
| Tangential distortion p1, p2 | −0.0003 ± 0.0003, 0.0001 ± 0.0004 | OpenCV fit std. dev. | — |
| RMS reprojection error | 1.06 ± 0.30 px | spread of per-view RMS | — |
| Output pixel pitch | 1.56 ± 0.04 µm | bound covering the 2.4% gap to spec (about half focus breathing) | 1.6 µm (2 x 0.8 µm) |
| Diagonal field of view | 79.5 ± 0.1° | propagated from f | 80° |
| MTF50, center (screen, 1.10 m) | 0.473 ± 0.017 (95% of Nyquist) | std. dev. over 36 edges | — |
| MTF50, corner mean / worst (screen) | 0.22 ± 0.07 / 0.09 ± 0.09 (top left out of focus; without it 0.25 ± 0.04 / 0.20 ± 0.04) | stated bound: half the range between corners | — |
| MTF50, printed target at 179 mm (comparison) | 0.189 ± 0.005 center | std. dev. over 10 captures | — |
| Conversion gain K | 0.0702 ± 0.0011 DN/e⁻ | weighted fit covariance | 0.0788 (DNG `NoiseProfile`) |
| Read noise | 7.1 ± 1.1 e⁻ | propagated from the fit | 7.19 e⁻ (DNG `NoiseProfile`) |
| Saturation capacity | 13 650 ± 210 e⁻ | propagated from K | — |
| Dynamic range | 10.8 ± 0.2 bit (EMVA 1288, SNR = 1) | propagated from K and read noise | — |
| Dark current | 0.957 ± 0.016 e⁻/s | slope fit and K | — |
| Mean ΔE₀₀ before / after 3 x 3 matrix | 10.64 ± 0.24 / 3.77 ± 0.51 | std. dev. over 3 charts | — |
| Worst ΔE₀₀ after matrix | 10.71 ± 0.40 (red) | std. dev. over 3 charts | — |

## Tests

`python -m pytest camchar/tests` (run automatically by `reproduce.sh`): 69 known-answer tests on synthetic data whose answer is known by construction, plus 3 known limitations recorded on purpose as expected failures.

| Code | Gold standard |
|---|---|
| `camchar.noise` | simulated Poisson–Gaussian sensor of known gain and read noise; exact noiseless line; unbiased over 40–60 sensors; uncertainty coverage of both fits; fixed pattern cancels in pair differences; single frames overestimate the gain; closed forms for full well, dynamic range, SNR and the vendor-tag conversion |
| `camchar.mtf` | Gaussian-blurred edges with analytic MTF50 (sigma 0.8–2.0 px), with noise, both polarities; rendered five-square target; a synthetic handwriting mark |
| `camchar.strip` | edges blurred by a Gaussian and integrated over each pixel's area: MTF50 within 5% for MTF50 0.18–0.41 at 5–15° tilt; polarity; clipped strips rejected |
| `camchar.screen` | drawn square's area; its edge measures as an ideal pixel-area edge; a rendered screen with a distractor: corners within 0.5 px |
| `camchar.color` | a camera meeting the Luther condition: the fitted matrix recovers the true one and ΔE ≈ 0; white balance neutralizes gray; CIELAB known values; patch numbering |
| `camchar.geometry` | calibration recovers known intrinsics from projected boards; square size scales distances, not intrinsics; corners found on a rendered board |
| `camchar.linear` | sRGB against `colour-science`; known values; round trip |

Expected failures: the ordinary least-squares gain uncertainty is about 3.6 times too small; the strip method reads 6–8% low on very sharp edges near 20° tilt; a short, shallow, point-sampled strip aliases high.

## Layout

| Path | Contents |
|---|---|
| `reproduce.sh`, `requirements.txt` | the one command, and the pinned libraries |
| `images_manifest.csv` | every required image: operation point, role, size, SHA-256 |
| `camchar/` | library: raw files and camera identity, geometry, MTF, strip MTF, screen target, noise, color, linear light; `tests/` |
| `pipeline/` | the four measurements, `results.json`, figures, condensed report, image check |
| `tools/make_screen_target.py` | the laptop-screen slanted-edge target, at a screen's exact pixel size |
| `outputs/` | results, figures and the condensed report from the last run |

## Use of AI tools

Claude (Anthropic) was used to generate these scripts. The methods and visualizations are often inspired by content from the Fall 2026 term EE 513 class at Portland State University taught by Dr. James McNames. Code was reviewed typically through smoke tests, stress tests, and sanity tests dictated by the author Crow Crossman. All images linked are content generated by Crow Crossman and not AI.

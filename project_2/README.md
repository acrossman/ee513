# An Image Signal Processor From Raw

EE 513 Project 2 (Portland State University, Fall 2026). Crow Crossman.

## Problem

Raw file in, displayable image out, with every stage written from scratch: the classical pipeline, a learned denoiser trained on the noise model measured in Project 1, an ONNX deployment with measured latency, and a check of whether any of it helps an object detector. Same camera as Project 1: the Google Pixel 7a main camera.

**Status (week 1):** burst, detail and first scene captures taken. Burst alignment, noise estimation, bilinear and Malvar-He-Cutler demosaicing, the small DnCNN and the ONNX export are carried over from homeworks 5 and 7, with tests. The pipeline stages, the results table and the one-command reproduction come next.

## Captures

All raw files are DNG, Bayer RGGB, 10 bit (black level 64 DN, saturation 1023 DN, measured in Project 1), 4624 x 3472, shot with Open Camera. The full set is not in this repository (each DNG is 32 MB); `data/` holds the crop the tests and reproduction run on.

| Set | Frames | ISO | Shutter (s) | Subject | How |
|---|---|---|---|---|---|
| Burst 1 | 20 | 45 | 1/60 | Apple IIe keyboard | phone held still on a solid surface, no touching between frames |
| Burst 2 | 20 | 499 | 1/60 | same | same |
| Burst 3 | 20 | 8000 | 1/60 | same | same; about two thirds of the green sites are clipped |
| Detail | 3 | 100 | 1/60, 1/10, 1 | bookshelf (fine text, saturated spines) | |
| Scenes | 5 so far | 100 to 8980 | 1/200 to 1/60 | everyday scenes, raw + phone JPEG from one shutter press | |

`data/burst_2_crop.npy` is the first 16 frames of burst 2, raw mosaic, rows 1480 to 1991 and columns 2300 to 2811 (even offsets, so it starts on a red site), `uint16`, shape (16, 512, 512). It was made with:

```
python tools/make_burst_crop.py <burst_2 folder> data/burst_2_crop.npy
```

## Code

| File | What it does | From |
|---|---|---|
| `isp/rawio.py` | load a DNG mosaic, check RGGB and the black level, scale to 0 to 1 | HW7 item 1 |
| `isp/burst.py` | align frames by Hann-windowed phase correlation and a sub-pixel shift | HW7 item 2 |
| `isp/noise.py` | noise from the difference of two frames (standard deviation and MAD / 0.6745), and the fit sigma^2 = read + k * brightness | HW7 item 4 |
| `isp/metrics.py` | PSNR (Bhandari, Kadambi and Raskar, Eq. 2.2) | HW7 item 5 |
| `isp/bilinear.py`, `isp/mhc.py` | bilinear and Malvar-He-Cutler demosaicing | HW5 |
| `isp/dncnn.py` | the lecture 11 SmallDnCNN, training on random patches, denoising | HW7 items 6 to 10 |
| `isp/deploy.py` | ONNX export with free height and width, onnxruntime run and timing | HW7 items 16 and 17 |

## Tests

```
uv sync --group learning
python -m pytest tests
```

The tests check that both demosaicing methods reproduce a constant image (bilinear everywhere, MHC away from the 2-pixel border) and keep every measured sample, that alignment recovers a known shift, that both noise estimates and the noise model fit recover known values, that PSNR of a known error is right, and that the ONNX model matches PyTorch.

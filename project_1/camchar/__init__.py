"""camchar: camera characterization from raw captures.

Functions for characterizing a camera from its raw captures:

- :mod:`camchar.raw`       reading DNGs: metadata, camera identity, green planes, pairing captures
- :mod:`camchar.geometry`  checkerboard calibration, distortion, pixel pitch, field of view
- :mod:`camchar.mtf`       slanted-edge MTF, MTF50, target-square location, MTF models
- :mod:`camchar.noise`     photon transfer curve, read noise, dynamic range, SNR, dark current
- :mod:`camchar.linear`    sRGB encoding and decoding, averaging in linear light
- :mod:`camchar.strip`     slanted-edge MTF from a strip that follows an edge (any orientation)
- :mod:`camchar.screen`    the laptop-screen slanted-edge target: drawing it and finding its square
- :mod:`camchar.color`     white balance, color correction matrix and CIEDE2000 on ColorChecker photos

Conventions: raw values are digital numbers (DN); the green plane is the G1 site of an
RGGB tile (row 0, column 1); "green px" means one green-plane sample, two sensor pixels apart.
"""
from . import color, geometry, linear, mtf, noise, raw, screen, strip

__version__ = "0.1.0"
__all__ = ["color", "geometry", "linear", "mtf", "noise", "raw", "screen", "strip"]

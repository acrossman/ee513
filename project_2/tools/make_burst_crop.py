import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.rawio import load_mosaic

N_FRAMES = 16
ROW, COL, SIZE = 1480, 2300, 512

folder = sys.argv[1]
out = sys.argv[2]
files = sorted(glob.glob(f"{folder}/*.dng"))[:N_FRAMES]
crop = np.stack([load_mosaic(f)[ROW:ROW + SIZE, COL:COL + SIZE] for f in files])
np.save(out, crop)
print(f"{len(files)} frames from {folder}")
print(f"crop rows {ROW}-{ROW + SIZE - 1}, columns {COL}-{COL + SIZE - 1}, shape {crop.shape}, {crop.dtype}")
print(f"saved {out}, {Path(out).stat().st_size / 1e6:.1f} MB")

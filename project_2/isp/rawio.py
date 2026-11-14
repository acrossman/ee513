import numpy as np
import rawpy

BLACK = 64.0
SATURATION = 1023.0


def load_mosaic(path):
    with rawpy.imread(path) as raw:
        pattern = "".join(chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten())
        assert pattern == "RGGB", pattern
        assert list(raw.black_level_per_channel) == [64, 64, 64, 64]
        return raw.raw_image_visible.copy()


def normalize(mosaic):
    return (mosaic.astype(np.float32) - BLACK) / (SATURATION - BLACK)


def green_sites(img):
    return img[0::2, 1::2]

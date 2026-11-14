import numpy as np


def psnr(img, reference, peak=1.0):
    return 10 * np.log10(peak ** 2 / np.mean((img - reference) ** 2))

import numpy as np
from scipy.ndimage import convolve

red_kernel = (1/4)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) 
green_kernel = (1/4)*np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
blue_kernel = red_kernel

# psu.mosaic_rggb.
SITES = [(0, 0, 0),    # red
         (0, 1, 1),    # green on red rows
         (1, 0, 1),    # green on blue rows
         (1, 1, 2)]    # blue


def bilinear(img):
    height, width = img.shape
    output = np.zeros((height, width, 3))

    for channel, kernel in [(0, red_kernel), (1, green_kernel), (2, blue_kernel)]:
        mask = np.zeros((height, width))
        for row, col, site_channel in SITES:
            if site_channel == channel:
                mask[row::2, col::2] = 1

        plane = img * mask
        total = convolve(plane, kernel, mode="constant", cval=0)
        weight = convolve(mask, kernel, mode="constant", cval=0)
        output[:, :, channel] = total / weight

    return output

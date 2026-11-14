import numpy as np
from scipy.ndimage import convolve
from isp.bilinear import bilinear

alpha = 1/2
beta = 5/8
gamma = 3/4

red_missing_green = alpha * (1/4)*np.array([[ 0, 0, -1, 0,  0],
                                            [ 0, 0,  0, 0,  0],
                                            [-1, 0,  4, 0, -1],
                                            [ 0, 0,  0, 0,  0],
                                            [ 0, 0, -1, 0,  0]])

red_missing_blue = gamma * (1/4)*np.array([[ 0, 0, -1, 0,  0],
                                           [ 0, 0,  0, 0,  0],
                                           [-1, 0,  4, 0, -1],
                                           [ 0, 0,  0, 0,  0],
                                           [ 0, 0, -1, 0,  0]])

blue_missing_green = red_missing_green
blue_missing_red = red_missing_blue

g_red_row_missing_red = beta * (1/10)*np.array([[ 0,  0,  1,  0,  0],
                                                [ 0, -2,  0, -2,  0],
                                                [-2,  0, 10,  0, -2],
                                                [ 0, -2,  0, -2,  0],
                                                [ 0,  0,  1,  0,  0]])

g_red_row_missing_blue = g_red_row_missing_red.T

g_blue_row_missing_blue = g_red_row_missing_red
g_blue_row_missing_red = g_red_row_missing_blue


cases = {"red_missing_green":       (0, 0, 1, red_missing_green),
         "red_missing_blue":        (0, 0, 2, red_missing_blue),
         "blue_missing_green":      (1, 1, 1, blue_missing_green),
         "blue_missing_red":        (1, 1, 0, blue_missing_red),
         "g_red_row_missing_red":   (0, 1, 0, g_red_row_missing_red),
         "g_red_row_missing_blue":  (0, 1, 2, g_red_row_missing_blue),
         "g_blue_row_missing_blue": (1, 0, 2, g_blue_row_missing_blue),
         "g_blue_row_missing_red":  (1, 0, 0, g_blue_row_missing_red)}


def mhc(img):
    output = bilinear(img)

    for name, (row, col, channel, kernel) in cases.items():
        fix = convolve(img, kernel, mode="constant", cval=0)
        output[row::2, col::2, channel] += fix[row::2, col::2]

    return output

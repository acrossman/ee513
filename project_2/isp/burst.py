import cv2
import numpy as np


def shift_of(a, b, win):
    (dx, dy), response = cv2.phaseCorrelate(a * win, b * win)
    return dx, dy, response


def align(frames):
    ref = frames[0]
    H, W = ref.shape
    window = cv2.createHanningWindow((W, H), cv2.CV_32F)
    shifts = [(0.0, 0.0)]
    aligned = [ref]
    for frame in frames[1:]:
        dx, dy, _ = shift_of(ref, frame, window)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned.append(cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT))
        shifts.append((dx, dy))
    return np.stack(aligned), np.array(shifts)

"""The laptop-screen slanted-edge target: drawing it, and finding its square in a photo.

A screen edge has no toner spread, so it measures the camera without the blur of a printed target.
:func:`coverage` draws a tilted square anti-aliased by pixel area; :func:`find_target_square` and
:func:`refine` locate the square's corners in a green plane to sub-pixel precision so each side can be
measured with :func:`camchar.strip.strip_mtf`. Tested in ``camchar/tests/test_screen.py``.
"""
import cv2
import numpy as np

LIGHT, DARK = 0.60, 0.15          # target levels, linear light: 4:1 contrast (ISO 12233 recommends it)
TILT_DEG = 5.0                    # square tilt
SS = 16                           # samples per screen pixel along each axis when drawing


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def coverage(width, height, side, tilt_deg):
    """Fraction of each screen pixel inside a centered square of `side` px tilted by tilt_deg."""
    th = np.radians(tilt_deg)
    c, s = np.cos(th), np.sin(th)
    out = np.zeros((height, width))
    sub = (np.arange(SS) + 0.5) / SS
    for row0 in range(0, height, 64):                 # in bands, to keep memory small
        rows = np.arange(row0, min(row0 + 64, height))
        y = (rows[:, None] + sub[None, :]).ravel() - height / 2
        x = (np.arange(width)[:, None] + sub[None, :]).ravel() - width / 2
        X, Y = np.meshgrid(x, y)
        u, v = c * X + s * Y, -s * X + c * Y          # coordinates in the square's frame
        inside = (np.abs(u) <= side / 2) & (np.abs(v) <= side / 2)
        out[rows[0]:rows[-1] + 1] = inside.reshape(len(rows), SS, width, SS).mean(axis=(1, 3))
    return out


def find_target_square(g):
    """Corners (4 x 2, green px) of the dark target square, or None."""
    sm = cv2.GaussianBlur(g, (0, 0), 1.5)
    local = cv2.GaussianBlur(sm, (0, 0), 40)
    mask = (sm < 0.7 * local).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue
        quad = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
        if len(quad) != 4 or not cv2.isContourConvex(quad.astype(np.int32)):
            continue
        sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
        if max(sides) / min(sides) > 1.15:
            continue
        inner = np.zeros(g.shape, np.uint8); cv2.fillConvexPoly(inner, quad.astype(np.int32), 1)
        outer = cv2.dilate(inner, np.ones((25, 25), np.uint8)) - inner
        ratio = g[inner == 1].mean() / g[outer == 1].mean()
        if ratio < 0.5 and (best is None or area > best[0]):
            best = (area, quad, ratio)
    return None if best is None else best[1]


def refine(g, quad):
    """Sharpen the corners: fit a line to each side's edge points, intersect neighbouring lines."""
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        L = np.linalg.norm(q - p); u = (q - p) / L; n = np.array([-u[1], u[0]])
        pts = []
        for t in np.linspace(0.15 * L, 0.85 * L, 60):
            c = p + t * u
            s = np.arange(-12, 12.5, 0.5)
            xy = (c[None, :] + s[:, None] * n[None, :]).astype(np.float32)
            prof = cv2.remap(g.astype(np.float32), xy[:, 0].reshape(1, -1), xy[:, 1].reshape(1, -1), cv2.INTER_LINEAR).ravel()
            d = np.abs(np.diff(prof)); k = int(np.argmax(d))
            if 0 < k < len(d) - 1:
                a, b, cc = d[k - 1], d[k], d[k + 1]
                pts.append(c + (s[k] + 0.25 + 0.25 * (a - cc) / (a - 2 * b + cc)) * n)
        pts = np.array(pts); m = pts.mean(0); _, _, vt = np.linalg.svd(pts - m)
        lines.append((m, vt[0]))
    corners = []
    for i in range(4):
        (m1, d1), (m2, d2) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.column_stack([d1, -d2]), m2 - m1)
        corners.append(m1 + t[0] * d1)
    return np.array(corners)

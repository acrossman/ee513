"""Linear light: the sRGB transfer function and averaging in linear light.

sRGB is a short linear segment near black joined to a 1/2.4 power curve,
which together approximate a 1/2.2 gamma; OpenCV has no standalone version of it.
"""
import numpy as np


def srgb_encode(x):
    """Linear light to sRGB-encoded values.

    Parameters
    ----------
    x : float or ndarray
        Linear values; clipped to [0, 1].

    Returns
    -------
    float or ndarray
        Encoded values in [0, 1]: ``12.92 x`` up to 0.0031308, ``1.055 x^(1/2.4) - 0.055`` above.

    Examples
    --------
    >>> srgb_encode(0.5)   # half the light
    0.7354
    """
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def srgb_decode(y):
    """sRGB-encoded values to linear light; the inverse of :func:`srgb_encode`.

    Parameters
    ----------
    y : float or ndarray
        Encoded values; clipped to [0, 1].

    Returns
    -------
    float or ndarray
        Linear values in [0, 1].

    Examples
    --------
    >>> srgb_decode(0.5)   # an encoded 0.5 carries 21% of the light
    0.2140
    """
    y = np.clip(y, 0, 1)
    return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def box_reduce(img, factor):
    """Downscale by averaging non-overlapping ``factor`` x ``factor`` blocks.

    Edge rows and columns that do not fill a whole block are dropped.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Image.
    factor : int
        Block size.

    Returns
    -------
    ndarray, shape (H // factor, W // factor)
    """
    h, w = (s // factor * factor for s in img.shape)
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def reduce_in_linear_light(encoded, factor):
    """Downscale sRGB-encoded values correctly: decode, average, re-encode.

    Averaging encoded values directly comes out too dark wherever a block mixes dark and
    bright pixels, because the sRGB curve bends downward.

    Parameters
    ----------
    encoded : ndarray, shape (H, W)
        sRGB-encoded image in [0, 1].
    factor : int
        Block size.

    Returns
    -------
    ndarray, shape (H // factor, W // factor)
        sRGB-encoded result.
    """
    return srgb_encode(box_reduce(srgb_decode(encoded), factor))

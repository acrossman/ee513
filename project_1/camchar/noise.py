"""Noise: photon transfer curve, read noise, dynamic range, SNR and dark current.

The model is
``Var[D] = K (E[D] - B) + K^2 sigma_r^2``: variance against mean signal is a line with slope
K (conversion gain, DN/e-) and intercept K^2 sigma_r^2 (read noise sigma_r in e-).
"""
from dataclasses import dataclass

import numpy as np


def ptc_point(a, b, black):
    """Mean signal and pair-difference variance for one flat-field pair.

    Parameters
    ----------
    a, b : ndarray
        The same patch from the two frames of a pair, in DN.
    black : float
        Black level, DN. Subtracted from the mean only; a constant does not change a variance.

    Returns
    -------
    mean : float
        Average of the two frame means minus the black level, DN.
    var : float
        Half the variance of ``a - b``, DN^2. Fixed-pattern noise is identical in both
        frames and cancels; the random parts add, so the variance is halved.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mean = (a.mean() + b.mean()) / 2 - black
    var = np.var(a - b) / 2
    return float(mean), float(var)


@dataclass
class PTCFit:
    """Result of :func:`fit_ptc`. Uncertainties are 1 sigma from the fit covariance.

    Attributes
    ----------
    K, sK : float
        Conversion gain and its uncertainty, DN/e-.
    intercept, s_intercept : float
        Intercept K^2 sigma_r^2 and its uncertainty, DN^2.
    sigma_r, s_sigma_r : float
        Read noise and its uncertainty, e-; 0 (and NaN) if the intercept is negative.
    cov : ndarray, shape (2, 2)
        Covariance of (slope, intercept).
    used : ndarray of bool
        Points included in the fit.
    method : str
        ``"unweighted"`` or ``"weighted"``.
    chi2_dof : float
        Weighted fits only: chi-square per degree of freedom of the residuals against
        the expected scatter of each variance. Near 1 if the points scatter only as noise
        predicts; above 1 means extra scatter. NaN for unweighted fits.
    """
    K: float
    sK: float
    intercept: float
    s_intercept: float
    sigma_r: float
    s_sigma_r: float
    cov: np.ndarray
    used: np.ndarray
    method: str = "unweighted"
    chi2_dof: float = float("nan")

    @property
    def correlation(self):
        """Correlation coefficient between slope and intercept."""
        return float(self.cov[0, 1] / (self.sK * self.s_intercept))


def fit_ptc(mean, var, full_scale, frac=0.7, weighted=False, n_pixels=None, inflate=True):
    """Fit the photon transfer curve below a fraction of saturation.

    Parameters
    ----------
    mean : ndarray
        Mean signal above black per pair, DN (from :func:`ptc_point`).
    var : ndarray
        Pair-difference variance per pair, DN^2.
    full_scale : float
        White level minus black level, DN.
    frac : float, optional
        Fit only points with ``mean < frac * full_scale``; near saturation clipping bends
        the variance over. Default 0.7.
    weighted : bool, optional
        False (default): ordinary (unweighted) least squares. True: weight
        each point by the inverse of its expected scatter, ``var * sqrt(2 / (n_pixels - 1))``,
        the standard deviation of a variance estimated from ``n_pixels`` samples.
    n_pixels : int, optional
        Pixels per patch (e.g. 16384 for 128 x 128). Required when ``weighted`` is True.
    inflate : bool, optional
        Weighted fits only: if the residuals scatter more than expected (``chi2_dof > 1``),
        scale the covariance by ``chi2_dof`` so the uncertainties include that extra
        scatter. Default True.

    Returns
    -------
    PTCFit

    Notes
    -----
    The read-noise uncertainty is propagated through ``sigma_r = sqrt(c) / K`` including
    the slope-intercept covariance.

    The unweighted fit's covariance assumes every point scatters equally, but a variance
    estimate scatters in proportion to its size, so the bright points are much noisier.
    On a simulated sensor its gain is unbiased but ``sK`` is about 3.6 times too small;
    the weighted fit gives correct uncertainties for both slope and intercept
    (``camchar/tests/test_noise.py``).

    Both fits assume the points lie on a straight line; a curved PTC (e.g. illumination
    differences between the frames of a pair) biases K high and the intercept low.

    Examples
    --------
    >>> fit_ptc(means, var_pair, 1023 - 64.009)                                    # ordinary least squares
    >>> fit_ptc(means, var_pair, 1023 - 64.009, weighted=True, n_pixels=128 * 128)
    """
    mean, var = np.asarray(mean, dtype=float), np.asarray(var, dtype=float)
    used = mean < full_scale * frac
    chi2_dof = np.nan
    if weighted:
        if n_pixels is None:
            raise ValueError("a weighted fit needs n_pixels, the number of pixels per patch")
        sigma = var[used] * np.sqrt(2 / (n_pixels - 1))
        (K, c), cov = np.polyfit(mean[used], var[used], 1, w=1 / sigma, cov="unscaled")
        resid = (var[used] - (K * mean[used] + c)) / sigma
        chi2_dof = float(np.sum(resid ** 2) / (used.sum() - 2))
        if inflate and chi2_dof > 1:
            cov = cov * chi2_dof
    else:
        (K, c), cov = np.polyfit(mean[used], var[used], 1, cov=True)
    sK, sc = np.sqrt(np.diag(cov))
    sigma_r = np.sqrt(max(c, 0)) / K
    if c > 0:
        d_c, d_K = 1 / (2 * K * np.sqrt(c)), -sigma_r / K
        s_sigma_r = np.sqrt(d_c ** 2 * sc ** 2 + d_K ** 2 * sK ** 2 + 2 * d_c * d_K * cov[0, 1])
    else:
        s_sigma_r = np.nan
    return PTCFit(float(K), float(sK), float(c), float(sc), float(sigma_r), float(s_sigma_r), cov, used,
                  "weighted" if weighted else "unweighted", chi2_dof)


def full_well(full_scale, K, sK=0.0):
    """Electrons at the white level.

    Parameters
    ----------
    full_scale : float
        White level minus black level, DN.
    K : float
        Conversion gain, DN/e-.
    sK : float, optional
        Uncertainty of K. Default 0.

    Returns
    -------
    value, uncertainty : float
        ``full_scale / K`` and its propagated uncertainty, e-.

    Notes
    -----
    This is the most the raw file can record. If the converter saturates before the
    photodiode fills, the physical capacity is higher.
    """
    fw = full_scale / K
    return float(fw), float(fw * sK / K)


def dynamic_range_stops(full_well_e, sigma_r, convention="read_noise", s_full_well=0.0, s_sigma_r=0.0):
    """Dynamic range in stops (powers of two).

    Parameters
    ----------
    full_well_e : float
        Saturation signal, e-.
    sigma_r : float
        Read noise, e-.
    convention : {"read_noise", "snr1"}, optional
        Low end of the range: ``"read_noise"`` takes the signal equal to the read noise;
        ``"snr1"`` takes the signal where SNR = 1. Default "read_noise".
    s_full_well, s_sigma_r : float, optional
        Uncertainties, for the propagated uncertainty. Default 0.

    Returns
    -------
    value, uncertainty : float
        Stops, ``log2(full_well / low_end)``. The uncertainty is propagated for the
        ``"read_noise"`` convention and NaN otherwise.
    """
    if convention == "read_noise":
        low, s = sigma_r, np.hypot(s_full_well / full_well_e, s_sigma_r / sigma_r) / np.log(2)
    elif convention == "snr1":
        low, s = snr_one_signal(sigma_r), np.nan
    else:
        raise ValueError(f"unknown convention {convention!r}")
    return float(np.log2(full_well_e / low)), float(s)


def snr_model(signal_e, sigma_r):
    """Predicted signal-to-noise ratio, ``mu / sqrt(mu + sigma_r^2)``.

    Parameters
    ----------
    signal_e : float or ndarray
        Mean signal mu, e-.
    sigma_r : float
        Read noise, e-.

    Returns
    -------
    float or ndarray
        SNR. Rises as sqrt(mu) where photon noise dominates and as mu / sigma_r where read
        noise dominates; the regimes meet at the knee ``mu = sigma_r^2``.
    """
    mu = np.asarray(signal_e, dtype=float)
    return mu / np.sqrt(mu + sigma_r ** 2)


def snr_one_signal(sigma_r):
    """Signal at which the model SNR equals 1.

    Parameters
    ----------
    sigma_r : float
        Read noise, e-.

    Returns
    -------
    float
        ``(1 + sqrt(1 + 4 sigma_r^2)) / 2``, e-.
    """
    return float((1 + np.sqrt(1 + 4 * sigma_r ** 2)) / 2)


def noise_profile_to_gain(S, O, full_scale):
    """Convert a DNG ``NoiseProfile`` (S, O) pair into a gain and read noise.

    The tag models variance in normalized signal as ``S x + O`` with ``x = (D - B)/(W - B)``.
    Multiplying through by ``(W - B)^2`` gives the PTC model in DN.

    Parameters
    ----------
    S, O : float
        Scale and offset for one color plane.
    full_scale : float
        White level minus black level used by the file, DN.

    Returns
    -------
    K : float
        ``S * full_scale``, DN/e-.
    sigma_r : float
        ``sqrt(O) * full_scale / K``, e-.
    read_var : float
        ``O * full_scale^2``, the read variance K^2 sigma_r^2 in DN^2.
    """
    K = S * full_scale
    read_var = O * full_scale ** 2
    return float(K), float(np.sqrt(read_var) / K), float(read_var)


def dark_current(exposure_s, level_dn, K, sK=0.0):
    """Dark current from the mean level of capped frames against exposure time.

    Parameters
    ----------
    exposure_s : ndarray
        Exposure time of each capture, s.
    level_dn : ndarray
        Mean raw level of each capture, DN.
    K : float
        Conversion gain, DN/e-.
    sK : float, optional
        Uncertainty of K. Default 0.

    Returns
    -------
    dict
        ``slope`` and ``offset`` (DN/s and DN) with ``s_slope``, ``s_offset``, and
        ``e_per_s`` with ``s_e_per_s`` combining the fit and gain uncertainties.

    Notes
    -----
    Dark current roughly doubles every 6-8 degrees C; record whether the phone was warm.
    """
    (slope, offset), cov = np.polyfit(np.asarray(exposure_s, float), np.asarray(level_dn, float), 1, cov=True)
    s_slope, s_offset = np.sqrt(np.diag(cov))
    e = slope / K
    return dict(slope=float(slope), s_slope=float(s_slope), offset=float(offset), s_offset=float(s_offset),
                e_per_s=float(e), s_e_per_s=float(abs(e) * np.hypot(s_slope / slope, sK / K)))

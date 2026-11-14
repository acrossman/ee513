import numpy as np


def frame_difference(a, b):
    return ((b - a) / np.sqrt(2)).ravel()


def sigma_std(d):
    return d.std()


def sigma_mad(d):
    return np.median(np.abs(d - np.median(d))) / 0.6745


def fit_noise_model(clean, d, groups=20):
    b_all = clean.ravel()
    edges = np.quantile(b_all, np.linspace(0, 1, groups + 1))
    group = np.clip(np.searchsorted(edges, b_all, side="right") - 1, 0, groups - 1)
    level = [np.median(b_all[group == g]) for g in range(groups)]
    var_g = [sigma_mad(d[group == g]) ** 2 for g in range(groups)]
    k_shot, read_var = np.polyfit(level, var_g, 1)
    return read_var, k_shot

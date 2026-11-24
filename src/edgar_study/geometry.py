"""Population code geometry as in Tilbury et al. (2025): binned repeats, rcvPCA and spectrum fits.

The paper bins the Stringer et al. recording's random grating directions into 256 direction bins with
exactly six repeats per bin. rcvPCA smooths one repeat with a circular Gaussian kernel whose width is
chosen by cross-validation, takes eigenvectors of its covariance, and estimates the stimulus-related
variance along each as the covariance of two other repeats, averaged over all 60 assignments of six
repeats. Power-law (n^-alpha) and Gaussian-decay (exp(-n^2 / 2M^2)) fits use dimensions 10 < n < 50.
"""

import itertools
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit

FIT_RANGE = (11, 49)
SMOOTHING_WIDTHS = np.geomspace(0.005, 1.0, 48)


class SpectrumFit(NamedTuple):
    slope: float
    intercept: float
    mse: float


def bin_repeats(response, angles, n_bins=256, n_repeats=6, seed=0):
    """Average each direction bin's trials into `n_repeats` repeats of distinct, randomly grouped trials.

    Trials beyond a multiple of `n_repeats` are discarded at random. Returns cells × bins × repeats.
    """
    response, angles = np.asarray(response), np.asarray(angles)
    rng = np.random.default_rng(seed)
    bins = np.floor(np.mod(angles, 2 * np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    out = np.empty((response.shape[0], n_bins, n_repeats))
    for b in range(n_bins):
        trials = np.flatnonzero(bins == b)
        if trials.size < n_repeats:
            raise ValueError(f"Direction bin {b} has fewer than {n_repeats} trials.")
        kept = rng.permutation(trials)[: trials.size // n_repeats * n_repeats].reshape(n_repeats, -1)
        out[:, b, :] = response[:, kept].mean(axis=2)
    return out


def bin_centres(n_bins=256):
    return (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins


def smoothing_kernel(n_bins, width):
    """Circular Gaussian smoothing matrix over `n_bins` evenly spaced angles, rows summing to one."""
    angles = 2 * np.pi * np.arange(n_bins) / n_bins
    difference = np.abs(angles[:, None] - angles[None, :])
    distance = np.minimum(difference, 2 * np.pi - difference)
    kernel = np.exp(-(distance**2) / (2 * width**2))
    return kernel / kernel.sum(axis=1, keepdims=True)


def pca_spectrum(curves):
    """PCA eigenvalues (descending) of cells × angles tuning curves, each centred over angles."""
    centred = np.asarray(curves) - np.mean(curves, axis=1, keepdims=True)
    covariance = centred.T @ centred / centred.shape[0]
    return np.linalg.eigvalsh(covariance)[::-1]


def rcvpca(binned, widths=SMOOTHING_WIDTHS):
    """Regularized cross-validated PCA spectrum of cells × bins × repeats responses.

    Averages over every choice of one repeat for the smoothed eigenvector estimate and an unordered pair
    of the others for the variance estimate. Values can be negative: the estimate is unbiased.
    """
    binned = np.asarray(binned, dtype=float)
    cells, n_bins, n_repeats = binned.shape
    if n_repeats < 3:
        raise ValueError("rcvPCA needs at least three repeats.")
    centred = binned - binned.mean(axis=1, keepdims=True)
    flat = centred.transpose(0, 2, 1).reshape(cells, n_repeats * n_bins)
    cross = (flat.T @ flat / cells).reshape(n_repeats, n_bins, n_repeats, n_bins).transpose(0, 2, 1, 3)
    kernels = [smoothing_kernel(n_bins, w) for w in widths]
    spectra = []
    for first in range(n_repeats):
        rest = [r for r in range(n_repeats) if r != first]
        for second, third in itertools.combinations(rest, 2):
            errors = [
                np.trace(k.T @ cross[first, first] @ k)
                - np.trace(k.T @ (cross[first, second] + cross[first, third]))
                for k in kernels
            ]
            k = kernels[int(np.argmin(errors))]
            _, vectors = np.linalg.eigh(k.T @ cross[first, first] @ k)
            vectors = vectors[:, ::-1]
            shared = (cross[second, third] + cross[third, second]) / 2
            spectra.append(np.einsum("in,ij,jn->n", vectors, shared, vectors))
    return np.mean(spectra, axis=0)


def independent_noise(residuals):
    """Rotate each repeat's residuals to a different direction bin so repeats share no noise.

    Residuals from the repeat mean sum to zero across repeats, so any two are negatively correlated
    and bias rcvPCA's cross-repeat estimate downward. Shifting repeat r by r × bins / repeats keeps
    every value and each cell's noise level but makes the repeats independent.
    """
    residuals = np.asarray(residuals)
    n_bins, n_repeats = residuals.shape[1:]
    return np.stack(
        [np.roll(residuals[:, :, r], r * (n_bins // n_repeats), axis=1) for r in range(n_repeats)], axis=2
    )


def _usable(spectrum, fit_range):
    n = np.arange(1, len(spectrum) + 1)
    keep = (n >= fit_range[0]) & (n <= fit_range[1]) & (np.asarray(spectrum) > 0)
    return n[keep].astype(float), np.log(np.asarray(spectrum)[keep])


def _fit(spectrum, fit_range, space, transform):
    n, log_values = _usable(spectrum, fit_range)
    feature = transform(n)
    slope, intercept = np.polyfit(feature, log_values, 1)
    if space == "linear":
        values = np.exp(log_values)
        (slope, intercept), _ = curve_fit(
            lambda f, a, b: np.exp(a * f + b), feature, values, p0=(slope, intercept), maxfev=20_000
        )
        residual = values - np.exp(slope * feature + intercept)
    elif space == "log":
        residual = log_values - (slope * feature + intercept)
    else:
        raise ValueError(f"Unknown fit space {space!r}.")
    return float(slope), float(intercept), float(np.mean(residual**2))


def fit_power_law(spectrum, fit_range=FIT_RANGE, space="log"):
    """Least-squares fit of λ_n = e^c n^−α over `fit_range`, in log or linear space; slope is α."""
    slope, intercept, mse = _fit(spectrum, fit_range, space, np.log)
    return SpectrumFit(-slope, intercept, mse)


def fit_gaussian_decay(spectrum, fit_range=FIT_RANGE, space="log"):
    """Least-squares fit of λ_n = e^c exp(−n² / 2M²) over `fit_range`, in log or linear space; slope is M."""
    coefficient, intercept, mse = _fit(spectrum, fit_range, space, np.square)
    width = float(np.sqrt(-1 / (2 * coefficient))) if coefficient < 0 else float("inf")
    return SpectrumFit(width, intercept, mse)

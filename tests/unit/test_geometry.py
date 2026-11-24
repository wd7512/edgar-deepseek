import numpy as np
import pytest

from edgar_study import geometry


def test_trials_are_binned_into_equal_repeats_of_distinct_trials():
    angles = np.repeat(np.linspace(0, 2 * np.pi, 8, endpoint=False) + 0.01, 13)
    response = np.tile(np.arange(angles.size, dtype=float), (3, 1))

    binned = geometry.bin_repeats(response, angles, n_bins=8, n_repeats=6, seed=0)

    assert binned.shape == (3, 8, 6)
    pair_sums = 2 * binned[0, 0]
    dropped = sum(range(13)) - pair_sums.sum()
    assert np.allclose(pair_sums, np.round(pair_sums))
    assert dropped in range(13)
    assert np.all((pair_sums >= 1) & (pair_sums <= 23))


def test_a_bin_with_fewer_trials_than_repeats_is_an_error():
    angles = np.array([0.1, 0.2, 3.3, 3.4, 3.5])
    with pytest.raises(ValueError, match="fewer than 3 trials"):
        geometry.bin_repeats(np.ones((1, 5)), angles, n_bins=2, n_repeats=3)


def test_the_smoothing_kernel_is_circular_and_rows_sum_to_one():
    kernel = geometry.smoothing_kernel(16, 0.4)

    assert np.allclose(kernel.sum(axis=1), 1)
    assert np.isclose(kernel[0, 1], kernel[0, 15])
    assert np.allclose(kernel, kernel.T)


def known_population(n_cells=400, n_bins=64, alpha=4.0, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(np.column_stack([np.ones(n_bins), rng.standard_normal((n_bins, n_bins - 1))]))
    basis = q[:, 1:]
    variances = np.arange(1, n_bins) ** -alpha
    loadings = rng.standard_normal((n_cells, n_bins - 1)) * np.sqrt(variances)
    return loadings @ basis.T, variances


def test_pca_spectrum_recovers_the_variances_of_a_known_population():
    curves, variances = known_population()

    spectrum = geometry.pca_spectrum(curves)

    assert spectrum.shape == (64,)
    assert np.allclose(spectrum[:5] / variances[:5], 1, atol=0.25)


def smooth_population(n_cells=2000, n_bins=64, alpha=4.0, seed=0):
    """Tuning curves built from Fourier modes with variance k^-alpha, as smooth as real tuning curves."""
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * np.arange(n_bins) / n_bins
    k = np.arange(1, n_bins // 4)
    modes = np.concatenate([np.cos(np.outer(theta, k)), np.sin(np.outer(theta, k))], axis=1)
    scale = np.sqrt(np.concatenate([k, k]) ** -alpha)
    return (rng.standard_normal((n_cells, modes.shape[1])) * scale) @ modes.T


def test_rcvpca_is_close_to_noise_free_pca_and_unbiased_under_pure_noise():
    signal = smooth_population()
    rng = np.random.default_rng(1)
    noisy = signal[:, :, None] + 0.5 * rng.standard_normal((2000, 64, 6))

    estimate = geometry.rcvpca(noisy)
    truth = geometry.pca_spectrum(signal)

    assert np.allclose(estimate[:6] / truth[:6], 1, atol=0.2)
    noise_only = geometry.rcvpca(rng.standard_normal((2000, 64, 6)))
    assert abs(noise_only[:10].mean()) < 0.02


def test_spectrum_fits_recover_a_power_law_and_prefer_it_to_gaussian_decay():
    n = np.arange(1, 101)

    power = geometry.fit_power_law(n**-4.0)
    gaussian = geometry.fit_gaussian_decay(n**-4.0)

    assert np.isclose(power.slope, 4.0)
    assert power.mse < 1e-12 < gaussian.mse


def test_gaussian_decay_fit_recovers_its_width():
    n = np.arange(1, 101)

    fit = geometry.fit_gaussian_decay(np.exp(-(n**2) / (2 * 9.0**2)))

    assert np.isclose(fit.slope, 9.0)


def test_fits_skip_negative_estimates():
    n = np.arange(1, 101).astype(float)
    spectrum = n**-3.0
    spectrum[20] = -1e-6

    assert np.isclose(geometry.fit_power_law(spectrum).slope, 3.0)


def test_repeat_residuals_bias_rcvpca_down_and_rotated_residuals_do_not():
    signal = smooth_population(alpha=2.0)
    rng = np.random.default_rng(2)
    recorded = signal[:, :, None] + rng.standard_normal((2000, 64, 6))
    residuals = recorded - recorded.mean(axis=2, keepdims=True)
    flat_tail = slice(30, 64)

    biased = geometry.rcvpca(np.zeros_like(recorded) + residuals)
    unbiased = geometry.rcvpca(np.zeros_like(recorded) + geometry.independent_noise(residuals))

    assert biased[flat_tail].mean() < -0.01
    assert abs(unbiased[flat_tail].mean()) < 0.005


def test_rotated_residuals_keep_each_repeats_values():
    residuals = np.random.default_rng(0).standard_normal((3, 12, 6))

    rotated = geometry.independent_noise(residuals)

    assert rotated.shape == residuals.shape
    assert np.allclose(np.sort(rotated, axis=1), np.sort(residuals, axis=1))
    assert not np.allclose(rotated[:, :, 1], residuals[:, :, 1])


def test_linear_space_fit_recovers_a_power_law():
    n = np.arange(1, 101)

    assert np.isclose(geometry.fit_power_law(3 * n**-4.0, space="linear").slope, 4.0, atol=1e-3)
    assert np.isclose(
        geometry.fit_gaussian_decay(np.exp(-(n**2) / 200), space="linear").slope, 10.0, atol=1e-3
    )

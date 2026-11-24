"""Tilbury et al. (2025)'s winning tuning curve, its parameter estimator, and the Fig. 3e ablations.

`ai_model` and `ai_estimator` are Supplementary Text 2.8 and 2.9 with NumPy clips replaced by JAX
clips. The ablations follow the methods: "Gaussian+p+asym" puts the second peak exactly opposite the
first, and "Gaussian+p" also gives each peak one width. Their estimators drop the matching entries
from the paper's estimate.
"""

import inspect

import jax.numpy as jnp
import numpy as np


def _signed(theta, centre):
    return jnp.arctan2(jnp.sin(theta - centre), jnp.cos(theta - centre))


def _peak(theta, centre, amplitude, left, right, exponent):
    distance = _signed(theta, centre)
    width = jnp.where(distance >= 0, right, left)
    return amplitude * jnp.exp(-0.5 * (jnp.abs(distance) / width) ** exponent)


def ai_model(
    theta,
    theta_pref=0.0,
    baseline=0.0,
    amplitude_1=1.0,
    amplitude_2=0.0,
    tuning_width_1_left=1.0,
    tuning_width_1_right=1.0,
    tuning_width_2_left=1.0,
    tuning_width_2_right=1.0,
    exponent_1=2.0,
    exponent_2=2.0,
    angle_offset_2=0.0,
):
    """The paper's 11-parameter offset asymmetric double generalized Gaussian."""
    theta_pref = jnp.clip(theta_pref, 0, 2 * jnp.pi)
    baseline = jnp.clip(baseline, 0, None)
    amplitude_1 = jnp.clip(amplitude_1, 0, None)
    amplitude_2 = jnp.clip(amplitude_2, 0, None)
    widths = [jnp.clip(w, 0.01, None) for w in (tuning_width_1_left, tuning_width_1_right)]
    widths += [jnp.clip(w, 0.01, None) for w in (tuning_width_2_left, tuning_width_2_right)]
    exponent_1 = jnp.clip(exponent_1, 0.1, 10)
    exponent_2 = jnp.clip(exponent_2, 0.1, 10)
    angle_offset_2 = jnp.clip(angle_offset_2, -jnp.pi, jnp.pi)
    second = (theta_pref + jnp.pi + angle_offset_2) % (2 * jnp.pi)
    return (
        baseline
        + _peak(theta, theta_pref, amplitude_1, widths[0], widths[1], exponent_1)
        + _peak(theta, second, amplitude_2, widths[2], widths[3], exponent_2)
    )


def antipodal_asymmetric_model(
    theta,
    theta_pref=0.0,
    baseline=0.0,
    amplitude_1=1.0,
    amplitude_2=0.0,
    tuning_width_1_left=1.0,
    tuning_width_1_right=1.0,
    tuning_width_2_left=1.0,
    tuning_width_2_right=1.0,
    exponent_1=2.0,
    exponent_2=2.0,
):
    """ "Gaussian+p+asym": the AI model with the second peak exactly opposite the first."""
    return ai_model(
        theta,
        theta_pref,
        baseline,
        amplitude_1,
        amplitude_2,
        tuning_width_1_left,
        tuning_width_1_right,
        tuning_width_2_left,
        tuning_width_2_right,
        exponent_1,
        exponent_2,
        0.0,
    )


def antipodal_symmetric_model(
    theta,
    theta_pref=0.0,
    baseline=0.0,
    amplitude_1=1.0,
    amplitude_2=0.0,
    tuning_width_1=1.0,
    tuning_width_2=1.0,
    exponent_1=2.0,
    exponent_2=2.0,
):
    """ "Gaussian+p": opposite peaks, each with one width and its own exponent."""
    return ai_model(
        theta,
        theta_pref,
        baseline,
        amplitude_1,
        amplitude_2,
        tuning_width_1,
        tuning_width_1,
        tuning_width_2,
        tuning_width_2,
        exponent_1,
        exponent_2,
        0.0,
    )


def ai_estimator(theta, spike_counts):
    """The paper's parameter estimator for the winning model (Supplementary Text 2.9)."""
    n_bins = 75
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    k = np.exp(-0.5 * (np.arange(-5, 6)) ** 2)
    k /= np.sum(k)
    sums_padded = np.pad(sums, (5, 5), mode="wrap")
    counts_padded = np.pad(counts, (5, 5), mode="wrap")
    tuning_curve = np.convolve(sums_padded, k, mode="valid") / (
        np.convolve(counts_padded, k, mode="valid") + 1e-8
    )
    baseline = np.min(tuning_curve)
    peak_1_idx = np.argmax(tuning_curve)
    theta_pref = peak_1_idx * (2 * np.pi / n_bins)
    amplitude_1 = tuning_curve[peak_1_idx] - baseline
    anti_pref_idx = (peak_1_idx + n_bins // 2) % n_bins
    amplitude_2 = max(0, tuning_curve[anti_pref_idx] - baseline)
    theta_pref_2_est = anti_pref_idx * (2 * np.pi / n_bins)
    angle_offset_2 = theta_pref_2_est - (theta_pref + np.pi) % (2 * np.pi)
    width = np.pi / 8
    exponent = 2.0
    return np.array(
        [
            theta_pref,
            baseline,
            amplitude_1,
            amplitude_2,
            width,
            width,
            width,
            width,
            exponent,
            exponent,
            angle_offset_2,
        ]
    )


def antipodal_asymmetric_estimator(theta, spike_counts):
    return ai_estimator(theta, spike_counts)[:10]


def antipodal_symmetric_estimator(theta, spike_counts):
    return ai_estimator(theta, spike_counts)[[0, 1, 2, 3, 4, 6, 8, 9]]


def ai_wave_model(
    theta,
    theta_pref=0.0,
    baseline=0.0,
    amplitude_1=1.0,
    amplitude_2=0.0,
    tuning_width_1_left=1.0,
    tuning_width_1_right=1.0,
    tuning_width_2_left=1.0,
    tuning_width_2_right=1.0,
    exponent_1=2.0,
    exponent_2=2.0,
    angle_offset_2=0.0,
    wave_cos=0.0,
    wave_sin=0.0,
):
    """The AI model plus one sinusoid per stimulus cycle, the extra wave in this run's winner."""
    peaks = ai_model(
        theta,
        theta_pref,
        baseline,
        amplitude_1,
        amplitude_2,
        tuning_width_1_left,
        tuning_width_1_right,
        tuning_width_2_left,
        tuning_width_2_right,
        exponent_1,
        exponent_2,
        angle_offset_2,
    )
    return peaks + wave_cos * jnp.cos(theta) + wave_sin * jnp.sin(theta)


def ai_wave_estimator(theta, spike_counts):
    return np.concatenate([ai_estimator(theta, spike_counts), [0.0, 0.0]])


LADDER = {
    "Gaussian+p": (antipodal_symmetric_model, antipodal_symmetric_estimator),
    "Gaussian+p+asym": (antipodal_asymmetric_model, antipodal_asymmetric_estimator),
    "AI model": (ai_model, ai_estimator),
}


def parameter_names(model):
    return [name for name in inspect.signature(model).parameters if name != "theta"]

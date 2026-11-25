"""Measure a fitted equation against held-out trials."""

import math
from functools import partial

import jax.numpy as jnp
import numpy as np

from edgar_study.frontier.images import ParentFit
from edgar_study.frontier.protocol import predict, solve_model, split_trials
from edgar_study.frontier.seed import BASELINE_ID


def double_gaussian_parent(up):
    """The starting equation, fitted under the same protocol as every candidate."""
    seeds = up["seed_programs"]
    model, estimator = seeds.neuron_model_2_jax, seeds.parameter_estimator_2
    return ParentFit(BASELINE_ID, model, estimator, partial(solve_model, model, estimator))


def measure(model, solve, cohort, max_iter):
    """Fit with `solve` and report whole-curve and per-30-degree-bin held-out error."""
    x, _ = cohort
    y = cohort[1]
    score, mse, count, params = solve(cohort, max_iter)
    if not math.isfinite(score):
        raise ValueError("Nonfinite fit.")
    train, held = split_trials(x)
    angles = x[:, held]
    errors = np.asarray((predict(model, angles, params) - y[:, held]) ** 2)
    training_mse = float(jnp.mean((predict(model, x[:, train], params) - y[:, train]) ** 2))
    if not math.isfinite(training_mse) or not np.isfinite(errors).all():
        raise ValueError("Nonfinite predictions.")
    bins = np.floor((np.rad2deg(np.asarray(angles)) % 360) / 30).astype(int)
    return {
        "score": score,
        "mse": mse,
        "training_mse": training_mse,
        "parameters": count,
        "angle_mse": [float(errors[bins == i].mean()) if (bins == i).any() else None for i in range(12)],
    }

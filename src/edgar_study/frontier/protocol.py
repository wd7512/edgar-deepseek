"""One fitting protocol for every equation, so scores are comparable.

Each model brings its own start (its estimator, run on training trials only). Every model then gets the
same budget: the same number of starts, the same seeded jitter and the same iteration cap. Parameters
are put in canonical order first, so equivalent equations get identical fits. Held-out trials are never
used to choose among starts.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import TRIAL_SEED
from edgar_study.upstream import MSE_SCALE, PENALTY

STARTS = 4


class Fit(NamedTuple):
    score: float
    mse: float
    training_mse: float
    count: int
    params: np.ndarray


def split_trials(x):
    """Return the training and held-out trial indices shared by every fit."""
    order = jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(x.shape[1]))
    half = x.shape[1] // 2
    return order[:half], order[half:]


def predict(model, x, params):
    return jax.vmap(model)(x, *params.T)


def fit_model(model, estimator, cohort, max_iter, *, starts=STARTS, seed=0):
    """Fit `model` to every cell from its estimator's start plus seeded jitter; score on held-out trials."""
    x, y = cohort
    train, held = split_trials(x)
    xt, yt = np.asarray(x)[:, train], np.asarray(y)[:, train]
    start = np.stack([np.asarray(estimator(xt[i], yt[i]), dtype=np.float32) for i in range(len(xt))])
    fitted = fitting.fit_multistart(
        model,
        start,
        xt,
        yt,
        starts=starts,
        seed=seed,
        max_iterations=min(max_iter, fitting.LM_ITERATIONS),
    )
    params = jnp.asarray(fitted.params)
    count = int(params.shape[1])

    def error(angles, responses):
        return float(jnp.mean((predict(model, jnp.asarray(angles), params) - jnp.asarray(responses)) ** 2))

    mse = error(np.asarray(x)[:, held], np.asarray(y)[:, held])
    training_mse = error(xt, yt)
    if not (math.isfinite(mse) and math.isfinite(training_mse)):
        raise ValueError("Nonfinite fit.")
    return Fit(mse * MSE_SCALE + count * PENALTY, mse, training_mse, count, np.asarray(params))


def canonical(equation):
    """Return the equation's model and estimator with parameters in canonical order."""
    names = tuple(equation.initial)
    position = [equation.canonical_names.index(name) for name in names]

    def model(theta, *params):
        return equation.model(theta, *(params[i] for i in position))

    def estimator(_theta, _response):
        return tuple(float(equation.initial[name]) for name in equation.canonical_names)

    return model, estimator


def fit_equation(equation, cohort, max_iter, **options):
    """Fit a compiled equation under the shared protocol; parameters come back in its declared order."""
    model, estimator = canonical(equation)
    result = fit_model(model, estimator, cohort, max_iter, **options)
    names = tuple(equation.initial)
    declared = [equation.canonical_names.index(name) for name in names]
    return result._replace(params=result.params[:, declared])


def solve(equation, cohort, max_iter, **options):
    """Fit and return `(score, mse, parameter count, parameters)` as the runner and images expect."""
    result = fit_equation(equation, cohort, max_iter, **options)
    return result.score, result.mse, result.count, result.params


def solve_model(model, estimator, cohort, max_iter, **options):
    """`solve` for a model with its own estimator, such as the double Gaussian."""
    result = fit_model(model, estimator, cohort, max_iter, **options)
    return result.score, result.mse, result.count, result.params

"""One fitting protocol for every equation, so scores are comparable.

Each model brings its own start (its estimator, run on training trials only). Every model then gets the
same budget: the same number of starts, the same seeded jitter and the same iteration cap. Parameters
are put in canonical order first, so equivalent equations get identical fits. Held-out trials are never
used to choose among starts.
"""

import hashlib
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import TRIAL_SEED
from edgar_study.upstream import MSE_SCALE, PENALTY

STARTS = 5
JITTER = fitting.MULTISTART_SCALE


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


def jittered_starts(start, xt, yt, starts, seed, scale):
    """The estimator start plus `starts - 1` jittered copies, each cell's jitter drawn from its own data.

    A cell's starts then depend only on that cell and the seed, never on which other cells are fitted
    with it or in what order.
    """
    n = start.shape[1]
    noise = np.stack(
        [
            np.random.default_rng(
                [seed, int.from_bytes(hashlib.sha256(xt[i].tobytes() + yt[i].tobytes()).digest()[:8], "big")]
            ).standard_normal((starts - 1, n))
            for i in range(len(start))
        ],
        axis=1,
    )
    jittered = start[None] + scale * noise * (np.abs(start)[None] + 1)
    return np.concatenate([start[None], jittered.astype(start.dtype)])


def fit_model(model, estimator, cohort, max_iter, *, starts=STARTS, seed=0, warm=None):
    """Fit `model` to every cell from its estimator's start plus seeded jitter; score on held-out trials.

    `warm` (cells, parameters), when given, replaces the last jittered start, so the other starts are the
    same ones a cold fit with one fewer start would use. It is ignored when only one start is allowed.
    """
    x, y = cohort
    train, held = split_trials(x)
    xt, yt = np.asarray(x)[:, train], np.asarray(y)[:, train]
    start = np.stack([np.asarray(estimator(xt[i], yt[i]), dtype=np.float32) for i in range(len(xt))])
    candidates = jittered_starts(start, xt, yt, starts, seed, JITTER)
    if warm is not None and starts > 1:
        warm = np.asarray(warm, dtype=candidates.dtype)
        if warm.shape != start.shape:
            raise ValueError(f"Warm start has shape {warm.shape}, expected {start.shape}.")
        candidates[-1] = warm
    fitted = fitting.fit_from_starts(
        model, candidates, xt, yt, max_iterations=min(max_iter, fitting.LM_ITERATIONS)
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
    names = tuple(equation.initial)
    warm = options.pop("warm", None)
    if warm is not None:
        warm = np.asarray(warm)[:, [names.index(name) for name in equation.canonical_names]]
    result = fit_model(model, estimator, cohort, max_iter, warm=warm, **options)
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

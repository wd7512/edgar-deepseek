"""Per-neuron comparison of two winners on the locked-test cells, for the no-LLM null hypothesis."""

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study.upstream import MSE_SCALE


def cell_losses(model, params, x, y, score_columns):
    """Each cell's held-out loss, 10 × mean squared error on the scoring trials, as upstream averages it."""
    columns = jnp.asarray(score_columns)
    xs, ys = jnp.asarray(x)[:, columns], jnp.asarray(y)[:, columns]

    def one(p, xc, yc):
        return MSE_SCALE * jnp.mean((model(xc, *p) - yc) ** 2)

    return np.asarray(jax.vmap(one)(jnp.asarray(params), xs, ys), dtype=float)


def paired_bootstrap(a, b, penalty_delta=0.0, n=10_000, seed=0):
    """Mean of a − b over cells plus `penalty_delta`, a 95% bootstrap interval over cells, and the
    share of resamples in which `a` is not better (the one-sided p for "a beats b")."""
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), (n, len(difference)))
    means = difference[draws].mean(axis=1) + penalty_delta
    return {
        "delta": float(difference.mean() + penalty_delta),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "p_not_better": float(np.mean(means >= 0)),
        "cells": int(len(difference)),
    }

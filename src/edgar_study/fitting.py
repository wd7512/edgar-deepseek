"""Faster per-cell parameter fitting for upstream's `objective`.

Upstream fits each program with 1,000 Adam steps driven from a Python loop, syncing with the
host after every step. That is reasonable on a GPU but slow on CPU. This module keeps
upstream's checks, data split and scoring by calling upstream's own `objective` with
`max_iter=0`, and swaps only the fitting loop:

* `upstream`: upstream's loop, untouched.
* `adam`:     the same Adam schedule and best-parameter bookkeeping, compiled into one XLA loop.
* `lm`:       batched Levenberg–Marquardt. Each cell is an independent least-squares problem with a
              handful of parameters, so second-order steps converge in tens of iterations.
* `lm-bounded`: Levenberg–Marquardt that moves each parameter at most `LM_BOUNDED_MAX_STEP` per
              iteration and stops each cell once an accepted step lowers its cost by less than
              `LM_BOUNDED_TOLERANCE` (relative), or after `lm_iterations`. Unbounded steps can throw a
              parameter past a program's clip, where its gradient is zero and it stays stuck; the
              bound keeps fits near the path Adam takes. A step that hits the bound raises the
              damping, so ill-conditioned starts turn towards scaled gradient descent.

Both fitters split the cells across every JAX device. On a CPU, `edgar_study/__init__.py` exposes
one device per core before JAX starts, because XLA otherwise runs this elementwise work on about
one thread.

`auto` picks `lm-bounded`: about 8× faster than upstream's loop over the paper's curves and
upstream's seeds, with scores within 0.7% (see `scripts/fit_timings/results.md`). It fits each cell
to a tolerance instead of upstream's fixed 1,000 steps, so scores are not identical; pass
`upstream` or `adam` to reproduce upstream's scores.
"""

import functools
import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

FITTERS = ("auto", "upstream", "adam", "lm", "lm-bounded")
LM_ITERATIONS = 100
LM_BOUNDED_MAX_STEP = 0.25
LM_BOUNDED_TOLERANCE = 1e-4
MAX_DAMPING = 1e7

log = logging.getLogger(__name__)


class BoundedFit(NamedTuple):
    params: np.ndarray
    iterations: int
    converged: int


def resolve(fitter):
    return "lm-bounded" if fitter == "auto" else fitter


def install(hypothesis_engine, fitter, lm_iterations=LM_ITERATIONS):
    """Replace hypothesis_engine.objective with a version that uses `fitter`. Returns the resolved name."""
    fitter = resolve(fitter)
    if fitter != "upstream":
        hypothesis_engine.objective = make_objective(hypothesis_engine.objective, fitter, lm_iterations)
    return fitter


def make_objective(upstream_objective, fitter, lm_iterations=LM_ITERATIONS):
    if fitter not in FITTERS[2:]:
        raise ValueError(f"Unknown fitter {fitter!r}.")

    @functools.wraps(upstream_objective)
    def objective(
        neuron_model,
        param_estimator,
        loss_func,
        x,
        y,
        param_penalty_weight=0.1,
        fit_params=True,
        random_seed=0,
        FAILED_PROGRAM_COST=jnp.inf,
        tol=1e-2,
        max_iter=1_000,
        use_param_estimator=True,
    ):
        # Upstream runs the estimator, all validity checks and the initial scores; no fitting.
        result = upstream_objective(
            neuron_model,
            param_estimator,
            loss_func,
            x,
            y,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            random_seed=random_seed,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            tol=tol,
            max_iter=0,
            use_param_estimator=use_param_estimator,
        )
        initial_loss, initial_params, unfitted_loss, params = result
        if not fit_params or params.shape[1] == 0 or unfitted_loss == FAILED_PROGRAM_COST:
            return result

        # The same trial split as upstream's objective.
        order = jax.random.permutation(jax.random.PRNGKey(random_seed), jnp.arange(y.shape[1]))
        train, test = order[: y.shape[1] // 2], order[y.shape[1] // 2 :]
        x_train, y_train, x_test, y_test = x[:, train], y[:, train], x[:, test], y[:, test]

        if fitter == "lm" and loss_func.__name__ == "quadratic_loss":
            fitted = fit_lm(neuron_model, params, x_train, y_train, lm_iterations)
        elif fitter == "lm-bounded" and loss_func.__name__ == "quadratic_loss":
            fit = fit_lm_bounded(neuron_model, params, x_train, y_train, max_iterations=lm_iterations)
            cells = params.shape[0]
            log.info(
                "lm-bounded stopped after %d iterations with %d of %d cells converged",
                fit.iterations,
                fit.converged,
                cells,
                extra={"iterations": fit.iterations, "converged": fit.converged, "cells": cells},
            )
            fitted = fit.params
        else:
            fitted = fit_adam(neuron_model, loss_func, params, x_train, y_train, max_iter)

        per_cell = _per_cell_loss(neuron_model, loss_func)
        final = jnp.nanmean(per_cell(fitted, x_test, y_test)) + param_penalty_weight * fitted.shape[1]
        final = float(
            jnp.nan_to_num(
                final, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST
            )
        )
        return initial_loss, initial_params, final, fitted

    return objective


def _shard(*arrays):
    """Pad the cell axis to a multiple of the device count and split it across devices.
    Returns the sharded arrays and a mask that is 1 for real cells and 0 for padding."""
    devices = jax.devices()
    n_cells = arrays[0].shape[0]
    padded = -(-n_cells // len(devices)) * len(devices)
    index = np.concatenate([np.arange(n_cells), np.zeros(padded - n_cells, dtype=int)])
    sharding = NamedSharding(Mesh(np.array(devices), ("cells",)), PartitionSpec("cells"))
    mask = jax.device_put(jnp.asarray(np.arange(padded) < n_cells, jnp.float32), sharding)
    return [jax.device_put(jnp.asarray(a)[index], sharding) for a in arrays], mask, n_cells


def _per_cell_loss(model, loss_func):
    return jax.vmap(lambda p, xc, yc: jnp.mean(loss_func(model(xc, *p), yc)))


def fit_adam(model, loss_func, params, x, y, max_iter, learning_rate=3e-3):
    """Upstream's Adam loop in one compiled while_loop, with identical bookkeeping:
    stop at the first non-finite loss or parameter, keep the parameters after the best step."""
    import optax

    (params, x, y), mask, n_cells = _shard(params, x, y)
    per_cell = _per_cell_loss(model, loss_func)

    def loss(p):
        return jnp.sum(per_cell(p, x, y) * mask) / n_cells

    value_and_grad = jax.value_and_grad(loss)
    opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)

    def cond(state):
        step, _, _, _, _, stop = state
        return (step < max_iter) & ~stop

    def body(state):
        step, p, opt_state, best_loss, best_p, _ = state
        value, grad = value_and_grad(p)
        updates, opt_state = opt.update(grad, opt_state, p)
        p = optax.apply_updates(p, updates)
        stop = ~jnp.isfinite(value) | ~jnp.all(jnp.isfinite(p))  # ty: ignore[invalid-argument-type]
        better = ~stop & (value < best_loss)
        return (
            step + 1,
            p,
            opt_state,
            jnp.where(better, value, best_loss),
            jnp.where(better, p, best_p),  # ty: ignore[no-matching-overload]
            stop,
        )

    @jax.jit
    def run(p):
        return jax.lax.while_loop(cond, body, (0, p, opt.init(p), loss(p), p, False))[4]

    return jax.device_get(run(params))[:n_cells]


def _least_squares(model, x, y):
    """Per-cell cost and damped Gauss–Newton step for `model`'s squared error on (x, y)."""

    def residual(p, xc, yc):
        return model(xc, *p) - yc

    residuals = jax.vmap(residual)
    jacobians = jax.vmap(jax.jacfwd(residual))

    def cost(p):
        r = residuals(p, x, y)
        c = jnp.sum(r * r, axis=1)
        return jnp.where(jnp.isfinite(c), c, jnp.inf)

    def step(p, damping):
        r, J = residuals(p, x, y), jacobians(p, x, y)
        gradient = jnp.einsum("ctk,ct->ck", J, r)
        hessian = jnp.einsum("ctk,ctl->ckl", J, J)
        diagonal = jnp.diagonal(hessian, axis1=1, axis2=2)
        scale = jnp.maximum(diagonal, 1e-9 * jnp.max(diagonal, axis=1, keepdims=True) + 1e-12)
        damped = hessian + jax.vmap(jnp.diag)(damping[:, None] * scale)
        s = jnp.linalg.solve(damped, -gradient[..., None])[..., 0]
        return jnp.where(jnp.isfinite(s), s, 0.0)

    return cost, step


def _update_damping(damping, accept, capped=False):
    """Lower damping after a full accepted step; raise it after a rejected or capped one, so a step
    that keeps hitting the cap turns towards scaled gradient descent instead of a truncated
    Gauss–Newton step."""
    lowered = jnp.where(capped, damping * 2, damping / 3)
    return jnp.clip(jnp.where(accept, lowered, damping * 4), 1e-7, MAX_DAMPING)


def fit_lm(model, params, x, y, iterations=LM_ITERATIONS):
    """Batched Levenberg–Marquardt on each cell's squared error. Steps are accepted only when they
    lower that cell's error, so every cell ends at least as well fitted as its starting point."""
    (params, x, y), _, n_cells = _shard(params, x, y)
    cost, lm_step = _least_squares(model, x, y)

    def body(_, state):
        p, damping, c = state
        candidate = p + lm_step(p, damping)
        new_cost = cost(candidate)
        accept = new_cost < c
        return (
            jnp.where(accept[:, None], candidate, p),
            _update_damping(damping, accept),
            jnp.where(accept, new_cost, c),
        )

    @jax.jit
    def run(p):
        start = (p, jnp.full(p.shape[0], 1e-3, p.dtype), cost(p))
        return jax.lax.fori_loop(0, iterations, body, start)[0]

    return jax.device_get(run(params))[:n_cells]


def fit_lm_bounded(
    model,
    params,
    x,
    y,
    max_step=LM_BOUNDED_MAX_STEP,
    tolerance=LM_BOUNDED_TOLERANCE,
    max_iterations=LM_ITERATIONS,
):
    """Levenberg–Marquardt with each parameter's move capped at `max_step` per iteration.

    A cell stops once an accepted step lowers its squared error by less than `tolerance` of the
    current error, or once its damping saturates; the loop ends when every cell has stopped or after
    `max_iterations`. Returns the fitted parameters, the iterations run and how many cells stopped
    by tolerance rather than by the iteration cap.
    """
    (params, x, y), _, n_cells = _shard(params, x, y)
    cost, lm_step = _least_squares(model, x, y)

    def cond(state):
        i, _, _, _, done = state
        return (i < max_iterations) & ~jnp.all(done)

    def body(state):
        i, p, damping, c, done = state
        step = lm_step(p, damping)
        biggest = jnp.max(jnp.abs(step), axis=1, keepdims=True)
        capped = biggest[:, 0] > max_step
        candidate = p + step * jnp.minimum(1.0, max_step / jnp.maximum(biggest, 1e-30))
        new_cost = cost(candidate)
        accept = ~done & (new_cost < c)
        converged = (accept & (c - new_cost <= tolerance * c)) | (damping >= MAX_DAMPING)
        return (
            i + 1,
            jnp.where(accept[:, None], candidate, p),
            jnp.where(done, damping, _update_damping(damping, accept, capped)),
            jnp.where(accept, new_cost, c),
            done | converged,
        )

    @jax.jit
    def run(p):
        start = (0, p, jnp.full(p.shape[0], 1e-3, p.dtype), cost(p), jnp.zeros(p.shape[0], bool))
        i, p, _, _, done = jax.lax.while_loop(cond, body, start)
        return p, i, done

    fitted, iterations, done = jax.device_get(run(params))
    return BoundedFit(fitted[:n_cells], int(iterations), int(np.sum(done[:n_cells])))


def benchmark_report(rows):
    """Format benchmark rows as a Markdown table."""
    lines = ["| Program | Fitter | Seconds | Train loss | Score |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['program']} | {row['fitter']} | {row['seconds']:.1f} | "
            f"{row['train_loss']:.4f} | {row['score']:.4f} |"
        )
    return "\n".join(lines)


def train_loss(model, loss_func, params, x, y, random_seed=0):
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(random_seed), jnp.arange(y.shape[1])))
    train = order[: y.shape[1] // 2]
    return float(jnp.nanmean(_per_cell_loss(model, loss_func)(params, x[:, train], y[:, train])))

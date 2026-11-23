"""Levenberg-Marquardt in JAX on the CPU, compiled once per program so timings are fit-only."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from common import (
    LM_MAX_ITERATIONS,
    LM_MAX_STEP,
    LM_TOLERANCE,
    MAX_DAMPING,
    load,
    summarise,
    time_call,
)

from edgar_study import fitting
from edgar_study.curves import CURVES


def make_lm(curve, p0, x, y, bounded, max_iterations):
    (p0_sharded, x_sharded, y_sharded), _, _ = fitting._shard(p0, x, y)
    cost, lm_step = fitting._least_squares(curve.model, x_sharded, y_sharded)

    def body(state):
        i, p, damping, c, done = state
        step = lm_step(p, damping)
        capped = jnp.zeros(p.shape[0], bool)
        if bounded:
            biggest = jnp.max(jnp.abs(step), axis=1, keepdims=True)
            capped = biggest[:, 0] > LM_MAX_STEP
            step = step * jnp.minimum(1.0, LM_MAX_STEP / jnp.maximum(biggest, 1e-30))
        candidate = p + step
        new_cost = cost(candidate)
        accept = ~done & (new_cost < c)
        if bounded:
            converged = (accept & (c - new_cost <= LM_TOLERANCE * c)) | (damping >= MAX_DAMPING)
            done = done | converged
        return (
            i + 1,
            jnp.where(accept[:, None], candidate, p),
            fitting._update_damping(damping, accept, capped),
            jnp.where(accept, new_cost, c),
            done,
        )

    def cond(state):
        if not bounded:
            return state[0] < max_iterations
        return (state[0] < max_iterations) & ~jnp.all(state[4])

    @jax.jit
    def run(p):
        start = (
            jnp.asarray(0),
            p,
            jnp.full(p.shape[0], 1e-3, p.dtype),
            cost(p),
            jnp.zeros(p.shape[0], bool),
        )
        i, p, _, _, done = jax.lax.while_loop(cond, body, start)
        return p, i, done

    def call():
        return jax.block_until_ready(run(p0_sharded))

    return call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=LM_MAX_ITERATIONS)
    args = parser.parse_args(argv)
    for name in args.programs:
        data = load(name)
        curve = next(c for c in CURVES if c.name == name)
        x = jnp.asarray(data["x_tr"])
        y = jnp.asarray(data["y_tr"])
        p0 = jnp.asarray(data["p0"])
        for bounded in (False, True):
            call = make_lm(curve, p0, x, y, bounded, args.iterations)
            seconds, (params, iterations, done) = time_call(call, args.repeats)
            params = np.asarray(jax.device_get(params))[: data["x_tr"].shape[0]]
            summarise(
                data,
                params,
                seconds,
                int(iterations),
                "jax",
                "cpu",
                "lm-bounded" if bounded else "lm",
                {"converged": int(np.sum(np.asarray(done)[: data["x_tr"].shape[0]]))},
            )


if __name__ == "__main__":
    main()

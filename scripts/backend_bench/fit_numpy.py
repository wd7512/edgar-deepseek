"""Batched Levenberg-Marquardt in numpy, the reference runtime for the benchmark."""

import argparse

import numpy as np
from common import (
    LM_MAX_ITERATIONS,
    LM_MAX_STEP,
    LM_TOLERANCE,
    MAX_DAMPING,
    jacobian,
    load,
    predict,
    summarise,
    time_call,
)


def fit(program, x, y, p0, bounded, max_iterations=LM_MAX_ITERATIONS):
    params = p0.astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)
    damping = np.full(params.shape[0], 1e-3, dtype=np.float32)
    cost = np.sum((predict(program, x, params) - y) ** 2, axis=1, dtype=np.float32)
    done = np.zeros(params.shape[0], bool)
    iterations = 0
    for iteration in range(max_iterations):
        if bounded and done.all():
            break
        with np.errstate(all="ignore"):
            residuals = predict(program, x, params) - y
            jac = jacobian(program, x, params)
            gradient = np.matmul(jac.transpose(0, 2, 1), residuals[..., None])[..., 0]
            hessian = np.matmul(jac.transpose(0, 2, 1), jac)
            diagonal = np.diagonal(hessian, axis1=1, axis2=2)
            scale = np.maximum(diagonal, 1e-9 * np.max(diagonal, axis=1, keepdims=True) + 1e-12)
            damped = hessian + diagonal_embedding(damping[:, None] * scale)
            step = np.linalg.solve(damped, -gradient[..., None])[..., 0]
            step = np.where(np.isfinite(step), step, 0.0)
            capped = np.zeros(params.shape[0], bool)
            if bounded:
                biggest = np.max(np.abs(step), axis=1, keepdims=True)
                capped = biggest[:, 0] > LM_MAX_STEP
                step = step * np.minimum(1.0, LM_MAX_STEP / np.maximum(biggest, 1e-30))
            candidate = params + step
            new_cost = np.sum((predict(program, x, candidate) - y) ** 2, axis=1, dtype=np.float32)
            accept = ~done & (new_cost < cost)
            if bounded:
                converged = (accept & (cost - new_cost <= LM_TOLERANCE * cost)) | (damping >= MAX_DAMPING)
                raised = np.where(capped, damping * 2, damping / 3)
                update = np.clip(np.where(accept, raised, damping * 4), 1e-7, MAX_DAMPING)
                damping = np.where(done, damping, update)
                done = done | converged
            else:
                damping = np.clip(np.where(accept, damping / 3, damping * 4), 1e-7, MAX_DAMPING)
            params = np.where(accept[:, None], candidate, params)
            cost = np.where(accept, new_cost, cost)
            iterations = iteration + 1
    return params, iterations, int(done.sum())


def diagonal_embedding(d):
    out = np.zeros((d.shape[0], d.shape[1], d.shape[1]))
    index = np.arange(d.shape[1])
    out[:, index, index] = d
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=LM_MAX_ITERATIONS)
    args = parser.parse_args(argv)
    for name in args.programs:
        data = load(name)
        for bounded in (False, True):
            seconds, (params, iterations, converged) = time_call(
                lambda bounded=bounded, name=name, data=data: fit(
                    name,
                    data["x_tr"],
                    data["y_tr"],
                    data["p0"],
                    bounded,
                    args.iterations,
                ),
                args.repeats,
            )
            summarise(
                data,
                params,
                seconds,
                iterations,
                "numpy",
                "cpu",
                "lm-bounded" if bounded else "lm",
                {"converged": converged},
            )


if __name__ == "__main__":
    main()

"""Levenberg-Marquardt and Adam in MLX on the Apple GPU.

MLX has no GPU linear solve in this version, so each 5x5 or 11x11 system is solved on the CPU
stream (unified memory, no copy); model and Jacobian evaluation stay on the GPU.
"""

import argparse
import math

import mlx.core as mx
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


def peak(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = mx.arctan2(mx.sin(x - phi), mx.cos(x - phi))
    sigma = mx.where(
        d >= 0,
        mx.clip(sigma_plus, 0.05, 3.0),
        mx.clip(sigma_minus, 0.05, 3.0),
    )
    p = mx.clip(exponent, 0.5, 8.0) if isinstance(exponent, mx.array) else min(max(exponent, 0.5), 8.0)
    u = mx.abs(d) / sigma
    return mx.maximum(amplitude, 0.0) * mx.exp(-0.5 * u**p)


def model(program, x, p):
    if program == "winner":
        phi1, phi2, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        a1, a2 = p[:, 3:4], p[:, 4:5]
        s1p, s1m, s2p, s2m = p[:, 5:6], p[:, 6:7], p[:, 7:8], p[:, 8:9]
        p1, p2 = p[:, 9:10], p[:, 10:11]
        return mx.maximum(b, 0.0) + peak(x, phi1, a1, s1p, s1m, p1) + peak(x, phi2, a2, s2p, s2m, p2)
    phi, b, a1, a2, s = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4], p[:, 4:5]
    return mx.maximum(b, 0.0) + peak(x, phi, a1, s, s, 2.0) + peak(x, phi + 3.141592653589793, a2, s, s, 2.0)


def peak_parts(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = mx.arctan2(mx.sin(x - phi), mx.cos(x - phi))
    sigma = mx.where(
        d >= 0,
        mx.clip(sigma_plus, 0.05, 3.0),
        mx.clip(sigma_minus, 0.05, 3.0),
    )
    p = mx.clip(exponent, 0.5, 8.0) if isinstance(exponent, mx.array) else min(max(exponent, 0.5), 8.0)
    absolute = mx.abs(d)
    u = absolute / sigma
    q = u**p
    e = mx.exp(-0.5 * q)
    gain = mx.maximum(amplitude, 0.0)
    dphi = 0.5 * gain * e * p * mx.sign(d) * q / (absolute + 1e-12)
    dsigma = 0.5 * gain * e * q * p / sigma
    ds_plus = mx.where(d >= 0, dsigma, 0.0)
    ds_minus = mx.where(d < 0, dsigma, 0.0)
    dp = -0.5 * gain * e * q * mx.log(u + 1e-12)
    active = mx.where(amplitude > 0, 1.0, 0.0)
    return gain * e, dphi, e * active, ds_plus, ds_minus, dp


def jacobian(program, x, p):
    if program == "winner":
        first = peak_parts(x, p[:, 0:1], p[:, 3:4], p[:, 5:6], p[:, 6:7], p[:, 9:10])
        second = peak_parts(x, p[:, 1:2], p[:, 4:5], p[:, 7:8], p[:, 8:9], p[:, 10:11])
        columns = [
            first[1],
            second[1],
            mx.where(p[:, 2:3] > 0, 1.0, 0.0),
            first[2],
            second[2],
            first[3],
            first[4],
            second[3],
            second[4],
            first[5],
            second[5],
        ]
    else:
        first = peak_parts(x, p[:, 0:1], p[:, 2:3], p[:, 4:5], p[:, 4:5], 2.0)
        second = peak_parts(x, p[:, 0:1] + 3.141592653589793, p[:, 3:4], p[:, 4:5], p[:, 4:5], 2.0)
        columns = [
            first[1] + second[1],
            mx.where(p[:, 1:2] > 0, 1.0, 0.0),
            first[2],
            second[2],
            first[3] + first[4] + second[3] + second[4],
        ]
    return mx.stack([mx.broadcast_to(column, x.shape) for column in columns], axis=2)


def fit_lm(program, x, y, p0, bounded, max_iterations):
    p = p0
    damping = mx.full((p.shape[0],), 1e-3, dtype=mx.float32)
    cost = mx.sum((model(program, x, p) - y) ** 2, axis=1)
    done = mx.zeros((p.shape[0],), dtype=mx.bool_)
    iterations = 0
    for iteration in range(max_iterations):
        mx.eval(cost, done)
        if bounded and bool(mx.all(done)):
            break
        residual = model(program, x, p) - y
        jac = jacobian(program, x, p)
        gradient = mx.einsum("ctk,ct->ck", jac, residual)
        hessian = mx.einsum("ctk,ctl->ckl", jac, jac)
        diagonal = mx.diagonal(hessian, axis1=1, axis2=2)
        scale = mx.maximum(diagonal, 1e-9 * mx.max(diagonal, axis=1, keepdims=True) + 1e-12)
        damped = hessian + (damping[:, None] * scale)[:, :, None] * mx.eye(p.shape[1], dtype=mx.float32)
        with mx.stream(mx.cpu):
            step = mx.linalg.solve(damped, -gradient[..., None])[..., 0]
        step = mx.where(mx.isfinite(step), step, 0.0)
        capped = mx.zeros((p.shape[0],), dtype=mx.bool_)
        if bounded:
            biggest = mx.max(mx.abs(step), axis=1, keepdims=True)
            capped = biggest[:, 0] > LM_MAX_STEP
            step = step * mx.minimum(1.0, LM_MAX_STEP / mx.maximum(biggest, 1e-30))
        candidate = p + step
        new_cost = mx.sum((model(program, x, candidate) - y) ** 2, axis=1)
        accept = mx.logical_and(mx.logical_not(done), new_cost < cost)
        if bounded:
            converged = mx.logical_or(
                mx.logical_and(accept, cost - new_cost <= LM_TOLERANCE * cost),
                damping >= MAX_DAMPING,
            )
            raised = mx.where(capped, damping * 2, damping / 3)
            update = mx.clip(mx.where(accept, raised, damping * 4), 1e-7, MAX_DAMPING)
            damping = mx.where(done, damping, update)
            done = mx.logical_or(done, converged)
        else:
            damping = mx.clip(mx.where(accept, damping / 3, damping * 4), 1e-7, MAX_DAMPING)
        p = mx.where(accept[:, None], candidate, p)
        cost = mx.where(accept, new_cost, cost)
        mx.eval(p, damping, cost, done)
        iterations = iteration + 1
    return p, iterations, int(mx.sum(done))


def fit_adam(program, x, y, p0, steps, learning_rate=3e-3):
    p = p0
    m = mx.zeros_like(p)
    v = mx.zeros_like(p)
    best = p
    best_loss = math.inf
    done = 0
    for step in range(1, steps + 1):
        residual = model(program, x, p) - y
        loss = mx.mean(residual * residual)
        gradient = 2 * mx.einsum("ctk,ct->ck", jacobian(program, x, p), residual) / residual.size
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        m_hat = m / (1 - 0.9**step)
        v_hat = v / (1 - 0.999**step)
        p = p - learning_rate * m_hat / (mx.sqrt(v_hat) + 1e-8)
        mx.eval(p, loss)
        value = float(loss)
        if not math.isfinite(value) or not bool(mx.all(mx.isfinite(p))):
            break
        done = step
        if value < best_loss:
            best_loss = value
            best = p
    return best, done, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--algorithms", nargs="+", default=["lm", "lm-bounded", "adam"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=LM_MAX_ITERATIONS)
    parser.add_argument("--adam-steps", type=int, default=1000)
    args = parser.parse_args(argv)
    for name in args.programs:
        data = load(name)
        x = mx.array(data["x_tr"])
        y = mx.array(data["y_tr"])
        p0 = mx.array(data["p0"])
        for algorithm in args.algorithms:
            if algorithm == "adam":
                seconds, (params, iterations, converged) = time_call(
                    lambda name=name, x=x, y=y, p0=p0: fit_adam(name, x, y, p0, args.adam_steps), args.repeats
                )
                mx.eval(params)
            else:
                bounded = algorithm == "lm-bounded"
                seconds, (params, iterations, converged) = time_call(
                    lambda bounded=bounded, name=name, x=x, y=y, p0=p0: fit_lm(
                        name, x, y, p0, bounded, args.iterations
                    ),
                    args.repeats,
                )
            summarise(
                data,
                np.asarray(params),
                seconds,
                iterations,
                "mlx",
                "gpu",
                algorithm,
                {"converged": int(converged)},
            )


if __name__ == "__main__":
    main()

"""Levenberg-Marquardt and Adam in TensorFlow (graph mode, CPU)."""

import argparse

import numpy as np
import tensorflow as tf
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
    d = tf.atan2(tf.sin(x - phi), tf.cos(x - phi))
    sigma = tf.where(
        d >= 0,
        tf.clip_by_value(sigma_plus, 0.05, 3.0),
        tf.clip_by_value(sigma_minus, 0.05, 3.0),
    )
    p = tf.clip_by_value(exponent, 0.5, 8.0) if tf.is_tensor(exponent) else min(max(exponent, 0.5), 8.0)
    u = tf.abs(d) / sigma
    return tf.maximum(amplitude, 0.0) * tf.exp(-0.5 * tf.pow(u, p))


def make_model(program):
    def model(x, p):
        if program == "winner":
            phi1, phi2, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
            a1, a2 = p[:, 3:4], p[:, 4:5]
            s1p, s1m, s2p, s2m = p[:, 5:6], p[:, 6:7], p[:, 7:8], p[:, 8:9]
            p1, p2 = p[:, 9:10], p[:, 10:11]
            return tf.maximum(b, 0.0) + peak(x, phi1, a1, s1p, s1m, p1) + peak(x, phi2, a2, s2p, s2m, p2)
        phi, b, a1, a2, s = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4], p[:, 4:5]
        return tf.maximum(b, 0.0) + peak(x, phi, a1, s, s, 2.0) + peak(x, phi + np.pi, a2, s, s, 2.0)

    return model


def peak_parts(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = tf.atan2(tf.sin(x - phi), tf.cos(x - phi))
    sigma = tf.where(
        d >= 0,
        tf.clip_by_value(sigma_plus, 0.05, 3.0),
        tf.clip_by_value(sigma_minus, 0.05, 3.0),
    )
    p = tf.clip_by_value(exponent, 0.5, 8.0) if tf.is_tensor(exponent) else min(max(exponent, 0.5), 8.0)
    absolute = tf.abs(d)
    u = absolute / sigma
    q = tf.pow(u, p)
    e = tf.exp(-0.5 * q)
    gain = tf.maximum(amplitude, 0.0)
    dphi = 0.5 * gain * e * p * tf.sign(d) * q / (absolute + 1e-12)
    dsigma = 0.5 * gain * e * q * p / sigma
    ds_plus = tf.where(d >= 0, dsigma, tf.zeros_like(dsigma))
    ds_minus = tf.where(d < 0, dsigma, tf.zeros_like(dsigma))
    dp = -0.5 * gain * e * q * tf.math.log(u + 1e-12)
    active = tf.cast(amplitude > 0, e.dtype)
    return gain * e, dphi, e * active, ds_plus, ds_minus, dp


def analytic_jacobian(program, x, p):
    cells, trials = x.shape
    columns = []
    if program == "winner":
        first = peak_parts(x, p[:, 0:1], p[:, 3:4], p[:, 5:6], p[:, 6:7], p[:, 9:10])
        second = peak_parts(x, p[:, 1:2], p[:, 4:5], p[:, 7:8], p[:, 8:9], p[:, 10:11])
        columns = [
            first[1],
            second[1],
            tf.cast(p[:, 2:3] > 0, p.dtype),
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
        second = peak_parts(x, p[:, 0:1] + np.pi, p[:, 3:4], p[:, 4:5], p[:, 4:5], 2.0)
        columns = [
            first[1] + second[1],
            tf.cast(p[:, 1:2] > 0, p.dtype),
            first[2],
            second[2],
            first[3] + first[4] + second[3] + second[4],
        ]
    return tf.stack([tf.broadcast_to(column, (cells, trials)) for column in columns], axis=2)


def make_fit(program, max_iterations, bounded):
    model = make_model(program)

    @tf.function
    def run(x, y, p0):
        def cost_of(p):
            residuals = model(x, p) - y
            return tf.reduce_sum(residuals * residuals, axis=1)

        p = p0
        damping = tf.fill([tf.shape(p)[0]], tf.constant(1e-3, tf.float32))
        current = cost_of(p)
        done = tf.zeros([tf.shape(p)[0]], tf.bool)

        def condition(i, p, damping, cost, done):
            if bounded:
                return tf.logical_and(i < max_iterations, tf.logical_not(tf.reduce_all(done)))
            return i < max_iterations

        def body(i, p, damping, cost, done):
            residuals = model(x, p) - y
            jac = analytic_jacobian(program, x, p)
            gradient = tf.einsum("ctk,ct->ck", jac, residuals)
            hessian = tf.einsum("ctk,ctl->ckl", jac, jac)
            diagonal = tf.linalg.diag_part(hessian)
            scale = tf.maximum(diagonal, 1e-9 * tf.reduce_max(diagonal, axis=1, keepdims=True) + 1e-12)
            damped = hessian + tf.linalg.diag(damping[:, None] * scale)
            step = tf.linalg.solve(damped, -gradient[..., None])[..., 0]
            step = tf.where(tf.math.is_finite(step), step, tf.zeros_like(step))
            capped = tf.zeros([tf.shape(p)[0]], tf.bool)
            if bounded:
                biggest = tf.reduce_max(tf.abs(step), axis=1, keepdims=True)
                capped = biggest[:, 0] > LM_MAX_STEP
                step = step * tf.minimum(1.0, LM_MAX_STEP / tf.maximum(biggest, 1e-30))
            candidate = p + step
            residuals = model(x, candidate) - y
            new_cost = tf.reduce_sum(residuals * residuals, axis=1)
            accept = tf.logical_and(tf.logical_not(done), new_cost < cost)
            if bounded:
                converged = tf.logical_or(
                    tf.logical_and(accept, cost - new_cost <= LM_TOLERANCE * cost),
                    damping >= MAX_DAMPING,
                )
                raised = tf.where(capped, damping * 2, damping / 3)
                update = tf.clip_by_value(tf.where(accept, raised, damping * 4), 1e-7, MAX_DAMPING)
                damping = tf.where(done, damping, update)
                done = tf.logical_or(done, converged)
            else:
                damping = tf.clip_by_value(tf.where(accept, damping / 3, damping * 4), 1e-7, MAX_DAMPING)
            p = tf.where(accept[:, None], candidate, p)
            cost = tf.where(accept, new_cost, cost)
            return i + 1, p, damping, cost, done

        start = (tf.constant(0), p, damping, current, done)
        i, p, damping, cost, done = tf.while_loop(condition, body, start)
        return p, i, tf.reduce_sum(tf.cast(done, tf.int32))

    return run


def program_params(program):
    return 11 if program == "winner" else 5


def make_adam(program, steps):
    model = make_model(program)

    @tf.function
    def step(x, y, p, m, v, t):
        with tf.GradientTape() as tape:
            tape.watch(p)
            residuals = model(x, p) - y
            loss = tf.reduce_mean(residuals * residuals)
        gradient = tape.gradient(loss, p)
        t = t + 1
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        m_hat = m / (1 - tf.pow(tf.constant(0.9), tf.cast(t, tf.float32)))
        v_hat = v / (1 - tf.pow(tf.constant(0.999), tf.cast(t, tf.float32)))
        update = 3e-3 * m_hat / (tf.sqrt(v_hat) + 1e-8)
        return p - update, m, v, t, loss

    def run(x, y, p0):
        p = p0
        m = tf.zeros_like(p)
        v = tf.zeros_like(p)
        t = tf.constant(0)
        best = p
        best_loss = np.inf
        done = 0
        for _ in range(steps):
            p, m, v, t, loss = step(x, y, p, m, v, t)
            value = float(loss)
            if not np.isfinite(value) or not bool(tf.reduce_all(tf.math.is_finite(p))):
                break
            done += 1
            if value < best_loss:
                best_loss = value
                best = p
        return best, done

    return run


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
        x = tf.constant(data["x_tr"])
        y = tf.constant(data["y_tr"])
        p0 = tf.constant(data["p0"])
        for algorithm in args.algorithms:
            if algorithm == "adam":
                run = make_adam(name, args.adam_steps)
                seconds, (params, iterations) = time_call(
                    lambda run=run, x=x, y=y, p0=p0: run(x, y, p0), args.repeats
                )
                converged = 0
            else:
                bounded = algorithm == "lm-bounded"
                run = make_fit(name, args.iterations, bounded)
                seconds, (params, iterations, converged) = time_call(
                    lambda run=run, x=x, y=y, p0=p0: run(x, y, p0), args.repeats
                )
            summarise(
                data,
                params.numpy(),
                seconds,
                int(iterations),
                "tensorflow",
                "cpu",
                algorithm,
                {"converged": int(converged), "tensorflow": tf.__version__},
            )


if __name__ == "__main__":
    main()

"""Per-cell Levenberg-Marquardt in numba, parallel over cells (CPU)."""

import argparse
import time

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
from numba import njit, prange

SIGMA_LO, SIGMA_HI = 0.05, 3.0
EXPONENT_LO, EXPONENT_HI = 0.5, 8.0


@njit(inline="always", fastmath=True)
def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


@njit(inline="always", fastmath=True)
def _peak(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = _wrap(x - phi)
    sigma = sigma_plus if d >= 0 else sigma_minus
    sigma = min(max(sigma, SIGMA_LO), SIGMA_HI)
    p = min(max(exponent, EXPONENT_LO), EXPONENT_HI)
    absolute = abs(d)
    u = absolute / sigma
    q = u**p
    e = np.exp(-0.5 * q)
    gain = amplitude if amplitude > 0 else 0.0
    active = e if amplitude > 0 else 0.0
    dphi = 0.5 * gain * e * p * np.sign(d) * q / (absolute + 1e-12)
    dsigma = 0.5 * gain * e * q * p / sigma
    ds_plus = dsigma if d >= 0 else 0.0
    ds_minus = dsigma if d < 0 else 0.0
    dp = -0.5 * gain * e * q * np.log(u + 1e-12)
    return gain * e, dphi, active, ds_plus, ds_minus, dp


@njit(inline="always", fastmath=True)
def _predict(x, y, p, winner):
    if winner:
        phi1, phi2, b, a1, a2 = p[0], p[1], p[2], p[3], p[4]
        s1p, s1m, s2p, s2m, p1, p2 = p[5], p[6], p[7], p[8], p[9], p[10]
        first = _peak(x, phi1, a1, s1p, s1m, p1)[0]
        second = _peak(x, phi2, a2, s2p, s2m, p2)[0]
        return (b if b > 0 else 0.0) + first + second
    phi, b, a1, a2, s = p[0], p[1], p[2], p[3], p[4]
    first = _peak(x, phi, a1, s, s, 2.0)[0]
    second = _peak(x, phi + np.pi, a2, s, s, 2.0)[0]
    return (b if b > 0 else 0.0) + first + second


@njit(fastmath=True)
def _cost(x, y, p, winner):
    total = 0.0
    for t in range(x.shape[0]):
        r = _predict(x[t], y[t], p, winner) - y[t]
        total += r * r
    return total


@njit(inline="always", fastmath=True)
def _fill_row(x, p, winner, row, out):
    if winner:
        phi1, phi2, b, a1, a2 = p[0], p[1], p[2], p[3], p[4]
        s1p, s1m, s2p, s2m, p1, p2 = p[5], p[6], p[7], p[8], p[9], p[10]
        first = _peak(x, phi1, a1, s1p, s1m, p1)
        second = _peak(x, phi2, a2, s2p, s2m, p2)
        out[0] = (b if b > 0 else 0.0) + first[0] + second[0]
        row[0] = first[1]
        row[1] = second[1]
        row[2] = 1.0
        row[3] = first[2]
        row[4] = second[2]
        row[5] = first[3]
        row[6] = first[4]
        row[7] = second[3]
        row[8] = second[4]
        row[9] = first[5]
        row[10] = second[5]
    else:
        phi, b, a1, a2, s = p[0], p[1], p[2], p[3], p[4]
        first = _peak(x, phi, a1, s, s, 2.0)
        second = _peak(x, phi + np.pi, a2, s, s, 2.0)
        out[0] = (b if b > 0 else 0.0) + first[0] + second[0]
        row[0] = first[1] + second[1]
        row[1] = 1.0
        row[2] = first[2]
        row[3] = second[2]
        row[4] = first[3] + first[4] + second[3] + second[4]


@njit(parallel=True, fastmath=True, cache=True)
def _fit(x, y, p0, winner, bounded, max_iterations):
    cells, trials = x.shape
    k = p0.shape[1]
    out = np.empty_like(p0)
    iterations = np.zeros(cells, np.int32)
    converged = np.zeros(cells, np.int8)
    for c in prange(cells):
        p = p0[c].copy()
        damping = np.float32(1e-3)
        cost = _cost(x[c], y[c], p, winner)
        done = False
        row = np.zeros(k, np.float32)
        prediction = np.zeros(1, np.float32)
        hessian = np.zeros((k, k), np.float32)
        gradient = np.zeros(k, np.float32)
        step = np.zeros(k, np.float32)
        used = 0
        for iteration in range(max_iterations):
            if bounded and done:
                break
            for j in range(k):
                gradient[j] = 0.0
                for j2 in range(k):
                    hessian[j, j2] = 0.0
            for t in range(trials):
                _fill_row(x[c, t], p, winner, row, prediction)
                r = prediction[0] - y[c, t]
                for j in range(k):
                    gradient[j] += row[j] * r
                    for j2 in range(k):
                        hessian[j, j2] += row[j] * row[j2]
            diagonal_max = 0.0
            for j in range(k):
                diagonal_max = max(diagonal_max, hessian[j, j])
            for j in range(k):
                scale = max(hessian[j, j], 1e-9 * diagonal_max + 1e-12)
                hessian[j, j] += damping * scale
            step[:] = np.linalg.solve(hessian, -gradient)
            finite = True
            for j in range(k):
                if not np.isfinite(step[j]):
                    finite = False
            if not finite:
                for j in range(k):
                    step[j] = 0.0
            capped = False
            if bounded:
                biggest = 0.0
                for j in range(k):
                    biggest = max(biggest, abs(step[j]))
                if biggest > LM_MAX_STEP:
                    capped = True
                    factor = LM_MAX_STEP / max(biggest, 1e-30)
                    for j in range(k):
                        step[j] *= factor
            for j in range(k):
                row[j] = p[j] + step[j]
            new_cost = _cost(x[c], y[c], row, winner)
            accept = (not done) and new_cost < cost and np.isfinite(new_cost)
            if bounded:
                converged_cell = (accept and (cost - new_cost) <= LM_TOLERANCE * cost) or (
                    damping >= MAX_DAMPING
                )
                raised = damping * 2 if capped else damping / 3
                damping = min(max(raised if accept else damping * 4, 1e-7), MAX_DAMPING)
                done = done or converged_cell
            else:
                damping = min(max(damping / 3 if accept else damping * 4, 1e-7), MAX_DAMPING)
            if accept:
                for j in range(k):
                    p[j] = row[j]
                cost = new_cost
            used = iteration + 1
        out[c, :] = p
        iterations[c] = used
        converged[c] = 1 if done else 0
    return out, int(iterations.max()), int(converged.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=LM_MAX_ITERATIONS)
    args = parser.parse_args(argv)
    for name in args.programs:
        data = load(name)
        winner = name == "winner"
        for bounded in (False, True):
            start = time.perf_counter()
            _fit(
                data["x_tr"],
                data["y_tr"],
                data["p0"],
                winner,
                bounded,
                1,
            )
            compile_seconds = time.perf_counter() - start
            seconds, (params, iterations, converged) = time_call(
                lambda bounded=bounded, data=data, winner=winner: _fit(
                    data["x_tr"],
                    data["y_tr"],
                    data["p0"],
                    winner,
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
                "numba",
                "cpu",
                "lm-bounded" if bounded else "lm",
                {"converged": converged, "compile_seconds": round(compile_seconds, 3)},
            )


if __name__ == "__main__":
    main()

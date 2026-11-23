"""SciPy least-squares and L-BFGS-B on a cell subset, single-threaded, then extrapolated."""

import argparse
import json
import statistics
import time

import numpy as np
from common import jacobian, load, predict, score, train_loss
from scipy.optimize import least_squares, minimize


def fit_cell(program, x, y, p0, method):
    def residual(p):
        return predict(program, x[None, :], p[None, :])[0] - y

    def jac(p):
        return jacobian(program, x[None, :], p[None, :])[0]

    def cost(p):
        r = residual(p)
        return 0.5 * np.sum(r * r)

    def gradient(p):
        with np.errstate(all="ignore"):
            return jac(p).T @ residual(p)

    if method == "lbfgsb":
        result = minimize(
            cost,
            p0.astype(np.float64),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        return result.x, result.nfev

    if method == "bfgs":
        result = minimize(
            cost,
            p0.astype(np.float64),
            jac=gradient,
            method="BFGS",
            options={"maxiter": 1000},
        )
        return result.x, result.nfev

    if method == "powell":
        result = minimize(
            cost,
            p0.astype(np.float64),
            method="Powell",
            options={"maxiter": 1000, "maxfev": 5000},
        )
        return result.x, result.nfev

    result = least_squares(
        residual,
        p0.astype(np.float64),
        jac=jac,
        method=method,
        max_nfev=1000,
    )
    return result.x, result.nfev


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lm", "trf", "dogbox", "lbfgsb", "bfgs", "powell"],
    )
    parser.add_argument("--cells", type=int, default=200)
    args = parser.parse_args(argv)
    for name in args.programs:
        data = load(name)
        cells = min(args.cells, data["x_tr"].shape[0])
        x_tr, y_tr = data["x_tr"][:cells], data["y_tr"][:cells]
        x_te, y_te = data["x_te"][:cells], data["y_te"][:cells]
        for method in args.methods:
            start = time.perf_counter()
            outputs = [fit_cell(name, x_tr[i], y_tr[i], data["p0"][i], method) for i in range(cells)]
            seconds = time.perf_counter() - start
            params = np.stack([out[0] for out in outputs])
            evaluations = statistics.mean(out[1] for out in outputs)
            scores = [
                score(name, params[i : i + 1], x_te[i : i + 1], y_te[i : i + 1], data["n_params"])
                for i in range(cells)
            ]
            losses = [
                train_loss(name, params[i : i + 1], x_tr[i : i + 1], y_tr[i : i + 1]) for i in range(cells)
            ]
            print(
                json.dumps(
                    {
                        "backend": "scipy",
                        "device": "cpu",
                        "program": name,
                        "algorithm": method,
                        "cells": cells,
                        "seconds": round(seconds, 4),
                        "seconds_per_1055_cells": round(seconds * 1055 / cells, 3),
                        "iterations": round(evaluations, 1),
                        "train_loss": round(float(np.mean(losses)), 6),
                        "score": round(float(np.mean(scores)), 6),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

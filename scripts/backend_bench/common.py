"""Shared loading, scoring and reporting for the backend benchmark."""

import json
import statistics
import time
from pathlib import Path

import numpy as np

from edgar_study import curves_numpy

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "bench"

LM_MAX_STEP = 0.25
LM_TOLERANCE = 1e-4
LM_MAX_ITERATIONS = 100
MAX_DAMPING = 1e7


def load(program):
    path = DATA / f"{program}.npz"
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    data["program"] = str(data["program"])
    data["n_params"] = int(data["n_params"])
    return data


def predict(program, x, params):
    return curves_numpy.model(program, x, params)


def jacobian(program, x, params):
    return curves_numpy.jacobian(program, x, params)


def score(program, params, x, y, n_params):
    residuals = predict(program, x, params) - y
    return float(10 * np.mean(residuals**2) + 0.01 * n_params)


def train_loss(program, params, x, y):
    return float(np.mean((predict(program, x, params) - y) ** 2))


def time_call(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), out


def report(backend, device, program, algorithm, seconds, iterations, train, heldout, extra=None):
    row = {
        "backend": backend,
        "device": device,
        "program": program,
        "algorithm": algorithm,
        "seconds": round(float(seconds), 4),
        "iterations": int(iterations),
        "train_loss": round(float(train), 6),
        "score": round(float(heldout), 6),
    }
    if extra:
        row.update(extra)
    print(json.dumps(row), flush=True)
    return row


def summarise(data, params, seconds, iterations, backend, device, algorithm, extra=None):
    return report(
        backend,
        device,
        data["program"],
        algorithm,
        seconds,
        iterations,
        train_loss(data["program"], params, data["x_tr"], data["y_tr"]),
        score(data["program"], params, data["x_te"], data["y_te"], data["n_params"]),
        extra,
    )

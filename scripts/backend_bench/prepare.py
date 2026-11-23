"""Materialise the real recording's discovery cells for the backend benchmark.

Writes one `.npz` per program with the exact upstream trial split (PRNGKey(0), 50/50), the
estimator's starting parameters, and float32 arrays so every backend fits identical inputs.
"""

import argparse
import contextlib
import io
from pathlib import Path

import jax
import numpy as np

from edgar_study import curves
from edgar_study.datasets import load_dataset, prepare
from edgar_study.upstream import import_upstream

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "bench"


def discovery(dataset, modules):
    response, angles, _ = load_dataset(dataset, modules["utils"].extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")
    rows = groups["discovery"]
    return np.asarray(x[rows]), np.asarray(y[rows])


def prepare_program(name, x, y):
    curve = next(c for c in curves.CURVES if c.name == name)
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(0), np.arange(y.shape[1])))
    train, test = order[: y.shape[1] // 2], order[y.shape[1] // 2 :]
    p0 = np.stack([curve.estimator(x[i, train], y[i, train]) for i in range(x.shape[0])])
    arrays = {
        "program": np.array(name),
        "x_tr": x[:, train].astype(np.float32),
        "y_tr": y[:, train].astype(np.float32),
        "x_te": x[:, test].astype(np.float32),
        "y_te": y[:, test].astype(np.float32),
        "p0": p0.astype(np.float32),
        "n_params": np.array(curve.n_params),
    }
    path = OUT / f"{name}.npz"
    np.savez(path, **arrays)
    return path, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "gratings_drifting_GT1_2019_04_12_1.npy",
    )
    parser.add_argument("--programs", nargs="+", default=["double-gaussian", "winner"])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        modules = import_upstream(ROOT)
    x, y = discovery(args.dataset, modules)
    for name in args.programs:
        path, arrays = prepare_program(name, x, y)
        print(
            f"{path.name}: {arrays['x_tr'].shape[0]} cells x {arrays['x_tr'].shape[1]} "
            f"fit trials, {arrays['n_params']} params"
        )


if __name__ == "__main__":
    main()

"""How far the default `lm-bounded` fit is from the best fit we can find for each curve.

For each paper curve on the real recording's discovery cells, fit once with the search's default
`lm-bounded` and once with `fit_multistart`: the estimator's start plus perturbed copies, a tighter
tolerance and a larger iteration budget. The multi-start fit approximates the floor of the training
loss. A small gap means the search's fitter already finds the floor; a large one means programs are
scored on fits that a better optimiser would improve.

Run with `uv run scripts/fit_floor/run.py` and record results in `results.md`.
"""

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import curves, fitting
from edgar_study.datasets import load_dataset, prepare
from edgar_study.upstream import PENALTY, import_upstream

log = logging.getLogger("fit_floor")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVES = ("double-gaussian", "generalised", "winner")
IMPROVED = 0.01


def discovery_cells(path, utils, n_cells):
    response, angles, _ = load_dataset(path, utils.extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")
    cells = groups["discovery"][:n_cells]
    return x[cells], y[cells]


def trial_split(n_trials, random_seed=0):
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(random_seed), jnp.arange(n_trials)))
    return order[: n_trials // 2], order[n_trials // 2 :]


def per_cell(model, loss, params, x, y):
    values = jax.vmap(lambda p, xc, yc: jnp.mean(loss(model(xc, *p), yc)))(
        jnp.asarray(params), jnp.asarray(x), jnp.asarray(y)
    )
    return np.asarray(values)


def measure(name, model, loss, params, seconds, x, y, train, test):
    train_cells = per_cell(model, loss, params, x[:, train], y[:, train])
    test_cells = per_cell(model, loss, params, x[:, test], y[:, test])
    return {
        "fit": name,
        "seconds": round(seconds, 2),
        "train_loss": float(np.nanmean(train_cells)),
        "score": float(np.nanmean(test_cells) + PENALTY * params.shape[1]),
        "train_cells": train_cells,
    }


def compare(curve, engine, loss, x, y, starts, iterations, tolerance):
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, start = engine.objective(
            curve.model, curve.estimator, loss, x, y, param_penalty_weight=PENALTY, max_iter=0
        )
    start = np.asarray(start)
    train, test = trial_split(y.shape[1])
    x_train, y_train = x[:, train], y[:, train]

    began = time.perf_counter()
    default = fitting.fit_lm_bounded(curve.model, start, x_train, y_train).params
    default_row = measure(
        "lm-bounded", curve.model, loss, default, time.perf_counter() - began, x, y, train, test
    )

    began = time.perf_counter()
    floor = fitting.fit_multistart(
        curve.model, start, x_train, y_train, starts=starts, max_iterations=iterations, tolerance=tolerance
    )
    floor_row = measure(
        f"multistart ×{starts}",
        curve.model,
        loss,
        floor.params,
        time.perf_counter() - began,
        x,
        y,
        train,
        test,
    )

    improved = floor_row["train_cells"] < default_row["train_cells"] * (1 - IMPROVED)
    summary = {
        "program": curve.name,
        "params": curve.n_params,
        "cells": int(x.shape[0]),
        "default": {k: v for k, v in default_row.items() if k != "train_cells"},
        "floor": {k: v for k, v in floor_row.items() if k != "train_cells"},
        "train_gap": default_row["train_loss"] - floor_row["train_loss"],
        "score_gap": default_row["score"] - floor_row["score"],
        "cells_improved_by_1pct": float(np.mean(improved)),
        "cells_won_by_a_perturbed_start": float(np.mean(floor.best_start > 0)),
    }
    log.info("%s: %s", curve.name, json.dumps(summary))
    return summary


def table(rows):
    lines = [
        "| Program | Params | Fit | Seconds | Train loss | Score | Train gap | Score gap "
        "| Cells improved >1% | Won by perturbed start |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for key in ("default", "floor"):
            fit = row[key]
            extra = (
                f"{row['train_gap']:.4f} | {row['score_gap']:+.4f} | {row['cells_improved_by_1pct']:.1%} | "
                f"{row['cells_won_by_a_perturbed_start']:.1%}"
                if key == "floor"
                else " |  |  | "
            )
            lines.append(
                f"| {row['program']} | {row['params']} | {fit['fit']} | {fit['seconds']:.2f} | "
                f"{fit['train_loss']:.4f} | {fit['score']:.4f} | {extra} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "gratings_drifting_GT1_2019_04_12_1.npy"
    )
    parser.add_argument("--curves", nargs="+", default=list(DEFAULT_CURVES))
    parser.add_argument("--cells", type=int, default=264)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    upstream = import_upstream(ROOT)
    engine, loss = upstream["hypothesis_engine"], upstream["loss_functions"].quadratic_loss
    x, y = discovery_cells(args.dataset, upstream["utils"], args.cells)
    selected = [c for c in curves.CURVES if c.name in args.curves]
    rows = [compare(c, engine, loss, x, y, args.starts, args.iterations, args.tolerance) for c in selected]
    print(table(rows))


if __name__ == "__main__":
    main()

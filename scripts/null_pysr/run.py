"""PySR reference arm of the no-LLM null baseline.

PySR searches one formula f(theta, q1..qK) whose K parameters are fitted per neuron (a template
expression indexed by neuron), on a random subsample of the discovery cells and only their fit-half
trials. Every formula on its Pareto front is then converted to an upstream-style program, scored on
all discovery cells with our objective, and written as an EDGAR run folder, so `edgar rescore` picks
the winner on the selection cells and scores it once on the locked cells, exactly as for LLM runs.

Needs the optional `sr` dependency group (PySR and Julia):

    uv run --group sr python scripts/null_pysr/run.py --dataset data/....npy --results results
"""

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from edgar_study import fitting, symbolic
from edgar_study.datasets import load_dataset, prepare
from edgar_study.engine import git_sha
from edgar_study.upstream import PENALTY, code_key, import_upstream, score_program

ROOT = Path(__file__).resolve().parents[2]
PROPOSER = "PySR (no LLM)"
log = logging.getLogger(__name__)


def search(x, y, args):
    from pysr import PySRRegressor, TemplateExpressionSpec

    n_cells, n_trials = y.shape
    names = [f"q{i}" for i in range(1, args.parameters + 1)]
    spec = TemplateExpressionSpec(
        combine="f(theta, " + ", ".join(f"{n}[cell]" for n in names) + ")",
        expressions=["f"],
        variable_names=["theta", "cell"],
        parameters={n: n_cells for n in names},
    )
    model = PySRRegressor(
        expression_spec=spec,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "cos", "sin", "abs"],
        maxsize=symbolic.MAXSIZE,
        niterations=100_000,
        timeout_in_seconds=args.minutes * 60,
        batching=True,
        batch_size=4096,
        random_state=args.seed,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    cell = np.repeat(np.arange(1, n_cells + 1), n_trials).astype(float)
    model.fit(np.column_stack([np.asarray(x).ravel(), cell]), np.asarray(y).ravel())
    return model.equations_


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--parameters", type=int, default=6)
    parser.add_argument("--minutes", type=float, default=55)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    upstream = import_upstream(ROOT)
    he, utils, seeds = upstream["hypothesis_engine"], upstream["utils"], upstream["seed_programs"]
    fitter = fitting.install(he, "lm-bounded")
    response, angles, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    discovery = np.asarray(groups["discovery"])
    xd, yd = x[discovery], y[discovery]

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-pysr" + uuid.uuid4().hex[:4]
    run_dir = (args.results / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True)
    started = time.time()
    manifest = {
        "run_id": run_id,
        "method": "pysr",
        "seed": args.seed,
        "repo_sha": git_sha(ROOT),
        "upstream_commit": upstream["commit"],
        "dataset": str(args.dataset.resolve()),
        "iterations": 1,
        "models": {"translator": PROPOSER, "proposer": PROPOSER, "estimator": PROPOSER},
        "fitter": fitter,
        "islands": 1,
        "pysr": {"cells": args.cells, "parameters": args.parameters, "minutes": args.minutes},
        "started": started,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {run_dir}")

    picked = np.random.default_rng(args.seed).choice(len(discovery), args.cells, replace=False)
    fit_columns = np.asarray(indices["fit_trial_columns"])
    search_started = time.time()
    front = search(xd[picked][:, fit_columns], yd[picked][:, fit_columns], args)
    search_s = time.time() - search_started

    rows, census = [], []
    for k in (1, 2):
        score, *_ = score_program(
            upstream,
            getattr(seeds, f"neuron_model_{k}_jax"),
            getattr(seeds, f"parameter_estimator_{k}"),
            xd,
            yd,
            1_000,
        )
        census.append([-1, -1, k - 1, None, score, time.time() - started, None, None, None, 4 + k - 1])
    fit_started = time.time()
    with open(run_dir / "translations.jsonl", "w") as stream:
        for batch, equation in enumerate(front["equation"]):
            try:
                tree = symbolic.from_pysr(equation)
            except ValueError as error:
                log.info("Skipping %r: %s", equation.split(";")[0], error)
                continue
            if not symbolic.valid(tree):
                log.info("Skipping %r: no theta or no per-neuron parameter", equation.split(";")[0])
                continue
            code, jax_code = symbolic.to_numpy_source(tree), symbolic.to_jax_source(tree)
            estimator = symbolic.estimator_source(tree)
            score, *_, count, _ = score_program(
                upstream,
                utils.str_to_func(jax_code, "neuron_model"),
                utils.str_to_func(estimator, "parameter_estimator"),
                xd,
                yd,
                1_000,
            )
            log.info("%-60s discovery %.4f (%d params)", symbolic.to_infix(tree)[:60], score, count)
            stream.write(
                json.dumps(
                    {
                        "numpy_sha256": code_key(code),
                        "numpy_code": code,
                        "jax_code": jax_code,
                        "compiled": True,
                    }
                )
                + "\n"
            )
            rows.append(
                {
                    "program_code_string": code,
                    "parameter_estimator_code_string": estimator,
                    "jax_code_string": jax_code,
                    "iteration_number": 0,
                    "birth_island": 0,
                    "batch_index": batch,
                    "train_loss": score if np.isfinite(score) else np.inf,
                    "llm_name": PROPOSER,
                    "pysr_equation": equation.split(";")[0],
                }
            )
            census.append([0, 0, batch, PROPOSER, score, time.time() - started, None, None, None, count])

    stamp = pd.Timestamp.now()
    base = run_dir / "program_databases" / stamp.strftime("%m-%d") / stamp.strftime("%H-%M-%S")
    (base / "island_0").mkdir(parents=True)
    (base / "combined").mkdir()
    pd.DataFrame(rows).to_csv(base / "island_0" / "programs_db.csv", index=False)
    np.save(base / "combined" / "census.npy", np.array(census, dtype=object), allow_pickle=True)
    finished = time.time()
    manifest.update(
        status="completed",
        finished=finished,
        batch_size=max(len(front), 1),
        front_size=len(front),
        programs_scored=len(rows),
        spend_before_usd=0.0,
        spend_after_usd=0.0,
        stopped_early=False,
        timings={
            "total_s": finished - started,
            "stages": {
                "pysr_search": {"calls": 1, "busy_s": search_s, "wall_s": search_s},
                "fit": {
                    "calls": len(rows),
                    "busy_s": finished - fit_started,
                    "wall_s": finished - fit_started,
                },
            },
            "other_s": finished - started - search_s - (finished - fit_started),
        },
        penalty=PENALTY,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

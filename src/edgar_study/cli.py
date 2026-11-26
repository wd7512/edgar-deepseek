"""EDGAR study commands: offline baseline, OpenCode Go preflight, search runs and locked-cell rescoring."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from edgar_study.datasets import (
    ACTIVITY_THRESHOLD,
    CONCENTRATION_THRESHOLD,
    SPLIT_SEED,
    TRIAL_SEED,
    load_dataset,
    prepare,
    synthetic_recording,
)
from edgar_study.upstream import MSE_SCALE, PENALTY, UPSTREAM_COMMIT, import_upstream


def run_baseline(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise ValueError("The baseline requires a CPU JAX backend.")
    upstream = import_upstream(args.upstream.resolve())
    objective = upstream["hypothesis_engine"].objective
    quadratic_loss = upstream["loss_functions"].quadratic_loss
    seeds, utils = upstream["seed_programs"], upstream["utils"]
    if args.synthetic:
        response, angles = synthetic_recording()
        sha256 = None
    else:
        response, angles, sha256 = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, args.protocol)

    metrics = {}
    for partition in ("discovery", "selection"):
        rows = groups[partition]
        metrics[partition] = {}
        for label, model, estimator in (
            ("single_gaussian", seeds.neuron_model_1_jax, seeds.parameter_estimator_1),
            ("double_gaussian", seeds.neuron_model_2_jax, seeds.parameter_estimator_2),
        ):
            # EDGAR samples cells for JAX compatibility checks using NumPy's RNG.
            np.random.seed(0)
            initial_score, _, score, params = objective(
                model,
                estimator,
                quadratic_loss,
                x[rows],
                y[rows],
                random_seed=TRIAL_SEED,
                max_iter=1000,
                tol=1e-6,
                param_penalty_weight=PENALTY,
            )
            columns = indices["score_trial_columns"]
            predictions = jax.vmap(lambda stimuli, p, model=model: model(stimuli, *p))(
                x[rows][:, columns], params
            )
            mse = float(jnp.mean(jnp.square(predictions - y[rows][:, columns])))
            count = params.shape[1]
            if not np.isfinite([initial_score, score, mse]).all():
                raise ValueError(f"Nonfinite baseline metrics for {partition}/{label}.")
            if not np.isclose(score, MSE_SCALE * mse + PENALTY * count, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Upstream score differs from the recorded MSE for {partition}/{label}.")
            metrics[partition][label] = {
                "initial_score": initial_score,
                "score": score,
                "mse": mse,
                "parameters": count,
            }

    manifest = {
        "dataset": "synthetic" if args.synthetic else args.dataset.name,
        "dataset_sha256": sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol": args.protocol,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "activity_threshold": ACTIVITY_THRESHOLD,
        "concentration_threshold": CONCENTRATION_THRESHOLD,
        "split_seed": SPLIT_SEED,
        "trial_seed": TRIAL_SEED,
        "penalty": PENALTY,
        "cells_in_recording": response.shape[0],
        "trials_in_recording": response.shape[1],
        "trials_per_eligible_cell": x.shape[1],
        "partition_counts": {key: len(rows) for key, rows in groups.items()},
        "metrics": metrics,
    }
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staging = Path(temp) / "run"
        staging.mkdir()
        np.savez_compressed(staging / "splits.npz", **indices)
        (staging / "baseline.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
    print(f"Wrote {target / 'baseline.json'} and {target / 'splits.npz'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    upstream = argparse.ArgumentParser(add_help=False)
    upstream.add_argument("--upstream", type=Path, default=Path("."))
    fitter = argparse.ArgumentParser(add_help=False)
    fitter.add_argument(
        "--fitter",
        choices=("auto", "upstream", "adam", "lm", "lm-bounded"),
        default=None,
        help="Per-cell parameter fitting. auto (run default) is lm-bounded: "
        "Levenberg-Marquardt with a capped step that stops at a tolerance, about 8x faster "
        "than upstream with scores within 1%%. upstream is upstream's loop; adam is the same "
        "Adam steps compiled; lm is uncapped Levenberg-Marquardt. rescore defaults to the "
        "run's own fitter.",
    )
    dataset = argparse.ArgumentParser(add_help=False)
    dataset.add_argument("--dataset", type=Path, required=True)
    models = argparse.ArgumentParser(add_help=False)
    models.add_argument("--model", default="hy4-flash-free", help="Translator, proposer and estimator model.")
    models.add_argument(
        "--image-model",
        default=None,
        help="Optional vision model that describes diagnostic images for --model.",
    )
    models.add_argument(
        "--allow-paid",
        action="store_true",
        help="Permit models with a nonzero price (spend cap still applies).",
    )
    models.add_argument("--results", type=Path, default=Path("results"))
    models.add_argument("--max-tokens", type=int, default=32_000)
    models.add_argument(
        "--reasoning-effort",
        default="none",
        help="Effort for translator, routine proposals and estimators "
        "(upstream's non-thinking Gemini 2.0 tiers).",
    )

    baseline = commands.add_parser(
        "baseline", parents=[upstream], help="Fit and score the two Gaussian seeds."
    )
    source = baseline.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--synthetic", action="store_true")
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument(
        "--protocol", choices=("paper-fidelity", "strict-holdout-v1"), default="strict-holdout-v1"
    )
    baseline.set_defaults(func=run_baseline)

    preflight = commands.add_parser(
        "preflight", parents=[models], help="Check model availability, text and image calls."
    )
    preflight.set_defaults(func=lambda args: _load("preflight", "preflight")(args))

    run = commands.add_parser(
        "run",
        parents=[models, dataset, upstream, fitter],
        help="Run the upstream EDGAR search with OpenCode Go models.",
    )
    run.add_argument("--iterations", type=int, default=12)
    run.add_argument("--time-limit", type=float, default=600, help="Minutes.")
    run.add_argument(
        "--large-model",
        default=None,
        help="Model for every third generation (upstream's thinking tier). "
        "Defaults to --model with --large-reasoning-effort.",
    )
    run.add_argument("--large-reasoning-effort", default="high")
    run.add_argument(
        "--temperature-scale",
        type=float,
        default=0.5,
        help="Multiplies upstream's temperatures, which run from 1 to 2.",
    )
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument("--no-images", action="store_true", help="Text-only feedback condition.")
    run.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Wait for every proposal before translating, estimating and fitting, as upstream does.",
    )
    run.add_argument("--islands", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=6, help="Proposals per island per generation.")
    run.add_argument(
        "--cell-fraction",
        type=float,
        default=1.0,
        help="Search on this random fraction of the discovery cells only (rescore still "
        "uses the full selection and locked-test cells).",
    )
    run.set_defaults(func=lambda args: _load("engine", "run")(args))

    frontier = commands.add_parser(
        "frontier", parents=[dataset, upstream], help="Run a score-blind HY4 wave."
    )
    frontier.add_argument("--output", type=Path, required=True)
    frontier.add_argument(
        "--resume", action="store_true", help="Continue a checkpoint using the original options."
    )
    frontier.add_argument("--seed", type=int, default=0)
    frontier.add_argument("--batches", type=int, default=6)
    frontier.add_argument("--waves", type=int, default=1)
    frontier.add_argument("--graph", type=Path, help="Use a fixed catalog instead of HY4-generated ideas.")
    frontier.add_argument("--k", type=int, default=4, help="Maximum equations per generated probe.")
    frontier.add_argument(
        "--new-probability", type=float, default=0.8, help="Chance of a new idea vs. a revisit."
    )
    frontier.add_argument(
        "--starts",
        type=int,
        default=5,
        help="Starts per fit: the model's own, seeded jitters, and a warm start from the parent.",
    )
    frontier.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Follow-up turns per batch for the model to check equations and replace ones that fail.",
    )
    frontier.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Weight parents and revisits by exp(-(score - lowest) / T); the model never sees scores.",
    )
    frontier.add_argument(
        "--rank-bias",
        type=float,
        default=None,
        help="Weight every node choice by 1 / (1 + B * rank / worst rank), best score first; the worst node is"
        " 1 / (1 + B) as likely as the best. Not with --temperature.",
    )
    frontier.add_argument("--finalize", action="store_true", help="Use selection and locked-test cells once.")
    frontier.set_defaults(func=lambda args: _load("frontier.runner", "run")(args))

    fitbench = commands.add_parser(
        "fitbench",
        parents=[upstream],
        help="Benchmark fitters on frontier equations that contain one another.",
    )
    fitbench.add_argument("--dataset", type=Path, default=Path("data/gratings_drifting_GT1_2019_04_12_1.npy"))
    fitbench.add_argument("--cells", type=int, default=64, help="Exploration cells in the benchmark cohort.")
    fitbench.add_argument(
        "--fitters",
        default="protocol,warm",
        help="Comma-separated fitters to run (protocol, warm, single, reference); empty only reports.",
    )
    fitbench.add_argument("--output", type=Path, default=Path("results/fitbench"))
    fitbench.set_defaults(func=lambda args: _load("frontier.bench", "run")(args))

    ab = commands.add_parser(
        "prompt-ab", parents=[dataset, upstream], help="A/B test proposal prompts on fixed contexts."
    )
    ab.add_argument("--wave", type=Path, required=True, help="A finished run's wave.json for contexts.")
    ab.add_argument("--output", type=Path, required=True)
    ab.add_argument("--arms", default="v1,v2", help="Comma-separated prompt variants.")
    ab.add_argument("--samples", type=int, default=3, help="Replies per arm and context.")
    ab.add_argument("--cells", type=int, default=128, help="Exploration neurons each equation is fitted on.")
    ab.add_argument("--k", type=int, default=4)
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--max-iter", type=int, default=2000)
    ab.set_defaults(func=lambda args: _load("frontier.ab", "run")(args))

    comparison = commands.add_parser(
        "compare",
        parents=[dataset, upstream],
        help="Compare frontier candidates with the paper equation and double Gaussian under one protocol.",
    )
    comparison.add_argument("--wave", type=Path, required=True, help="A run's wave.json for candidates.")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--protocol", choices=("native", "common-start"), default="common-start")
    comparison.add_argument(
        "--cohort", choices=("exploration", "confirmation", "discovery"), default="confirmation"
    )
    comparison.add_argument("--top", type=int, default=3, help="Best nodes to compare, by exploration score.")
    comparison.add_argument("--nodes", help="Comma-separated node IDs instead of --top.")
    comparison.add_argument("--starts", type=int, default=5)
    comparison.add_argument("--seed", type=int, default=0)
    comparison.add_argument("--max-iter", type=int, default=2000)
    comparison.add_argument("--fitter", choices=("lm-bounded", "adam"), default="lm-bounded")
    comparison.add_argument("--learning-rate", type=float, default=3e-3, help="Adam only.")
    comparison.add_argument(
        "--patience", type=int, default=200, help="Adam: steps without improvement before a fit stops."
    )
    comparison.add_argument(
        "--tolerance", type=float, default=1e-6, help="Adam: relative improvement that resets patience."
    )
    comparison.add_argument("--cells", type=int, help="Use only the first N cells of the cohort.")
    comparison.add_argument(
        "--against", type=Path, help="Another summary.json to add; refused if its protocol differs."
    )
    comparison.set_defaults(func=lambda args: _load("frontier.compare", "run")(args))

    viewer = commands.add_parser("frontier-view", help="Serve a live frontier graph and batch replay.")
    viewer.add_argument("--run", type=Path, required=True, help="Run directory or wave.json.")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.set_defaults(func=lambda args: _load("frontier.viewer", "serve")(args))

    rescore = commands.add_parser(
        "rescore",
        parents=[dataset, upstream, fitter],
        help="Select on validation cells, score once on locked cells.",
    )
    rescore.add_argument("--run", type=Path, required=True)
    rescore.add_argument("--top", type=int, default=10)
    rescore.set_defaults(func=lambda args: _load("rescore", "rescore")(args))

    recover = commands.add_parser(
        "recover", help="Rebuild the program databases of a run that stopped early from its log and census."
    )
    recover.add_argument("--run", type=Path, required=True)
    recover.set_defaults(
        func=lambda args: print(f"Recovered {_load('recover', 'recover')(args.run)} programs")
    )

    report = commands.add_parser("report", help="Summarise a run and plot best-so-far discovery loss.")
    report.add_argument("--run", type=Path, required=True)
    report.set_defaults(func=lambda args: _load("report", "report")(args))

    args = parser.parse_args()
    args.func(args)


def _load(module, name):
    """Import subcommands lazily so the offline baseline needs no network client."""
    import importlib

    return getattr(importlib.import_module(f"edgar_study.{module}"), name)


if __name__ == "__main__":
    main()

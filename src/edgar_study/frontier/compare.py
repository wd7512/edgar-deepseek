"""Compare frontier candidates with the paper equation and the double Gaussian under one explicit protocol.

An exploration score is not comparable across models unless they share the cohort, trial split, number of
starts, seeds, optimizer budget and starting information. This module fixes all of those and records
them. Only discovery cells (exploration and confirmation) can be compared: selection and locked-test
cells are never touched here.

Protocols:
    native        each model starts from its own estimator (a candidate from its declared values).
    common-start  every model starts from the double Gaussian's fit to the same cell's training trials,
                  mapped into the model by parameter name (exactly, for the paper equation); parameters
                  with no counterpart start from the model's own value. The double Gaussian is refitted
                  from that point too, so every model gets the same starting information and budget.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import (
    DISCOVERY_PARTITION_SEED,
    TRIAL_SEED,
    discovery_partition,
    load_dataset,
    prepare,
)
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import JITTER, canonical, fit_training, predict, split_trials
from edgar_study.frontier.seed import BASE_NAMES, BASELINE_ID
from edgar_study.paper_models import ai_estimator, ai_model, parameter_names
from edgar_study.upstream import MSE_SCALE, PENALTY, import_upstream

PROTOCOLS = ("native", "common-start")
COHORTS = ("exploration", "confirmation", "discovery")
SAME_PROTOCOL = ("protocol", "cohort", "starts", "seed", "max_iter")
PAPER = "paper_equation"
PAPER_LINKS = {
    "theta_pref": ("m", None),
    "baseline": ("b", None),
    "amplitude_1": ("a1", None),
    "amplitude_2": ("a2", None),
    "tuning_width_1_left": ("s", None),
    "tuning_width_1_right": ("s", None),
    "tuning_width_2_left": ("s", None),
    "tuning_width_2_right": ("s", None),
    "exponent_1": (None, 2.0),
    "exponent_2": (None, 2.0),
    "angle_offset_2": (None, 0.0),
}
log = logging.getLogger(__name__)


class Spec(NamedTuple):
    """A model to evaluate: parameters in `names` order, and an estimator giving a start per cell."""

    name: str
    kind: str
    model: Callable
    estimator: Callable
    names: tuple[str, ...]


def cohort_rows(groups, name):
    """Cell rows of a discovery cohort. Selection and locked-test cells are refused."""
    if name not in COHORTS:
        raise ValueError(
            f"Unknown cohort {name!r}: choose from {COHORTS}; selection and locked cells are not comparable."
        )
    exploration, confirmation = discovery_partition(groups)
    return {"exploration": exploration, "confirmation": confirmation, "discovery": groups["discovery"]}[name]


def best_per_node(wave, top=None, nodes=None):
    """The best-scoring fitted equation of each node (not the first one), best nodes first."""
    best = {}
    for event in wave["episodes"]:
        for record in event["candidates"]:
            if record.get("status") != "fitted":
                continue
            score = record["metrics"]["score"]
            if event["node"] not in best or score < best[event["node"]][0]:
                best[event["node"]] = (score, record["proposal"])
    ranked = sorted(best.items(), key=lambda item: item[1][0])
    if nodes is not None:
        ranked = [item for item in ranked if item[0] in set(nodes)]
    if top is not None:
        ranked = ranked[:top]
    return {node: proposal for node, (_, proposal) in ranked}


def start_for(spec, reference, own):
    """The common start: the reference fit mapped into `spec` by name, other parameters from `own`."""
    start = np.array(own, dtype=np.float64)
    reference = np.asarray(reference)
    column = {name: reference[:, i] for i, name in enumerate(BASE_NAMES)}
    for i, name in enumerate(spec.names):
        if "theta_pref" in spec.names:
            source, constant = PAPER_LINKS.get(name, (None, None))
            if source:
                start[:, i] = column[source]
            elif constant is not None:
                start[:, i] = constant
        elif name in column:
            start[:, i] = column[name]
    return start


def paired(cell_mse, count, base_mse, base_count):
    """Mean per-cell held-out MSE difference from a control, its standard error, and the score difference."""
    diff = np.asarray(cell_mse, dtype=float) - np.asarray(base_mse, dtype=float)
    error = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    mean = float(np.mean(diff))
    return {
        "mse_difference": mean,
        "mse_standard_error": error,
        "score_difference": mean * MSE_SCALE + (count - base_count) * PENALTY,
        "score_standard_error": error * MSE_SCALE,
    }


def refuse_mixed(first, second):
    """Raise unless two results were produced under the same protocol, cohort, starts, seed and budget."""
    different = [key for key in SAME_PROTOCOL if first.get(key) != second.get(key)]
    if different:
        raise ValueError(f"Cannot mix protocols: {', '.join(different)} differ.")


def _cell_errors(model, params, x, y):
    return np.asarray(
        jnp.mean((predict(model, jnp.asarray(x), jnp.asarray(params)) - jnp.asarray(y)) ** 2, axis=1)
    )


def evaluate(specs, cohort, protocol, *, max_iter, starts, seed, reference=BASELINE_ID):
    """Fit every spec on training trials only and score it on held-out trials under one protocol."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}: choose from {PROTOCOLS}.")
    x, y = np.asarray(cohort[0]), np.asarray(cohort[1])
    train, held = split_trials(x)
    xt, yt, xh, yh = x[:, train], y[:, train], x[:, held], y[:, held]
    budget = min(max_iter, fitting.LM_ITERATIONS)
    shared = None
    summary = None
    if protocol == "common-start":
        base = next(spec for spec in specs if spec.name == reference)
        shared = np.asarray(
            fit_training(base.model, base.estimator, xt, yt, budget, starts=starts, seed=seed)
        )
        summary = {
            "model": reference,
            "training_mse": float(np.mean(_cell_errors(base.model, shared, xt, yt))),
            "mse": float(np.mean(_cell_errors(base.model, shared, xh, yh))),
        }
    models = {}
    for spec in specs:
        try:
            own = np.stack(
                [np.asarray(spec.estimator(xt[i], yt[i]), dtype=np.float64) for i in range(len(xt))]
            )
            start = own if shared is None else start_for(spec, shared, own)
            params = np.asarray(
                fit_training(spec.model, None, xt, yt, budget, starts=starts, seed=seed, start=start)
            )
            cell_mse, cell_training = (
                _cell_errors(spec.model, params, xh, yh),
                _cell_errors(spec.model, params, xt, yt),
            )
            if not (np.isfinite(cell_mse).all() and np.isfinite(cell_training).all()):
                raise ValueError(f"{int((~np.isfinite(cell_mse)).sum())} cells have nonfinite fits")
        except (ValueError, FloatingPointError, OverflowError, KeyError, TypeError) as exc:
            models[spec.name] = {"kind": spec.kind, "error": str(exc)}
            continue
        mse = float(np.mean(cell_mse))
        count = len(spec.names)
        models[spec.name] = {
            "kind": spec.kind,
            "parameters": count,
            "mse": mse,
            "training_mse": float(np.mean(cell_training)),
            "score": mse * MSE_SCALE + count * PENALTY,
            "cell_mse": cell_mse.tolist(),
            "cell_training_mse": cell_training.tolist(),
        }
    return {"reference": summary, "models": models}


def report(result, controls=(BASELINE_ID, PAPER)):
    """A markdown table: each model's error and score, and its paired score difference from each control."""
    protocol = result["protocol"]
    models = result["models"]
    controls = [name for name in controls if name in models and "error" not in models[name]]
    lines = [
        f"Protocol {protocol['protocol']}, cohort {protocol['cohort']}, {protocol['starts']} starts, "
        f"seed {protocol['seed']}, at most {protocol['max_iter']} iterations. Lower is better; "
        "differences are paired by cell, negative means better than the control.",
        "",
        "| model | params | training MSE | held-out MSE | score | "
        + " | ".join(f"vs {name} (± SE)" for name in controls)
        + " |",
        "| --- | ---: | ---: | ---: | ---: |" + " ---: |" * len(controls),
    ]
    for name, row in models.items():
        if "error" in row:
            lines.append(f"| {name} | – | – | – | – |" + f" not fitted: {row['error']} |" * bool(controls))
            continue
        versus = []
        for control in controls:
            if control == name:
                versus.append("–")
                continue
            base = models[control]
            d = paired(row["cell_mse"], row["parameters"], base["cell_mse"], base["parameters"])
            versus.append(f"{d['score_difference']:+.4f} ± {d['score_standard_error']:.4f}")
        lines.append(
            f"| {name} | {row['parameters']} | {row['training_mse']:.4f} | {row['mse']:.4f} "
            f"| {row['score']:.4f} | " + " | ".join(versus) + " |"
        )
    return "\n".join(lines)


def _specs(up, proposals):
    seeds = up["seed_programs"]
    specs = [
        Spec(BASELINE_ID, "control", seeds.neuron_model_2_jax, seeds.parameter_estimator_2, BASE_NAMES),
        Spec(PAPER, "control", ai_model, ai_estimator, tuple(parameter_names(ai_model))),
    ]
    for node, proposal in proposals.items():
        equation = compile_equation(proposal)
        model, estimator = canonical(equation)
        specs.append(Spec(node, "candidate", model, estimator, equation.canonical_names))
    return specs


def run(args):
    """Evaluate the chosen candidates and both controls on one cohort and write the protocol and results."""
    output = Path(args.output)
    wave = json.loads(Path(args.wave).read_text())
    proposals = best_per_node(wave, top=args.top, nodes=args.nodes.split(",") if args.nodes else None)
    if not proposals:
        raise SystemExit("The run has no fitted equations to compare.")
    up = import_upstream(args.upstream.resolve())
    fitting.install(up["hypothesis_engine"], "lm-bounded")
    response, angles, sha = load_dataset(args.dataset, up["utils"].extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    rows = cohort_rows(groups, args.cohort)
    if args.cells:
        rows = rows[: args.cells]
    cohort = (np.asarray(x)[rows], np.asarray(y)[rows])
    protocol = {
        "protocol": args.protocol,
        "cohort": args.cohort,
        "starts": args.starts,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "fitter": "lm-bounded",
        "relative_step": True,
        "jitter": JITTER,
        "reference": BASELINE_ID if args.protocol == "common-start" else None,
        "trial_seed": TRIAL_SEED,
        "discovery_partition_seed": DISCOVERY_PARTITION_SEED,
        "dataset_sha256": sha,
        "cells": len(rows),
        "cell_ids": indices["eligible_cell_ids"][rows].tolist(),
        "wave": str(args.wave),
        "candidates": list(proposals),
    }
    if args.against:
        other = json.loads(Path(args.against).read_text())
        refuse_mixed(protocol, other["protocol"])
    result = evaluate(
        _specs(up, proposals),
        cohort,
        args.protocol,
        max_iter=args.max_iter,
        starts=args.starts,
        seed=args.seed,
    )
    result["protocol"] = protocol
    result["equations"] = proposals
    if args.against:
        result["models"] = {**other["models"], **result["models"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(report(result))

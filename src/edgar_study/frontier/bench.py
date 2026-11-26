"""A fitting benchmark that needs no ground truth, built on equations that contain one another.

When a child equation contains its parent (fixing some of the child's parameters gives the parent exactly),
the child's best training error is at most the parent's on every neuron. A fitter that leaves a child worse
than its parent has missed the child's optimum, and the gap is a lower bound on how far it missed. Each
fitter is also measured against the lowest error any saved fitter, or any ancestor, has reached.

Fitters see training trials only; every error is computed here, the same way for every fitter.
"""

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import (
    DATASET_SHA256,
    DISCOVERY_PARTITION_SEED,
    discovery_partition,
    load_dataset,
    prepare,
)
from edgar_study.frontier.grammar import Equation, compile_equation
from edgar_study.frontier.protocol import RELATIVE_STEP, STARTS, fit_equation_training, predict, split_trials
from edgar_study.frontier.warm import Parent, declared_probes, verified_map
from edgar_study.upstream import import_upstream

log = logging.getLogger(__name__)

FIXTURE = Path(__file__).with_name("nested.json")
TOLERANCE = 1e-3
CELLS = 64
TINY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class Edge:
    child: str
    parent: str
    embed: Callable | None


@dataclass(frozen=True)
class Fixture:
    equations: dict[str, Equation]
    sources: dict[str, str]
    edges: tuple[Edge, ...]

    def subset(self, ids: Iterable[str]) -> "Fixture":
        """The listed equations, in fixture order, with the edges between them."""
        keep = [i for i in self.equations if i in set(ids)]
        edges = tuple(e for e in self.edges if e.child in keep and e.parent in keep)
        return Fixture({i: self.equations[i] for i in keep}, {i: self.sources[i] for i in keep}, edges)


@dataclass(frozen=True)
class Fitter:
    """`fit(equation, xt, yt, warm)` returns (cells, parameters) in the equation's declared order.

    With `warm`, the benchmark passes each child the best start, per cell, that its fitted parents give
    through their proven maps; otherwise `warm` is None.
    """

    fit: Callable
    warm: bool = False


def _no_fit(cohort, max_iter):
    raise TypeError("Fixture parents are used only to verify maps.")


def load_fixture(path=FIXTURE) -> Fixture:
    """Compile the fixture and prove every edge: its map must reproduce the parent's predictions."""
    raw = json.loads(Path(path).read_text())
    equations = {
        row["id"]: compile_equation({"equation": row["equation"], "initial": row["initial"]})
        for row in raw["equations"]
    }
    sources = {row["id"]: row["source"] for row in raw["equations"]}
    edges = []
    for row in raw["edges"]:
        child, parent = equations[row["child"]], equations[row["parent"]]
        link = Parent(row["parent"], tuple(parent.initial), parent.model, declared_probes(parent), _no_fit)
        embed = verified_map(dataclasses.replace(child, nesting={row["parent"]: row["map"]}), link)
        if embed is None:
            raise ValueError(f"{row['child']} does not contain {row['parent']} through its map.")
        edges.append(Edge(row["child"], row["parent"], embed))
    return Fixture(equations, sources, tuple(edges))


def protocol_fitter(starts=STARTS, max_iter=fitting.LM_ITERATIONS, relative_step=RELATIVE_STEP, **options):
    """The frontier's fitting protocol on training trials, with its budget as options."""

    def fit(equation, xt, yt, warm):
        return fit_equation_training(
            equation, xt, yt, max_iter, starts=starts, warm=warm, relative_step=relative_step, **options
        )

    return fit


FITTERS = {
    "protocol": Fitter(protocol_fitter()),
    "warm": Fitter(protocol_fitter(), warm=True),
    "single": Fitter(protocol_fitter(starts=1)),
    **{
        f"pruned-{n}{suffix}": Fitter(protocol_fitter(prune_after=n), warm=bool(suffix))
        for n in (5, 10, 20)
        for suffix in ("", "-warm")
    },
    "absolute": Fitter(protocol_fitter(relative_step=False)),
    "absolute-warm": Fitter(protocol_fitter(relative_step=False), warm=True),
    "reference": Fitter(protocol_fitter(starts=32, max_iter=400, relative_step=True), warm=True),
}


def cell_errors(equation, params, x, y):
    """Each cell's mean squared error; nonfinite errors count as infinite."""
    fitted = np.asarray(predict(equation.model, jnp.asarray(x), jnp.asarray(params, dtype=jnp.float32)))
    errors = np.mean((fitted - np.asarray(y)) ** 2, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def _warm_start(equation, edges, fitted, xt, yt):
    candidates = [
        np.asarray(e.embed(fitted[e.parent]), dtype=np.float32) for e in edges if e.parent in fitted
    ]
    if not candidates:
        return None
    costs = np.stack([cell_errors(equation, c, xt, yt) for c in candidates])
    return np.stack(candidates)[np.argmin(costs, axis=0), np.arange(len(xt))]


def evaluate(fixture: Fixture, fitter: Fitter, cohort) -> dict:
    """Fit every fixture equation on the cohort's training trials; return per-cell errors and timings."""
    x, y = (np.asarray(a) for a in cohort)
    train, held = (np.asarray(i) for i in split_trials(x))
    xt, yt, xh, yh = x[:, train], y[:, train], x[:, held], y[:, held]
    fitted, training, held_out, seconds = {}, {}, {}, {}
    for name, equation in fixture.equations.items():
        parents = [e for e in fixture.edges if e.child == name]
        warm = _warm_start(equation, parents, fitted, xt, yt) if fitter.warm else None
        started = time.perf_counter()
        params = np.asarray(fitter.fit(equation, xt, yt, warm), dtype=np.float32)
        seconds[name] = time.perf_counter() - started
        fitted[name] = params
        training[name] = cell_errors(equation, params, xt, yt).tolist()
        held_out[name] = cell_errors(equation, params, xh, yh).tolist()
        log.info("%s: training MSE %.4f in %.1f s", name, np.mean(training[name]), seconds[name])
    return {"training": training, "held_out": held_out, "seconds": seconds}


def violations(edges: Iterable[Edge], errors: Mapping, tolerance=TOLERANCE) -> list[dict]:
    """Per edge, the cells where the child fits worse than its parent by more than `tolerance` (relative)."""
    rows = []
    for edge in edges:
        if edge.child not in errors or edge.parent not in errors:
            continue
        child, parent = np.asarray(errors[edge.child], float), np.asarray(errors[edge.parent], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            excess = np.nan_to_num(child / np.maximum(parent, TINY) - 1, nan=0.0)
        rows.append(
            {
                "child": edge.child,
                "parent": edge.parent,
                "violated": int(np.sum(child > parent * (1 + tolerance))),
                "cells": len(child),
                "worst": float(max(np.max(excess), 0.0)),
            }
        )
    return rows


def best_known(edges: Sequence[Edge], runs: Sequence[Mapping]) -> dict:
    """The lowest error reached for each equation by any run, or by any ancestor through the edges."""
    names = {name for run in runs for name in run}
    best = {
        name: np.min([np.asarray(run[name], float) for run in runs if name in run], axis=0) for name in names
    }
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.child in best and edge.parent in best:
                lower = np.minimum(best[edge.child], best[edge.parent])
                if np.any(lower < best[edge.child]):
                    best[edge.child], changed = lower, True
    return best


def regret(errors: Mapping, best: Mapping, near=TOLERANCE) -> dict:
    """Mean relative excess over the best known error, the share of cells within `near` of it, and the
    equation with the largest mean excess."""
    relative = {}
    for name, values in errors.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.asarray(values, float) / np.maximum(best[name], TINY) - 1
        relative[name] = np.nan_to_num(ratio, nan=np.inf)
    pooled = np.concatenate(list(relative.values()))
    return {
        "mean": float(np.mean(pooled)),
        "at_best": float(np.mean(pooled <= near)),
        "worst_equation": max(relative, key=lambda name: float(np.mean(relative[name]))),
        "by_equation": {name: float(np.mean(r)) for name, r in relative.items()},
    }


def save_result(directory: Path, name: str, result: Mapping) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(result))


def load_results(directory: Path) -> dict:
    """Every fitter result saved for one cohort, by fitter name."""
    if not directory.is_dir():
        return {}
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}


def report(fixture: Fixture, results: Mapping) -> str:
    """Markdown tables: one row per fitter, then violated cells per edge and mean regret per equation."""
    errors = {
        name: {k: np.asarray(v, float) for k, v in r["training"].items()} for name, r in results.items()
    }
    best = best_known(fixture.edges, list(errors.values()))
    lines = [
        "| Fitter | Seconds | Violations | Mean regret | At best | Worst edge |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    edges, regrets = {}, {}
    for name, result in results.items():
        rows = violations(fixture.edges, errors[name])
        edges[name] = {(row["parent"], row["child"]): row["violated"] for row in rows}
        regrets[name] = regret(errors[name], best)
        worst = max(rows, key=lambda row: (row["violated"], row["worst"]), default=None)
        label = (
            f"{worst['parent']} ⊂ {worst['child']} ({worst['violated']})"
            if worst and worst["violated"]
            else "none"
        )
        lines.append(
            f"| {name} | {sum(result['seconds'].values()):.1f} | "
            f"{sum(row['violated'] for row in rows)} of {sum(row['cells'] for row in rows)} | "
            f"{regrets[name]['mean']:.2%} | {regrets[name]['at_best']:.0%} | {label} |"
        )
    names = list(results)
    lines += ["", "| Edge (violated cells) | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for edge in fixture.edges:
        counts = [str(edges[n].get((edge.parent, edge.child), "")) for n in names]
        lines.append(f"| {edge.parent} ⊂ {edge.child} | " + " | ".join(counts) + " |")
    lines += [
        "",
        "| Equation (mean regret) | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for equation in fixture.equations:
        cells = [
            f"{regrets[n]['by_equation'][equation]:.2%}" if equation in regrets[n]["by_equation"] else ""
            for n in names
        ]
        lines.append(f"| {equation} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cohort_id(cells: int) -> str:
    """A name for the benchmark cohort: the first `cells` exploration cells of the pinned recording."""
    key = f"{DATASET_SHA256}:{DISCOVERY_PARTITION_SEED}:{cells}"
    return f"cells{cells}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _cohort(args, directory):
    path = directory / "cohort.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["x"], saved["y"]
    upstream = import_upstream(args.upstream.resolve())
    response, angles, _ = load_dataset(args.dataset, upstream["utils"].extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")
    explore, _ = discovery_partition(groups)
    cells = explore[: args.cells]
    x, y = np.asarray(x)[cells], np.asarray(y)[cells]
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y)
    return x, y


def run(args):
    """`edgar fitbench`: run the named fitters on the cohort and print the report for every saved fitter."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = [name for name in args.fitters.split(",") if name]
    unknown = sorted(set(names) - set(FITTERS))
    if unknown:
        raise SystemExit(f"Unknown fitters {unknown}; choose from {sorted(FITTERS)}.")
    if args.cells < 1:
        raise SystemExit("--cells must be positive.")
    fixture = load_fixture()
    directory = args.output / cohort_id(args.cells)
    if names:
        cohort = _cohort(args, directory)
        for name in names:
            log.info("Fitting %d equations with %s on %d cells", len(fixture.equations), name, len(cohort[0]))
            save_result(directory, name, evaluate(fixture, FITTERS[name], cohort))
    print(report(fixture, load_results(directory)))

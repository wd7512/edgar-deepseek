"""Search tuning-curve networks with NEAT and score them with EDGAR's objective.

Each genome decodes to a `neuron_model` and a `parameter_estimator` (see
`edgar_study.neat_programs`), so a candidate is fitted and scored by the same
`hypothesis_engine.objective` an LLM-written program uses: 10 x held-out MSE plus 0.01 per fitted
weight. Fitness is the negative score.

The search runs on the discovery cells only. `select_and_score` then follows `rescore`'s rule:
the top unique genomes by discovery score are scored on the selection cells, and the winner and
the two seeds are scored once on the locked cells.
"""

import contextlib
import hashlib
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import neat
import numpy as np

from edgar_study import fitting, neat_programs
from edgar_study.upstream import PENALTY, import_upstream, score_program

INVALID_FITNESS = -1e6
FINAL_MAX_ITER = 2_000
log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    records: list[dict[str, Any]]
    generations: list[dict[str, Any]]
    attempts: int
    evaluations: int
    failed: int
    best: dict[str, Any] | None


def search(
    objective,
    loss_func,
    x,
    y,
    *,
    population_size,
    generations,
    seed=0,
):
    """Run NEAT on (x, y) with `objective` as the scorer and return every evaluated genome."""
    config = neat_programs.make_config(population_size)
    population = neat.Population(config, seed=seed)
    np.random.seed(seed)

    records = []
    rows = []
    cache = {}
    state = {"generation": 0, "evaluations": 0, "failed": 0}

    def evaluate(genomes, _config):
        generation = state["generation"]
        scores = []
        for genome_id, genome in genomes:
            program = neat_programs.decode(genome, config)
            key = hashlib.sha256((program.model_code + program.estimator_code).encode()).hexdigest()
            if key not in cache:
                cache[key] = _score(program, objective, loss_func, x, y, seed)
                state["evaluations"] += 1
            score = cache[key]
            genome.fitness = INVALID_FITNESS if score is None else -score
            if score is None:
                state["failed"] += 1
            scores.append(score)
            records.append(
                {
                    "generation": generation,
                    "genome": genome_id,
                    "code_key": key,
                    "score": score,
                    "n_params": program.n_params,
                    "hidden_nodes": program.hidden_nodes,
                    "connections": program.connections,
                    "activations": list(program.activations),
                    "model_code": program.model_code,
                    "estimator_code": program.estimator_code,
                }
            )
        valid = [score for score in scores if score is not None]
        rows.append(
            {
                "generation": generation,
                "attempts": len(genomes),
                "scored": len(valid),
                "best": min(valid) if valid else None,
                "mean": float(np.mean(valid)) if valid else None,
                "species": len(population.species.species),
            }
        )
        if valid:
            log.info(
                "generation %d: best %.4f, mean %.4f, %d scored, %d species",
                generation,
                rows[-1]["best"],
                rows[-1]["mean"],
                rows[-1]["scored"],
                rows[-1]["species"],
            )
        state["generation"] += 1

    with contextlib.redirect_stdout(io.StringIO()):
        population.run(evaluate, generations)

    scored = [record for record in records if record["score"] is not None]
    best = min(scored, key=lambda record: record["score"]) if scored else None
    return SearchResult(
        records=records,
        generations=rows,
        attempts=len(records),
        evaluations=state["evaluations"],
        failed=state["failed"],
        best=best,
    )


def _score(program, objective, loss_func, x, y, seed) -> float | None:
    try:
        model = neat_programs.load(program.model_code, "neuron_model")
        estimator = neat_programs.load(program.estimator_code, "parameter_estimator")
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, score, _ = objective(
                model,
                estimator,
                loss_func,
                x,
                y,
                param_penalty_weight=PENALTY,
                random_seed=0,
                max_iter=1_000,
                tol=1e-6,
            )
    except Exception as error:
        log.info("program failed: %s", error)
        return None
    return float(score) if np.isfinite(score) else None


def top_records(records, top) -> list[dict[str, Any]]:
    """The best `top` distinct programs by discovery score."""
    seen = set()
    chosen = []
    for record in sorted(
        records,
        key=lambda record: record["score"] if record["score"] is not None else float("inf"),
    ):
        if record["score"] is None or record["code_key"] in seen:
            continue
        seen.add(record["code_key"])
        chosen.append(record)
        if len(chosen) == top:
            break
    return chosen


def select_and_score(upstream, result, x, y, groups, top=10, fitter=None):
    """Score the top programs on selection cells, then the winner once on locked cells."""
    seeds = upstream["seed_programs"]
    candidates: list[dict[str, Any]] = [
        {
            "id": f"seed_{index}",
            "generation": -1,
            "model": getattr(seeds, f"neuron_model_{index}_jax"),
            "estimator": getattr(seeds, f"parameter_estimator_{index}"),
        }
        for index in (1, 2)
    ]
    for position, record in enumerate(top_records(result.records, top), start=1):
        candidates.append(
            {
                "id": f"neat_{position}",
                "generation": record["generation"],
                "discovery_score": record["score"],
                "n_params": record["n_params"],
                "model": neat_programs.load(record["model_code"], "neuron_model"),
                "estimator": neat_programs.load(record["estimator_code"], "parameter_estimator"),
                "model_code": record["model_code"],
                "estimator_code": record["estimator_code"],
            }
        )

    def score(candidate, cells) -> dict[str, Any] | None:
        if candidate["model"] is None or candidate["estimator"] is None:
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            total, mse, count, _ = score_program(
                upstream,
                candidate["model"],
                candidate["estimator"],
                x[cells],
                y[cells],
                FINAL_MAX_ITER,
            )
        if not np.isfinite(total):
            return None
        return {"score": float(total), "mse": float(mse), "parameters": int(count)}

    for candidate in candidates:
        candidate["selection"] = score(candidate, groups["selection"])

    learned = [candidate for candidate in candidates if candidate["generation"] >= 0]
    winner = min(
        (candidate for candidate in learned if candidate["selection"] is not None),
        key=lambda candidate: candidate["selection"]["score"],
        default=None,
    )
    for candidate in candidates[:2] + ([winner] if winner else []):
        candidate["locked_test"] = score(candidate, groups["locked_test"])

    single, double = candidates[:2]
    for candidate in candidates:
        candidate.pop("model", None)
        candidate.pop("estimator", None)
    return {
        "fitter": fitter,
        "selection_rule": "lowest selection-cell penalized score among the top "
        f"{top} unique NEAT programs by discovery score",
        "cell_counts": {name: int(len(cells)) for name, cells in groups.items()},
        "winner": winner,
        "double_gaussian": double,
        "single_gaussian": single,
        "delta_locked_test_vs_double_gaussian": (
            double["locked_test"]["score"] - winner["locked_test"]["score"]
            if winner and winner["locked_test"] and double["locked_test"]
            else None
        ),
        "candidates": candidates,
    }


def run(args):
    """Run a NEAT search on the discovery cells, then the strict holdout, and write the run directory."""
    from edgar_study.datasets import load_dataset, prepare

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    upstream = import_upstream(args.upstream.resolve())
    fitter = fitting.install(upstream["hypothesis_engine"], args.fitter or "auto")
    response, angles, sha256 = load_dataset(args.dataset, upstream["utils"].extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")

    rows = groups["discovery"]
    if args.cell_fraction < 1:
        rows = rows[: max(4, round(args.cell_fraction * len(rows)))]

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = (args.results / "neat" / run_id).resolve()
    run_dir.mkdir(parents=True)
    started = time.time()

    result = search(
        upstream["hypothesis_engine"].objective,
        upstream["loss_functions"].quadratic_loss,
        x[rows],
        y[rows],
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )
    rescore = select_and_score(upstream, result, x, y, groups, top=args.top, fitter=fitter)

    manifest = {
        "run_id": run_id,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256,
        "protocol": "strict-holdout-v1",
        "fitter": fitter,
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "cell_fraction": args.cell_fraction,
        "search_cells": int(len(rows)),
        "cell_counts": {name: int(len(cells)) for name, cells in groups.items()},
        "attempts": result.attempts,
        "evaluations": result.evaluations,
        "failed": result.failed,
        "best_discovery_score": result.best["score"] if result.best else None,
        "started": started,
        "finished": time.time(),
    }
    manifest["seconds"] = manifest["finished"] - started

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    with (run_dir / "programs.jsonl").open("w") as stream:
        for record in result.records:
            stream.write(json.dumps(record, default=float) + "\n")
    (run_dir / "search.json").write_text(
        json.dumps(
            {
                "generations": result.generations,
                "attempts": result.attempts,
                "evaluations": result.evaluations,
                "failed": result.failed,
                "best": result.best,
                "top": top_records(result.records, args.top),
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    (run_dir / "rescore.json").write_text(json.dumps(rescore, indent=2, default=float) + "\n")
    if rescore["winner"]:
        (run_dir / "winner.py").write_text(
            rescore["winner"]["model_code"] + "\n" + rescore["winner"]["estimator_code"]
        )
    print(f"Run directory: {run_dir}")
    return run_dir

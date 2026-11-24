"""The upstream EDGAR search with genetic programming in place of the LLM: the null baseline.

Upstream's `hypothesis_engine.main` runs unchanged (seeds, islands, selection, migration, fitting,
final databases). Only its three LLM stages are replaced at runtime:
  * generate_new_neuron_model -> crossover and mutation of the parents' expression trees
  * translate_to_jax -> the same tree written in JAX
  * generate_new_parameter_estimator -> the generic estimator built from the tree's parameter kinds
"""

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np

from edgar_study import fitting, symbolic, timing
from edgar_study.datasets import ACTIVITY_THRESHOLD, CONCENTRATION_THRESHOLD
from edgar_study.engine import discovery_subset, git_sha
from edgar_study.upstream import PENALTY, code_key, import_upstream

log = logging.getLogger(__name__)

PROPOSER = "GP (no LLM)"


def seed_code(seed_programs, which):
    """The NumPy source upstream stores for seed program `which` (1 or 2)."""
    function = getattr(seed_programs, f"neuron_model_{which}")
    source = inspect.getsource(function).replace(f"def neuron_model_{which}(", f"def neuron_model_v{which}(")
    return "import numpy as np \n" + source


class Proposer:
    """Stands in for upstream's three LLM stages, remembering the tree behind every program it writes."""

    def __init__(self, seed_programs, seed, translations_path=None, utils=None):
        self.rng = np.random.default_rng(seed)
        self.trees = {seed_code(seed_programs, i + 1): tree for i, tree in enumerate(symbolic.SEEDS)}
        self.translations_path = translations_path
        self.utils = utils

    def _compile(self, source, name):
        if self.utils is not None:
            return self.utils.str_to_func(source, name)
        namespace = {}
        exec(source, namespace)
        return namespace[name]

    def child(self, parents, mode):
        trees = [self.trees[code] for code in parents]
        for _ in range(10):
            base, donor = (trees[0], trees[-1]) if self.rng.random() < 0.5 else (trees[-1], trees[0])
            tree = symbolic.crossover(self.rng, base, donor) if self.rng.random() < 0.5 else base
            steps = int(self.rng.integers(1, 4)) if mode == "explore" else 1
            tree = symbolic.mutate(self.rng, tree, steps=steps)
            if symbolic.to_numpy_source(tree) not in self.trees:
                break
        return tree

    async def neuron_model(
        self,
        current_island,
        llm_name,
        client,
        spike_matrix,
        stimuli,
        mode="explore",
        k_max=2,
        temp=1,
        thinking_budget=1,
        img_dir=None,
    ):
        k = min(k_max, len(current_island))
        parents = current_island.sample(k, replace=False).reset_index(drop=True)
        parents = parents.sort_values(by="train_loss", ascending=False).reset_index(drop=True)
        ids = tuple(
            (parents["iteration_number"][i], parents["birth_island"][i], parents["batch_index"][i])
            for i in range(k)
        )
        tree = self.child(list(parents["program_code_string"]), mode)
        code = symbolic.to_numpy_source(tree)
        self.trees[code] = tree
        return code, symbolic.to_infix(tree), (ids[0], ids[-1])

    async def translate(self, code_string, client, llm_name="GP"):
        if code_string is None:
            return None, None
        jax_code = symbolic.to_jax_source(self.trees[code_string])
        if self.translations_path is not None:
            with open(self.translations_path, "a") as stream:
                record = {
                    "numpy_sha256": code_key(code_string),
                    "numpy_code": code_string,
                    "jax_code": jax_code,
                    "compiled": True,
                }
                stream.write(json.dumps(record) + "\n")
        return jax_code, self._compile(jax_code, "neuron_model")

    async def estimator(self, current_island, neuron_model_code_string, llm_name, client, *args, **kwargs):
        if neuron_model_code_string is None:
            return None, None
        source = symbolic.estimator_source(self.trees[neuron_model_code_string])
        return source, self._compile(source, "parameter_estimator")


def install(hypothesis_engine, proposer):
    """Replace upstream's LLM stages with `proposer`; patches module attributes."""
    hypothesis_engine.generate_new_neuron_model = proposer.neuron_model
    hypothesis_engine.translate_to_jax = proposer.translate
    hypothesis_engine.generate_new_parameter_estimator = proposer.estimator
    hypothesis_engine.genai.Client = lambda *args, **kwargs: None


def generations_completed(run_dir):
    return len(list(run_dir.glob("program_databases/*/*/iteration_updates/iteration_*")))


def run(args):
    """Run upstream's search with GP proposals inside a fresh run directory and return it."""
    args.dataset = Path(args.dataset).resolve()
    upstream = import_upstream(Path(args.upstream).resolve())
    hypothesis_engine = upstream["hypothesis_engine"]
    fitter = fitting.install(hypothesis_engine, args.fitter or "auto")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-gp" + uuid.uuid4().hex[:4]
    run_dir = (Path(args.results) / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True)
    proposer = Proposer(
        upstream["seed_programs"], args.seed, run_dir / "translations.jsonl", upstream["utils"]
    )
    install(hypothesis_engine, proposer)
    timings = timing.Timings(run_dir / "timings.jsonl")
    timing.install(hypothesis_engine, timings)

    manifest = {
        "run_id": run_id,
        "method": "gp",
        "seed": args.seed,
        "repo_sha": git_sha(Path(__file__).resolve().parents[2]),
        "upstream_commit": upstream["commit"],
        "dataset": str(args.dataset),
        "iterations": args.iterations,
        "time_limit_min": args.time_limit,
        "models": {"translator": PROPOSER, "proposer": PROPOSER, "estimator": PROPOSER},
        "use_image_feedback": False,
        "fitter": fitter,
        "islands": args.islands,
        "batch_size": args.batch_size,
        "maxsize": symbolic.MAXSIZE,
        "started": time.time(),
    }
    if args.cell_fraction < 1:
        data_path = run_dir / "search_cells.npy"
        cells = discovery_subset(
            args.dataset, upstream["utils"].extract_stimulus_related_response, args.cell_fraction, data_path
        )
        manifest.update(cell_fraction=args.cell_fraction, search_cell_ids=cells.tolist())
    else:
        data_path = args.dataset
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {run_dir}")

    status = "completed"
    previous = Path.cwd()
    os.chdir(run_dir)
    np.random.seed(args.seed)
    ring = [(island + 1) % args.islands for island in range(args.islands)]
    try:
        asyncio.run(
            hypothesis_engine.main(
                n_iterations=args.iterations,
                time_limit=args.time_limit,
                n_islands=args.islands,
                batch_size=args.batch_size,
                exploration_topology=ring,
                exploitation_topology=ring,
                use_image_feedback=False,
                use_large_every=0,
                param_penalty_weight=PENALTY,
                tiny_lm_name=PROPOSER,
                little_lm_name=PROPOSER,
                large_lm_name=PROPOSER,
                activity_thresh=ACTIVITY_THRESHOLD,
                conc_thresh=CONCENTRATION_THRESHOLD,
                data_path=str(data_path),
            )
        )
    except BaseException as error:
        status = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        os.chdir(previous)
        finished = time.time()
        completed = generations_completed(run_dir)
        manifest.update(
            status=status,
            finished=finished,
            spend_before_usd=0.0,
            spend_after_usd=0.0,
            generations_completed=completed,
            stopped_early=completed < args.iterations,
            timings=timing.summarise(timings.records, manifest["started"], finished),
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir

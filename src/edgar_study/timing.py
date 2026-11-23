"""Wall-clock timing of each stage of an upstream search, recorded by wrapping upstream's functions.

Upstream runs the LLM stages concurrently, so a stage's `busy_s` (the sum of its call durations) can
exceed its `wall_s` (the time during which at least one of its calls was running). `other_s` is the
run time covered by no timed stage: data loading, logging, census writes and similar.
"""

import functools
import inspect
import json
import time

FINAL_REFIT_MIN_ITERATIONS = 2_000

GENETIC_HELPERS = (
    "perform_island_deduplication",
    "perform_population_pruning",
    "perform_probabilistic_migration",
    "remove_duplicates",
)
PLOTS = ("plot_model_fits", "plot_single_model_fit", "plot_train_vs_test_loss")


class Timings:
    """Collects (stage, start, end) records and appends each to a JSON-lines file if `path` is set."""

    def __init__(self, path=None):
        self.path = path
        self.records = []

    def add(self, stage, start, end):
        record = {"stage": stage, "start": start, "end": end}
        self.records.append(record)
        if self.path is not None:
            with open(self.path, "a") as stream:
                stream.write(json.dumps(record) + "\n")

    def wrap(self, function, stage):
        """Return `function` timed under `stage`, a name or a callable of (args, kwargs) giving one."""
        label = stage if callable(stage) else lambda args, kwargs: stage

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def timed_async(*args, **kwargs):
                start = time.time()
                try:
                    return await function(*args, **kwargs)
                finally:
                    self.add(label(args, kwargs), start, time.time())

            return timed_async

        @functools.wraps(function)
        def timed(*args, **kwargs):
            start = time.time()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(label(args, kwargs), start, time.time())

        return timed


def _objective_stage(args, kwargs):
    return "final_refit" if kwargs.get("max_iter", 0) >= FINAL_REFIT_MIN_ITERATIONS else "fit"


def install(hypothesis_engine, timings):
    """Time upstream's LLM, fitting, plotting and population stages; patches module attributes."""
    engine = hypothesis_engine
    engine.objective = timings.wrap(engine.objective, _objective_stage)
    for name, stage in (
        ("generate_new_neuron_model", "proposal"),
        ("translate_to_jax", "translation"),
        ("generate_new_parameter_estimator", "estimator"),
        ("compute_evaluation_matrix", "evaluation_matrix"),
    ):
        setattr(engine, name, timings.wrap(getattr(engine, name), stage))
    for name in PLOTS:
        setattr(engine.diagnostic, name, timings.wrap(getattr(engine.diagnostic, name), "plots"))
    for name in GENETIC_HELPERS:
        setattr(
            engine.genetic_helpers, name, timings.wrap(getattr(engine.genetic_helpers, name), "population")
        )


def _union(intervals):
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def summarise(records, started, finished):
    """Per-stage calls, busy and wall seconds, plus the run time no stage covers."""
    stages = {}
    for record in records:
        stages.setdefault(record["stage"], []).append((record["start"], record["end"]))
    summary = {
        stage: {
            "calls": len(intervals),
            "busy_s": round(sum(end - start for start, end in intervals), 3),
            "wall_s": round(_union(intervals), 3),
        }
        for stage, intervals in stages.items()
    }
    covered = _union([interval for intervals in stages.values() for interval in intervals])
    total = finished - started
    return {"total_s": round(total, 3), "stages": summary, "other_s": round(total - covered, 3)}

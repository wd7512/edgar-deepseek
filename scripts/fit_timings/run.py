"""Time how long each paper curve takes to go through `hypothesis_engine.objective` under each fitter.

Every call is split into estimator time, JAX tracing and compilation time, and the rest (upstream's
checks, the fit and scoring). The fitters build a new jitted function on every call, so a search
pays compilation for every program it scores; it is reported, not excluded.

Run with `uv run scripts/fit_timings/run.py --size small` and record results in `results.md`.
"""

import argparse
import contextlib
import datetime
import io
import logging
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import monitoring

from edgar_study import curves, fitting
from edgar_study.upstream import PENALTY, import_upstream

log = logging.getLogger("fit_timings")

ROOT = Path(__file__).resolve().parents[2]
SIZES = {
    "small": {"n_cells": 16, "n_trials": 144, "steps": 100},
    "real": {"n_cells": 1_055, "n_trials": 428, "steps": 1_000},
}
FITTERS = ("upstream", "adam", "lm", "lm-bounded")
COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)


class CompileClock:
    """Running total of the seconds JAX reports spending on tracing and compilation."""

    def __init__(self):
        self.seconds = 0.0
        monitoring.register_event_duration_secs_listener(self._record)

    def _record(self, event, duration, **_):
        if event in COMPILE_EVENTS:
            self.seconds += duration


class StopLog(logging.Handler):
    """Keeps the iteration count and converged cells that `lm-bounded` logs for its last fit."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.last = None
        logging.getLogger(fitting.__name__).addHandler(self)

    def emit(self, record):
        if hasattr(record, "iterations"):
            self.last = record


def timed_estimator(estimator, spent):
    def estimate(theta, counts):
        start = time.perf_counter()
        try:
            return estimator(theta, counts)
        finally:
            spent.append(time.perf_counter() - start)

    return estimate


def time_fit(objective, curve, loss, x, y, steps, clock):
    """One call of `objective`, with its wall time split into estimator, compile and the rest."""
    spent = []
    compile_start = clock.seconds
    np.random.seed(0)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, score, params = objective(
            curve.model,
            timed_estimator(curve.estimator, spent),
            loss,
            x,
            y,
            param_penalty_weight=PENALTY,
            max_iter=steps,
        )
    total = time.perf_counter() - start
    compile_seconds = clock.seconds - compile_start
    estimator_seconds = sum(spent)
    return {
        "total": total,
        "estimator": estimator_seconds,
        "compile": compile_seconds,
        "rest": total - estimator_seconds - compile_seconds,
        "score": score,
        "params": params,
    }


def benchmark(fitters, selected, n_cells, n_trials, steps, lm_iterations, repeats):
    """One row per (fitter, curve) with the median of `repeats` timed calls."""
    with contextlib.redirect_stdout(io.StringIO()):
        modules = import_upstream(ROOT)
    engine, loss = modules["hypothesis_engine"], modules["loss_functions"].quadratic_loss
    x, y = curves.synthetic_cells(n_cells, n_trials)
    clock = CompileClock()
    stops = StopLog()
    time_fit(engine.objective, selected[0], loss, x[:2], y[:2], 1, clock)
    rows = []
    for fitter in fitters:
        objective = (
            engine.objective
            if fitter == "upstream"
            else fitting.make_objective(engine.objective, fitter, lm_iterations)
        )
        for curve in selected:
            stops.last = None
            calls = [time_fit(objective, curve, loss, x, y, steps, clock) for _ in range(repeats)]
            last = calls[-1]
            if stops.last is not None:
                stop = stops.last.__dict__
                iterations = f"{stop['iterations']} ({stop['converged']}/{stop['cells']} converged)"
            else:
                iterations = lm_iterations if fitter == "lm" else steps
            row = {
                "program": curve.name,
                "paper_id": curve.paper_id,
                "params": curve.n_params,
                "fitter": fitter,
                "iterations": iterations,
                **{
                    k: statistics.median(c[k] for c in calls)
                    for k in ("total", "estimator", "compile", "rest")
                },
                "train_loss": fitting.train_loss(curve.model, loss, last["params"], x, y),
                "score": last["score"],
            }
            log.info(
                "%s %s: %.2fs total, %.2fs compile, %.2fs rest",
                fitter,
                curve.name,
                row["total"],
                row["compile"],
                row["rest"],
            )
            rows.append(row)
    return rows


def table(rows):
    lines = [
        "| Program | Paper ID | Params | Fitter | Iterations | Total (s) | Estimator (s) | Compile (s) "
        "| Checks + fit (s) | Train loss | Score |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['program']} | {r['paper_id']} | {r['params']} | {r['fitter']} | {r['iterations']} | "
            f"{r['total']:.2f} | {r['estimator']:.2f} | {r['compile']:.2f} | {r['rest']:.2f} | "
            f"{r['train_loss']:.4f} | {r['score']:.4f} |"
        )
    return "\n".join(lines)


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def header(args, n_cells, n_trials, steps):
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    command = "uv run scripts/fit_timings/run.py " + shlex.join(sys.argv[1:])
    return "\n".join(
        [
            f"## {datetime.date.today().isoformat()}, `{args.size}`, commit `{commit}`",
            "",
            f"- Command: `{command.strip()}`",
            f"- Data: {n_cells} synthetic cells × {n_trials} trials, drawn from the winner (H27)",
            f"- Adam steps: {steps}; LM iterations: {args.lm_iterations}; median of {args.repeats} call(s)",
            f"- Host: {cpu_name()}, {os.cpu_count()} cores, "
            f"{len(jax.devices())} JAX {jax.default_backend()} device(s), JAX {jax.__version__}",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--cells", type=int, help="override the size's cell count")
    parser.add_argument("--trials", type=int, help="override the size's trial count")
    parser.add_argument("--steps", type=int, help="override the size's Adam step count")
    parser.add_argument("--lm-iterations", type=int, default=fitting.LM_ITERATIONS)
    parser.add_argument("--fitters", nargs="+", choices=FITTERS, default=list(FITTERS))
    parser.add_argument("--curves", nargs="+", choices=[c.name for c in curves.CURVES])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    size = SIZES[args.size]
    n_cells = args.cells or size["n_cells"]
    n_trials = args.trials or size["n_trials"]
    steps = args.steps or size["steps"]
    selected = [c for c in curves.CURVES if not args.curves or c.name in args.curves]
    rows = benchmark(args.fitters, selected, n_cells, n_trials, steps, args.lm_iterations, args.repeats)
    print(header(args, n_cells, n_trials, steps) + "\n\n" + table(rows))


if __name__ == "__main__":
    main()

"""The pinned upstream EDGAR code, its scoring convention and its output layout."""

import hashlib
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "0943922dcb73b12dbee8e440b3e1ff700cb49c10"
UPSTREAM_MODULES = (
    "diagnostic",
    "genetic_helpers",
    "hypothesis_engine",
    "loss_functions",
    "seed_programs",
    "utils",
)
UPSTREAM_DIR = Path("src/edgar_upstream")
PENALTY = 0.01  # score per free parameter, as in the paper and upstream main()
MSE_SCALE = 10  # loss_functions.quadratic_loss reports 10 × MSE


def import_upstream(checkout):
    """Import the six upstream modules from `src/edgar_upstream/` after checking they match the pinned commit byte for byte."""
    folder = checkout / UPSTREAM_DIR
    for name in UPSTREAM_MODULES:
        path = folder / f"{name}.py"
        expected = subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{UPSTREAM_COMMIT}:{path.name}"]
        )
        if path.read_bytes() != expected:
            raise ValueError(f"{path.name} differs from pinned upstream.")
    sys.path.insert(0, str(folder))
    import hypothesis_engine
    import loss_functions
    import seed_programs
    import utils

    return {
        "commit": UPSTREAM_COMMIT,
        "hypothesis_engine": hypothesis_engine,
        "loss_functions": loss_functions,
        "seed_programs": seed_programs,
        "utils": utils,
    }


def score_program(upstream, model, estimator, x, y, max_iter):
    """Fit and score one program the way upstream does. Returns (score, mse, parameter count, params)."""
    import numpy as np

    np.random.seed(0)  # upstream samples cells for its compatibility checks with NumPy's RNG
    _, _, score, params = upstream["hypothesis_engine"].objective(
        model,
        estimator,
        upstream["loss_functions"].quadratic_loss,
        x,
        y,
        random_seed=0,
        max_iter=max_iter,
        tol=1e-6,
        param_penalty_weight=PENALTY,
    )
    count = int(params.shape[1])
    return float(score), (float(score) - PENALTY * count) / MSE_SCALE, count, params


def code_key(code):
    """Key linking a proposed NumPy program to its JAX translation."""
    return hashlib.sha256(code.encode()).hexdigest()


def combined_file(run_dir, name):
    """A file in upstream's final `program_databases/<date>/<time>/combined/` directory, or None."""
    found = sorted(run_dir.glob(f"program_databases/*/*/combined/{name}"))
    if len(found) > 1:
        raise SystemExit(f"Expected one {name} under {run_dir}, found {len(found)}.")
    return found[0] if found else None


def searched_programs(run_dir):
    """Every learned program from the final island databases, best discovery loss first, one row per code.

    Upstream's combined database drops near-duplicates by `test_loss`, which is scored on the
    selection and locked-test cells, so picking candidates from it would leak those cells. The
    island databases are deduplicated on discovery loss only. `test_loss` is dropped so nothing
    downstream can use it.
    """
    import pandas as pd

    found = sorted(Path(run_dir).glob("program_databases/*/*"))
    islands = sorted(found[-1].glob("island_*/programs_db.csv")) if found else []
    if not islands:
        raise SystemExit(f"No island programs_db.csv under {run_dir}; has the run finished?")
    frame = pd.concat([pd.read_csv(path) for path in islands], ignore_index=True)
    frame = frame[frame["iteration_number"] >= 0].drop(columns="test_loss")
    return (
        frame.sort_values("train_loss", kind="stable")
        .drop_duplicates("program_code_string")
        .reset_index(drop=True)
    )

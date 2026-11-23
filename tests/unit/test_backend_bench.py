import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[2] / "scripts" / "backend_bench"
sys.path.insert(0, str(BENCH))

import common  # noqa: E402
import fit_numpy  # noqa: E402


def synthetic(name="double-gaussian", cells=4, trials=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, (cells, trials)).astype(np.float32)
    if name == "winner":
        truth = np.stack(
            [
                rng.uniform(0, 2 * np.pi, cells),
                rng.uniform(0, 2 * np.pi, cells),
                rng.uniform(0.05, 0.2, cells),
                rng.uniform(0.6, 1.2, cells),
                rng.uniform(0.2, 0.8, cells),
                rng.uniform(0.2, 0.6, cells),
                rng.uniform(0.2, 0.6, cells),
                rng.uniform(0.2, 0.6, cells),
                rng.uniform(0.2, 0.6, cells),
                rng.uniform(1.0, 2.0, cells),
                rng.uniform(1.0, 2.0, cells),
            ],
            axis=1,
        ).astype(np.float32)
    else:
        truth = np.stack(
            [
                rng.uniform(0, 2 * np.pi, cells),
                rng.uniform(0.05, 0.2, cells),
                rng.uniform(0.6, 1.2, cells),
                rng.uniform(0.2, 0.8, cells),
                rng.uniform(0.2, 0.6, cells),
            ],
            axis=1,
        ).astype(np.float32)
    y = common.predict(name, x, truth) + rng.normal(0, 0.02, (cells, trials)).astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32), truth


@pytest.mark.parametrize("name", ["double-gaussian", "winner"])
def test_bounded_lm_reduces_the_cost_and_converges_every_cell(name):
    x, y, truth = synthetic(name)
    start = truth + 0.15
    params, iterations, converged = fit_numpy.fit(name, x, y, start, True, max_iterations=60)
    before = np.sum((common.predict(name, x, start) - y) ** 2)
    after = np.sum((common.predict(name, x, params) - y) ** 2)
    assert after < before * 0.1
    assert converged == x.shape[0]
    assert 1 <= iterations <= 60


def test_score_of_a_perfect_fit_is_only_the_parameter_penalty():
    x, _, truth = synthetic()
    y = common.predict("double-gaussian", x, truth)
    assert common.score("double-gaussian", truth, x, y, 5) == pytest.approx(0.05)

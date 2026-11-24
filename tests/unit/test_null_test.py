from pathlib import Path

import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.null_test import cell_losses, paired_bootstrap
from edgar_study.upstream import PENALTY, import_upstream, score_program


def test_cell_losses_average_to_the_upstream_score():
    upstream = import_upstream(Path(".").resolve())
    fitting.install(upstream["hypothesis_engine"], "lm-bounded")
    seeds = upstream["seed_programs"]
    x, y, _, indices = prepare(*synthetic_recording(), "strict-holdout-v1")
    model, estimator = seeds.neuron_model_2_jax, seeds.parameter_estimator_2

    score, _, count, params = score_program(upstream, model, estimator, x, y, 200)
    losses = cell_losses(model, params, x, y, indices["score_trial_columns"])

    assert losses.shape == (y.shape[0],)
    assert losses.mean() + PENALTY * count == pytest.approx(score, rel=1e-5)


def test_bootstrap_interval_covers_zero_for_identical_losses():
    losses = np.random.default_rng(0).gamma(2.0, 1.0, 500)

    result = paired_bootstrap(losses, losses.copy())

    assert result["delta"] == 0
    assert result["low"] <= 0 <= result["high"]


def test_bootstrap_detects_a_clear_improvement_and_adds_the_parameter_penalty():
    rng = np.random.default_rng(1)
    base = rng.gamma(2.0, 1.0, 500)
    better = base - 0.2 + rng.normal(0, 0.05, 500)

    result = paired_bootstrap(better, base, penalty_delta=0.03)

    assert result["delta"] == pytest.approx(-0.17, abs=0.01)
    assert result["high"] < 0
    assert result["p_not_better"] < 0.001
    assert result["cells"] == 500

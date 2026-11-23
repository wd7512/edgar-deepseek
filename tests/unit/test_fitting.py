"""Offline checks that the fast fitters agree with upstream's."""

from pathlib import Path

import numpy as np
import pytest

from edgar_study import curves, fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.upstream import PENALTY, import_upstream


def test_fast_fitters_match_upstream():
    upstream = import_upstream(Path(".").resolve())
    engine, seeds = upstream["hypothesis_engine"], upstream["seed_programs"]
    loss = upstream["loss_functions"].quadratic_loss
    x, y, _, _ = prepare(*synthetic_recording(), "paper-fidelity")
    x, y = x[:15], y[:15]  # 15 cells do not divide evenly across devices, so padding is exercised
    model, estimator = seeds.neuron_model_2_jax, seeds.parameter_estimator_2

    def run(fitter):
        objective = (
            engine.objective if fitter == "upstream" else fitting.make_objective(engine.objective, fitter)
        )
        np.random.seed(0)
        _, initial_params, score, params = objective(
            model, estimator, loss, x, y, param_penalty_weight=PENALTY, max_iter=300
        )
        return score, params, fitting.train_loss(model, loss, params, x, y), initial_params

    reference, reference_params, _, initial = run("upstream")
    adam, adam_params, adam_train, _ = run("adam")
    _, _, lm_train, _ = run("lm")
    assert abs(adam - reference) < 1e-3, (adam, reference)
    assert np.allclose(adam_params, reference_params, atol=1e-3)
    assert (
        lm_train <= fitting.train_loss(model, loss, initial, x, y) + 1e-6
    )  # LM never ends worse than its start


def _winner_fits(fitter, n_cells=32, n_trials=428):
    upstream = import_upstream(Path(".").resolve())
    engine, loss = upstream["hypothesis_engine"], upstream["loss_functions"].quadratic_loss
    winner = next(c for c in curves.CURVES if c.name == curves.WINNER)
    x, y = curves.synthetic_cells(n_cells, n_trials)
    np.random.seed(0)
    _, _, score, params = fitting.make_objective(engine.objective, fitter)(
        winner.model, winner.estimator, loss, x, y, param_penalty_weight=PENALTY, max_iter=1000
    )
    return score, fitting.train_loss(winner.model, loss, params, x, y)


def test_bounded_lm_fits_the_winner_as_well_as_adam_and_scores_alike():
    adam_score, adam_train = _winner_fits("adam")
    bounded_score, bounded_train = _winner_fits("lm-bounded")
    assert bounded_train <= adam_train + 0.05, (bounded_train, adam_train)
    assert abs(bounded_score - adam_score) < 0.01 * adam_score, (bounded_score, adam_score)


def test_bounded_lm_stops_once_every_cell_reaches_tolerance():
    winner = next(c for c in curves.CURVES if c.name == curves.WINNER)
    x, y = curves.synthetic_cells(16, 200)
    start = np.stack([winner.estimator(np.asarray(x[i]), np.asarray(y[i])) for i in range(16)])
    fit = fitting.fit_lm_bounded(winner.model, start, x, y, max_iterations=500)
    assert fit.converged == 16
    assert 0 < fit.iterations < 500
    assert fit.params.shape == start.shape


def test_bounded_lm_caps_each_parameter_move_per_iteration():
    winner = next(c for c in curves.CURVES if c.name == curves.WINNER)
    x, y = curves.synthetic_cells(8, 200)
    start = np.stack([winner.estimator(np.asarray(x[i]), np.asarray(y[i])) for i in range(8)])
    fit = fitting.fit_lm_bounded(winner.model, start, x, y, max_step=0.1, max_iterations=3)
    assert fit.iterations == 3
    assert np.max(np.abs(fit.params - start)) <= 0.3 + 1e-5


def test_unknown_fitter_is_rejected():
    with pytest.raises(ValueError, match="no-such-fitter"):
        fitting.make_objective(lambda *a, **k: None, "no-such-fitter")

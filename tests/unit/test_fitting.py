"""Offline checks that the fast fitters agree with upstream's."""

import logging
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
    assert fit.stopped == 16
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


def test_bounded_lm_escapes_a_poor_start_as_adam_does():
    upstream = import_upstream(Path(".").resolve())
    engine, seeds = upstream["hypothesis_engine"], upstream["seed_programs"]
    loss = upstream["loss_functions"].quadratic_loss
    model, estimator = seeds.neuron_model_4_jax, seeds.parameter_estimator_4
    x, y = curves.synthetic_cells(32, 428)

    def train(fitter):
        np.random.seed(0)
        _, _, _, params = fitting.make_objective(engine.objective, fitter)(
            model, estimator, loss, x, y, param_penalty_weight=PENALTY, max_iter=1000
        )
        return fitting.train_loss(model, loss, params, x, y)

    assert train("lm-bounded") <= train("adam")


def test_auto_uses_the_bounded_lm_fitter():
    assert fitting.resolve("auto") == "lm-bounded"


def _cosine(theta, phi, a):
    import jax.numpy as jnp

    return jnp.abs(a) * jnp.cos(theta - phi)


def _cosine_cells():
    theta = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (6, 1))
    y = np.cos(theta).astype(np.float32)
    start = np.tile(np.array([np.pi, 1.0], dtype=np.float32), (6, 1))
    return theta, y, start


def _squared_error(params, theta, y):
    return np.sum((np.asarray(_cosine(theta, params[:, :1], params[:, 1:])) - y) ** 2, axis=1)


def test_multistart_escapes_a_start_where_a_single_fit_cannot_move():
    theta, y, start = _cosine_cells()

    single = fitting.fit_lm_bounded(_cosine, start, theta, y).params
    multi = fitting.fit_multistart(_cosine, start, theta, y, starts=8, seed=0)

    assert np.all(_squared_error(multi.params, theta, y) < 1e-3)
    assert np.all(_squared_error(single, theta, y) > 30)
    assert multi.params.shape == start.shape


def test_multistart_never_ends_worse_than_its_first_start():
    theta, y, start = _cosine_cells()
    start = start + np.float32(0.3) * np.arange(6, dtype=np.float32)[:, None] * np.array([1, 0], np.float32)

    single = fitting.fit_lm_bounded(_cosine, start, theta, y).params
    multi = fitting.fit_multistart(_cosine, start, theta, y, starts=3, seed=1)

    assert np.all(_squared_error(multi.params, theta, y) <= _squared_error(single, theta, y) + 1e-5)
    assert multi.best_start.shape == (6,)


def test_multistart_needs_at_least_one_start():
    with pytest.raises(ValueError, match="at least one start"):
        fitting.perturbed_starts(np.ones((2, 2)), starts=0)


def test_integer_starts_are_perturbed_as_floats():
    starts = fitting.perturbed_starts(np.ones((2, 3), dtype=int), starts=3, seed=0)

    assert np.issubdtype(starts.dtype, np.floating)
    assert not np.allclose(starts[1], starts[0])


def test_multistart_logs_how_many_fits_hit_the_iteration_cap(caplog):
    theta, y, start = _cosine_cells()
    with caplog.at_level(logging.INFO, logger="edgar_study.fitting"):
        fitting.fit_multistart(_cosine, start, theta, y, starts=3, max_iterations=1)

    assert any("of 18 fits stopped before the cap" in record.getMessage() for record in caplog.records)

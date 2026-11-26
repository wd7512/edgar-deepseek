"""The fair-comparison command: one explicit protocol, shared controls, and no locked cells."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from edgar_study.frontier import compare
from edgar_study.frontier.compare import (
    COHORTS,
    PROTOCOLS,
    Spec,
    best_per_node,
    cohort_rows,
    evaluate,
    paired,
    refuse_mixed,
    report,
    start_for,
)
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import canonical
from edgar_study.frontier.seed import BASE, BASE_NAMES
from edgar_study.paper_models import ai_estimator, ai_model, parameter_names

START = {"m": 3.0, "b": 0.3, "a1": 1.5, "a2": 1.0, "s": 0.6}


def _wrap(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def _double_gaussian(theta, m, b, a1, a2, s):
    return (
        b
        + a1 * jnp.exp(-0.5 * (_wrap(theta - m) / s) ** 2)
        + a2 * jnp.exp(-0.5 * (_wrap(theta - m - jnp.pi) / s) ** 2)
    )


def _estimator(theta, response):
    return np.array([float(theta[np.argmax(response)]), float(np.min(response)), 1.5, 1.0, 0.6])


def _cohort(cells=8, trials=60, seed=3):
    rng = np.random.default_rng(seed)
    theta = np.tile(np.linspace(0, 2 * np.pi, trials, endpoint=False), (cells, 1))
    m, s = rng.uniform(0, 6, (cells, 1)), rng.uniform(0.4, 0.8, (cells, 1))
    clean = np.asarray(_double_gaussian(jnp.asarray(theta), m, 0.3, 2.0, 0.8, s))
    return theta.astype(np.float32), (clean + 0.05 * rng.standard_normal(clean.shape)).astype(np.float32)


def _specs():
    model, names = _double_gaussian, BASE_NAMES
    controls = [Spec("double_gaussian", "control", model, _estimator, names)]
    extra = compile_equation({"equation": BASE + " + k*cos(theta-m)", "initial": {**START, "k": 0.0}})
    cand_model, cand_estimator = canonical(extra)
    controls.append(Spec("idea-1", "candidate", cand_model, cand_estimator, extra.canonical_names))
    return controls


def test_the_common_start_of_the_paper_equation_is_its_exact_double_gaussian_special_case():
    paper = Spec("paper", "control", ai_model, ai_estimator, tuple(parameter_names(ai_model)))
    reference = np.array([[1.1, 0.2, 2.0, 0.7, 0.5]])
    start = dict(zip(paper.names, start_for(paper, reference, np.zeros((1, 11)))[0], strict=True))
    assert start["theta_pref"] == pytest.approx(1.1) and start["baseline"] == pytest.approx(0.2)
    assert start["amplitude_1"] == pytest.approx(2.0) and start["amplitude_2"] == pytest.approx(0.7)
    widths = [start[k] for k in start if k.startswith("tuning_width")]
    assert len(widths) == 4 and widths == pytest.approx([0.5] * 4)
    assert start["exponent_1"] == start["exponent_2"] == 2.0 and start["angle_offset_2"] == 0.0


def test_a_candidate_shares_the_reference_parameters_by_name_and_keeps_its_own_initial_values():
    candidate = _specs()[1]
    reference = np.array([[1.1, 0.2, 2.0, 0.7, 0.5]])
    own = np.zeros((1, len(candidate.names)))
    own[0, candidate.names.index("k")] = 0.25
    start = dict(zip(candidate.names, start_for(candidate, reference, own)[0], strict=True))
    assert [start[n] for n in BASE_NAMES] == pytest.approx([1.1, 0.2, 2.0, 0.7, 0.5])
    assert start["k"] == 0.25


def test_under_common_start_a_model_containing_the_reference_never_trains_worse_than_it():
    cohort = _cohort()
    result = evaluate(_specs(), cohort, "common-start", max_iter=60, starts=3, seed=0)
    reference = result["reference"]["training_mse"]
    for name in ("double_gaussian", "idea-1"):
        assert result["models"][name]["training_mse"] <= reference + 1e-6
    assert result["models"]["idea-1"]["parameters"] == 6
    assert len(result["models"]["idea-1"]["cell_mse"]) == 8


def test_native_protocol_gives_every_model_its_own_start_and_no_reference():
    result = evaluate(_specs(), _cohort(), "native", max_iter=60, starts=3, seed=0)
    assert result["reference"] is None
    assert set(result["models"]) == {"double_gaussian", "idea-1"}


def test_the_paired_difference_is_per_cell_with_a_standard_error():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    other = base - np.array([0.2, 0.1, 0.3, 0.2])
    diff = paired(other, 6, base, 5)
    assert diff["mse_difference"] == pytest.approx(-0.2)
    assert diff["mse_standard_error"] == pytest.approx(np.std([-0.2, -0.1, -0.3, -0.2], ddof=1) / 2)
    assert diff["score_difference"] == pytest.approx(-0.2 * 10 + 0.01)


def test_comparisons_refuse_to_mix_protocols():
    one = {"protocol": "common-start", "cohort": "confirmation", "starts": 5, "seed": 0, "max_iter": 2000}
    refuse_mixed(one, dict(one))
    with pytest.raises(ValueError, match="protocol"):
        refuse_mixed(one, {**one, "protocol": "native"})
    with pytest.raises(ValueError, match="starts"):
        refuse_mixed(one, {**one, "starts": 3})


def test_only_discovery_cells_can_be_compared_and_locked_cells_are_refused():
    groups = {
        "discovery": np.arange(40),
        "selection": np.arange(40, 60),
        "locked_test": np.arange(60, 80),
    }
    explore, confirm = cohort_rows(groups, "exploration"), cohort_rows(groups, "confirmation")
    assert len(explore) == 30 and len(confirm) == 10
    assert set(explore) | set(confirm) == set(groups["discovery"])
    assert not set(explore) & set(confirm)
    assert set(cohort_rows(groups, "discovery")) == set(groups["discovery"])
    assert set(COHORTS) == {"exploration", "confirmation", "discovery"}
    for name in ("selection", "locked_test", "all"):
        with pytest.raises(ValueError, match="cohort"):
            cohort_rows(groups, name)
    assert set(PROTOCOLS) == {"native", "common-start"}


def test_the_best_equation_of_each_node_is_chosen_not_the_first_fitted_one():
    first = {"equation": "b + a*cos(theta-m)", "initial": {"a": 1.0, "b": 0.0, "m": 0.0}}
    second = {"equation": "b + a*cos(theta-m) + c", "initial": {"a": 1.0, "b": 0.0, "m": 0.0, "c": 0.1}}
    wave = {
        "episodes": [
            {
                "node": "idea-1",
                "candidates": [
                    {"status": "fitted", "proposal": first, "metrics": {"score": 28.9}},
                    {"status": "fit_failed", "proposal": first},
                    {"status": "fitted", "proposal": second, "metrics": {"score": 28.5}},
                ],
            },
            {
                "node": "idea-2",
                "candidates": [{"status": "fitted", "proposal": first, "metrics": {"score": 28.7}}],
            },
        ]
    }
    chosen = best_per_node(wave)
    assert chosen["idea-1"] == second and chosen["idea-2"] == first
    assert list(best_per_node(wave, top=1)) == ["idea-1"]
    assert list(best_per_node(wave, nodes=["idea-2"])) == ["idea-2"]


def test_the_report_names_the_protocol_and_every_model_with_its_difference_from_the_control():
    result = evaluate(_specs(), _cohort(), "common-start", max_iter=60, starts=3, seed=0)
    protocol = {"protocol": "common-start", "cohort": "confirmation", "starts": 3, "seed": 0, "max_iter": 60}
    text = report({"protocol": protocol, **result}, controls=("double_gaussian", "idea-1"))
    assert "common-start" in text and "confirmation" in text
    assert "vs double_gaussian" in text and "vs idea-1" in text and "| idea-1 |" in text
    json.dumps(result)


def test_module_exports_what_the_command_needs():
    assert callable(compare.run)


def test_adam_with_early_stopping_stops_before_the_cap_and_never_ends_worse_than_the_start():
    from edgar_study.fitting import fit_adam_from_starts

    theta, y = _cohort(cells=6)
    start = np.tile(np.array([[1.0, 0.2, 1.0, 0.5, 0.5]]), (6, 1))[None].astype(np.float32)
    fit = fit_adam_from_starts(_double_gaussian, start, theta, y, max_steps=4000, patience=50, tolerance=1e-3)
    before = np.mean((np.asarray(_double_gaussian(jnp.asarray(theta[0]), *start[0, 0])) - y[0]) ** 2)
    after = np.mean((np.asarray(_double_gaussian(jnp.asarray(theta[0]), *fit.params[0])) - y[0]) ** 2)
    assert after <= before and np.isfinite(fit.params).all()
    assert fit.steps < 4000 and fit.stopped.all()


def test_adam_keeps_the_best_of_several_starts_per_cell():
    from edgar_study.fitting import fit_adam_from_starts

    theta, y = _cohort(cells=4)
    good = np.tile(np.array([[3.0, 0.3, 2.0, 0.8, 0.6]]), (4, 1)).astype(np.float32)
    bad = good * 0 + np.array([0.0, 5.0, 0.0, 0.0, 3.0], dtype=np.float32)
    fit = fit_adam_from_starts(_double_gaussian, np.stack([bad, good]), theta, y, max_steps=50, patience=10)
    assert fit.best.shape == (4,) and set(fit.best.tolist()) <= {0, 1}
    assert (fit.costs <= fit.costs.max()).all() and np.isfinite(fit.costs).all()


def test_under_adam_a_model_containing_the_reference_never_trains_worse_than_it():
    result = evaluate(
        _specs(), _cohort(), "common-start", max_iter=200, starts=2, seed=0, fitter="adam", patience=30
    )
    reference = result["reference"]["training_mse"]
    for name in ("double_gaussian", "idea-1"):
        assert result["models"][name]["training_mse"] <= reference + 1e-6


def test_results_from_different_optimizers_are_not_comparable():
    one = {"protocol": "common-start", "cohort": "confirmation", "starts": 5, "seed": 0, "max_iter": 2000}
    with pytest.raises(ValueError, match="fitter"):
        refuse_mixed({**one, "fitter": "adam"}, {**one, "fitter": "lm-bounded"})

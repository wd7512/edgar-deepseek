from functools import partial
from pathlib import Path

import numpy as np
import pytest

from edgar_study.frontier.fits import double_gaussian_parent, measure
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import solve
from edgar_study.upstream import import_upstream

ROOT = Path(__file__).resolve().parents[3]


def cohort():
    x = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (10, 1))
    return x, (2 + np.cos(x)).astype(np.float32)


def test_measure_reports_whole_curve_and_per_bin_error_for_a_perfect_fit():
    equation = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}})
    result = measure(equation.model, partial(solve, equation), cohort(), 100)
    assert result["mse"] < 1e-8
    assert result["training_mse"] < 1e-8
    assert result["parameters"] == 2
    assert len(result["angle_mse"]) == 12
    assert all(value is not None and value < 1e-8 for value in result["angle_mse"])
    assert "initialization" not in result


def test_measure_scores_a_bad_fit_worse_than_a_good_one():
    good = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}})
    bad = compile_equation({"equation": "b + a*cos(2*theta)", "initial": {"b": 2, "a": 1}})
    scores = [measure(e.model, partial(solve, e), cohort(), 100)["mse"] for e in (good, bad)]
    assert scores[0] < scores[1]


def test_measure_refuses_a_nonfinite_fit():
    equation = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}})
    x, y = cohort()
    with pytest.raises(ValueError):
        measure(equation.model, partial(solve, equation), (x, np.full_like(y, np.nan)), 100)


def test_the_double_gaussian_is_fitted_under_the_same_protocol_as_candidates():
    parent = double_gaussian_parent(import_upstream(ROOT))
    x = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (10, 1))
    y = (1 + 2 * np.exp(-0.5 * (np.arctan2(np.sin(x - 1), np.cos(x - 1)) / 0.5) ** 2)).astype(np.float32)
    result = measure(parent.model, parent.solve, (x, y), 100)
    assert result["training_mse"] < 0.05
    assert result["parameters"] == 5

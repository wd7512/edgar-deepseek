"""Offline checks for the ten paper curves, their estimators and the synthetic cells."""

import math

import numpy as np
import pytest

from edgar_study import curves


def curve(name):
    return next(c for c in curves.CURVES if c.name == name)


def test_ten_curves_cover_the_paper_baseline_and_winner():
    names = [c.name for c in curves.CURVES]
    assert len(names) == 10
    assert len(set(names)) == 10
    assert {"double-gaussian", "winner"} <= set(names)
    assert curves.BASELINE == "double-gaussian"
    assert curves.WINNER == "winner"


def test_parameter_counts_match_the_paper():
    counts = {c.name: c.n_params for c in curves.CURVES}
    assert counts["winner"] == 11  # "a total of 11 parameters per cell"
    assert counts["double-gaussian"] == 5
    assert counts["single-gaussian"] == 4


def test_winner_matches_the_paper_equation():
    winner = curve("winner").model
    # b=1, A1=2, peak 1 at 0 with widths +0.5 / -1.0 and p=1; A2=0 switches off peak 2.
    params = dict(
        phi1=0.0, phi2=math.pi, b=1.0, a1=2.0, a2=0.0, s1p=0.5, s1m=1.0, s2p=0.3, s2m=0.3, p1=1.0, p2=2.0
    )
    theta = np.array([0.5, -1.0, 2 * math.pi - 1.0, 0.0])
    expected = 1 + 2 * np.exp(-0.5 * np.array([0.5 / 0.5, 1.0 / 1.0, 1.0 / 1.0, 0.0]))
    assert np.allclose(winner(theta, *curves.pack("winner", **params)), expected, atol=1e-6)


def test_winner_with_p2_and_shared_width_is_the_double_gaussian():
    theta = np.linspace(0, 2 * math.pi, 50, endpoint=False)
    gaussian = curve("double-gaussian").model(theta, 1.2, 0.3, 1.5, 0.7, 0.4)
    winner = curve("winner").model(
        theta,
        *curves.pack(
            "winner",
            phi1=1.2,
            phi2=1.2 + math.pi,
            b=0.3,
            a1=1.5,
            a2=0.7,
            s1p=0.4,
            s1m=0.4,
            s2p=0.4,
            s2m=0.4,
            p1=2.0,
            p2=2.0,
        ),
    )
    assert np.allclose(gaussian, winner, atol=1e-5)


@pytest.mark.parametrize("c", curves.CURVES, ids=lambda c: c.name)
def test_every_curve_is_periodic_and_estimates_finite_parameters(c):
    x, y = curves.synthetic_cells(n_cells=3, n_trials=120)
    params = np.asarray(c.estimator(x[0], y[0]))
    assert params.shape == (c.n_params,)
    assert np.all(np.isfinite(params))
    theta = np.array([0.3, 1.7, 4.0])
    assert np.allclose(c.model(theta, *params), c.model(theta + 2 * math.pi, *params), atol=1e-4)


def test_synthetic_cells_are_deterministic_and_normalised():
    x1, y1 = curves.synthetic_cells(n_cells=5, n_trials=60)
    x2, y2 = curves.synthetic_cells(n_cells=5, n_trials=60)
    assert x1.shape == y1.shape == (5, 60)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert np.allclose(np.linalg.norm(y1, axis=1), 100)

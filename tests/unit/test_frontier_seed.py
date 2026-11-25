from pathlib import Path

import numpy as np

from edgar_study.frontier_runner import compile_equation
from edgar_study.frontier_seed import baseline_node
from edgar_study.upstream import import_upstream


def test_starting_equation_matches_the_frozen_double_gaussian():
    upstream = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.linspace(0, 2 * np.pi, 37)
    initial = {"m": 0.73, "b": 0.2, "a1": 2.1, "a2": 0.4, "s": 0.51}
    equation = compile_equation({"equation": baseline_node()["equation"], "initial": initial})
    original = upstream["seed_programs"].neuron_model_2_jax(angles, *initial.values())
    np.testing.assert_allclose(equation.model(angles, *initial.values()), original, atol=1e-6)

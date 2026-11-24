from pathlib import Path

import numpy as np
import pytest

from edgar_study import symbolic
from edgar_study.upstream import import_upstream

GRID = np.linspace(0, 2 * np.pi, 64, endpoint=False) + 0.013


def compile_pair(tree):
    namespace_np, namespace_jax = {}, {}
    exec(symbolic.to_numpy_source(tree), namespace_np)
    exec(symbolic.to_jax_source(tree), namespace_jax)
    return namespace_np["neuron_model"], namespace_jax["neuron_model"]


def random_trees(n, seed=0):
    rng = np.random.default_rng(seed)
    return [symbolic.random_tree(rng, depth=4) for _ in range(n)]


def test_parameter_count_is_the_number_of_distinct_parameter_leaves():
    tree = ("add", ("param", "level_1", "level"), ("mul", ("param", "scale_1", "scale"), ("cos", ("theta",))))
    shared = ("add", tree, ("param", "level_1", "level"))

    assert symbolic.parameters(tree) == [("level_1", "level"), ("scale_1", "scale")]
    assert len(symbolic.parameters(shared)) == 2


def test_numpy_and_jax_sources_agree_on_random_trees():
    for tree in random_trees(40):
        model_np, model_jax = compile_pair(tree)
        values = [0.7] * len(symbolic.parameters(tree))
        expected = np.asarray(model_np(GRID, *values), dtype=float)
        got = np.asarray(model_jax(GRID, *values), dtype=float)
        assert got.shape == GRID.shape
        if not np.all(np.abs(expected) < 1e3):
            continue
        close = np.isclose(expected, got, rtol=1e-3, atol=1e-3)
        assert close.mean() >= 0.95, symbolic.to_infix(tree)


@pytest.mark.parametrize("which", [1, 2])
def test_seed_trees_reproduce_upstream_seed_programs(which):
    seeds = import_upstream(Path(".").resolve())["seed_programs"]
    reference = getattr(seeds, f"neuron_model_{which}")
    tree = symbolic.SEEDS[which - 1]
    model_np, _ = compile_pair(tree)
    rng = np.random.default_rng(which)
    for _ in range(5):
        values = {
            "pref_1": rng.uniform(0, 2 * np.pi),
            "level_1": rng.uniform(0, 3),
            "scale_1": rng.uniform(0, 5),
            "opposite_1": rng.uniform(0, 5),
            "width_1": rng.uniform(0.1, 2),
        }
        names = [name for name, _ in symbolic.parameters(tree)]
        mine = model_np(GRID, *[values[n] for n in names])
        upstream_names = {
            "theta_pref": "pref_1",
            "baseline": "level_1",
            "amplitude": "scale_1",
            "amplitude_1": "scale_1",
            "amplitude_2": "opposite_1",
            "tuning_width": "width_1",
        }
        argnames = reference.__code__.co_varnames[1 : reference.__code__.co_argcount]
        theirs = reference(GRID, *[values[upstream_names[a]] for a in argnames])
        assert np.allclose(mine, theirs, atol=1e-9)


def test_mutation_and_crossover_stay_valid():
    rng = np.random.default_rng(3)
    pool = list(symbolic.SEEDS) + random_trees(10, seed=4)
    for _ in range(300):
        a, b = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        child = symbolic.mutate(rng, symbolic.crossover(rng, a, b), steps=int(rng.integers(1, 4)))
        assert symbolic.size(child) <= symbolic.MAXSIZE
        assert symbolic.uses_theta(child)
        assert 1 <= len(symbolic.parameters(child)) <= symbolic.MAX_PARAMETERS
        pool.append(child)


def test_mutation_is_reproducible_for_a_seed():
    first = symbolic.mutate(np.random.default_rng(9), symbolic.SEEDS[1], steps=3)
    second = symbolic.mutate(np.random.default_rng(9), symbolic.SEEDS[1], steps=3)
    assert first == second


def test_parse_reads_pysr_style_strings():
    tree = symbolic.parse("p1 + p2 * exp(cos(x0 - p3)) / 2.5 + square(abs(x0))^0.5")
    model_np, _ = compile_pair(tree)
    names = [name for name, _ in symbolic.parameters(tree)]
    assert len(names) == 3
    values = dict(zip(names, [0.3, 1.7, 0.9], strict=True))
    expected = 0.3 + 1.7 * np.exp(np.cos(GRID - 0.9)) / 2.5 + np.abs(GRID**2) ** 0.5
    assert np.allclose(model_np(GRID, *values.values()), expected)


def test_parse_round_trips_infix():
    for tree in random_trees(20, seed=7):
        again = symbolic.parse(symbolic.to_infix(tree))
        model_a, _ = compile_pair(tree)
        model_b, _ = compile_pair(again)
        values = [symbolic.KINDS[kind].default for _, kind in symbolic.parameters(tree)]
        assert np.allclose(np.nan_to_num(model_a(GRID, *values)), np.nan_to_num(model_b(GRID, *values)))


def test_estimator_returns_one_start_per_parameter_inside_its_bounds():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    response = 1 + 4 * np.exp(np.cos(theta - 1.0) - 1) + rng.normal(0, 0.1, 200)
    for tree in list(symbolic.SEEDS) + random_trees(10, seed=11):
        source = symbolic.estimator_source(tree)
        assert not any(word in source for word in ("lstsq", "scipy.optimize", "curve_fit", "sklearn"))
        namespace = {}
        exec(source, namespace)
        start = namespace["parameter_estimator"](theta, response)
        kinds = [kind for _, kind in symbolic.parameters(tree)]
        assert start.shape == (len(kinds),)
        for value, kind in zip(start, kinds, strict=True):
            low, high = symbolic.KINDS[kind].bounds
            assert low <= value <= high


def test_seed_estimator_finds_the_preferred_direction():
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    response = 0.5 + 3 * np.exp(-0.5 * (np.angle(np.exp(1j * (theta - 2.0))) / 0.4) ** 2)
    namespace = {}
    exec(symbolic.estimator_source(symbolic.SEEDS[1]), namespace)
    start = dict(
        zip(
            [n for n, _ in symbolic.parameters(symbolic.SEEDS[1])],
            namespace["parameter_estimator"](theta, response),
            strict=True,
        )
    )
    assert abs(start["pref_1"] - 2.0) < 0.2
    assert abs(start["level_1"] - 0.5) < 0.2
    assert abs(start["scale_1"] - 3.0) < 0.3

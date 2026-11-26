"""The fitting benchmark: proven nestings, violation and regret arithmetic, and the training-only contract."""

import json

import numpy as np
import pytest

from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.frontier import bench
from edgar_study.frontier.protocol import split_trials


def test_every_nesting_in_the_fixture_is_proven_and_it_draws_on_both_runs():
    fixture = bench.load_fixture()

    assert len(fixture.equations) >= 15 and len(fixture.edges) >= 18
    sources = " ".join(fixture.sources.values())
    assert "run-20" in sources and "run-cold-10" in sources
    assert {(e.child, e.parent) for e in fixture.edges} >= {("W6", "W6b"), ("W6b", "W6"), ("SK9", "S7")}


def test_a_map_that_does_not_reproduce_its_parent_is_refused(tmp_path):
    raw = json.loads(bench.FIXTURE.read_text())
    edge = next(e for e in raw["edges"] if (e["child"], e["parent"]) == ("S7", "G5"))
    edge["map"]["d1"] = 0.3
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="S7.*G5"):
        bench.load_fixture(path)


def test_a_child_worse_than_its_parent_beyond_the_tolerance_is_a_violation():
    edges = (bench.Edge("c", "p", None),)
    errors = {"p": np.array([1.0, 1.0, 1.0, 2.0]), "c": np.array([0.5, 1.0005, 1.2, 2.5])}

    (row,) = bench.violations(edges, errors, tolerance=1e-3)

    assert (row["child"], row["parent"], row["violated"], row["cells"]) == ("c", "p", 2, 4)
    assert row["worst"] == pytest.approx(0.25)


def test_the_best_known_error_flows_down_chains_and_from_either_parent_and_across_runs():
    edges = (bench.Edge("c", "p", None), bench.Edge("g", "c", None), bench.Edge("g", "q", None))
    first = {
        "p": np.array([1.0, 2.0]),
        "c": np.array([1.5, 1.5]),
        "q": np.array([0.5, 3.0]),
        "g": np.array([2.0, 2.0]),
    }
    second = {**first, "c": np.array([0.9, 5.0])}

    best = bench.best_known(edges, [first, second])

    np.testing.assert_allclose(best["p"], [1.0, 2.0])
    np.testing.assert_allclose(best["c"], [0.9, 1.5])
    np.testing.assert_allclose(best["g"], [0.5, 1.5])


def test_an_equivalence_shares_its_best_value_both_ways():
    edges = (bench.Edge("a", "b", None), bench.Edge("b", "a", None))
    best = bench.best_known(edges, [{"a": np.array([2.0, 1.0]), "b": np.array([1.0, 3.0])}])

    np.testing.assert_allclose(best["a"], [1.0, 1.0])
    np.testing.assert_allclose(best["b"], [1.0, 1.0])


def test_regret_is_relative_to_the_best_known_value():
    best = {"a": np.array([1.0, 2.0])}
    row = bench.regret({"a": np.array([1.0, 3.0])}, best)

    assert row["mean"] == pytest.approx(0.25)
    assert row["at_best"] == pytest.approx(0.5)
    assert row["worst_equation"] == "a"


def _cohort(cells=6):
    x, y, _, _ = prepare(*synthetic_recording(), "paper-fidelity")
    return np.asarray(x[:cells]), np.asarray(y[:cells])


def test_a_fitter_sees_only_training_trials():
    fixture = bench.load_fixture().subset(["G5", "W6"])
    x, y = _cohort()
    _, held = split_trials(x)
    y = y.copy()
    y[:, np.asarray(held)] = 1e6
    seen = []

    def fit(equation, xt, yt, warm):
        seen.append(yt)
        return np.tile(np.array(list(equation.initial.values()), dtype=np.float32), (len(xt), 1))

    result = bench.evaluate(fixture, bench.Fitter(fit, warm=False), (x, y))

    assert len(seen) == 2
    assert all(s.shape == (6, x.shape[1] // 2) and s.max() < 1e6 for s in seen)
    assert set(result["training"]) == {"G5", "W6"} and min(result["held_out"]["G5"]) > 1e10


def test_warm_starts_through_the_proven_maps_leave_no_child_worse_than_its_parent():
    fixture = bench.load_fixture().subset(["G5", "W6", "S7", "SW8"])
    result = bench.evaluate(fixture, bench.FITTERS["warm"], _cohort())
    errors = {k: np.asarray(v) for k, v in result["training"].items()}

    assert [row["violated"] for row in bench.violations(fixture.edges, errors)] == [0, 0, 0, 0]


def test_results_are_saved_per_cohort_and_the_report_names_every_fitter(tmp_path):
    fixture = bench.load_fixture().subset(["G5", "W6"])
    results = {
        "fast": {"training": {"G5": [1.0, 1.0], "W6": [1.2, 0.9]}, "seconds": {"G5": 1.0, "W6": 2.0}},
        "slow": {"training": {"G5": [1.0, 1.0], "W6": [0.9, 0.9]}, "seconds": {"G5": 3.0, "W6": 4.0}},
    }
    for name, result in results.items():
        bench.save_result(tmp_path / "one", name, result)

    assert bench.load_results(tmp_path / "one") == results
    assert bench.load_results(tmp_path / "two") == {}
    table = bench.report(fixture, results)
    assert "| fast | 3.0 | 1 of 2 |" in table
    assert "| slow | 7.0 | 0 of 2 |" in table
    assert "G5 ⊂ W6" in table

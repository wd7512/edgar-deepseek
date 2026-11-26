import numpy as np
import pytest

from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import fit_equation, split_trials
from edgar_study.frontier.warm import Parent, complete_map, declared_probes, warm_start

CELLS = 6


def cohort():
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (CELLS, 1))
    b = rng.uniform(1, 3, (CELLS, 1))
    a = rng.uniform(0.5, 2, (CELLS, 1))
    c = rng.uniform(-1, 1, (CELLS, 1))
    return x, (b + a * np.cos(x) + c * np.sin(x) + 0.02 * rng.standard_normal(x.shape)).astype(np.float32)


def compile_(text, initial, nesting=None):
    proposal = {"equation": text, "initial": initial}
    if nesting is not None:
        proposal["nesting"] = nesting
    return compile_equation(proposal)


def parent_of(equation, node_id="parent", fitted=None):
    names = tuple(equation.initial)
    probes = np.array([list(equation.initial.values()), [v + 0.37 for v in equation.initial.values()]])
    if fitted is None:
        fitted = fit_equation(equation, cohort(), 100).params
    return Parent(node_id, names, equation.model, probes, lambda _cohort, _max_iter: fitted)


def test_matching_names_carry_the_parents_fitted_values_and_new_parameters_take_values_that_reproduce_it():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("b + a*cos(theta) + c*sin(theta)", {"b": 0, "a": 0, "c": 0.5})
    link = parent_of(parent)
    start = warm_start(child, [link], cohort(), 100)
    np.testing.assert_allclose(start[:, 0], link.fit(None, 0)[:, 0])
    np.testing.assert_allclose(start[:, 1], link.fit(None, 0)[:, 1])
    np.testing.assert_allclose(start[:, 2], 0.0)


def test_a_child_whose_new_parameters_have_no_neutral_value_keeps_its_own_start_for_them():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("b + a*cos(theta) + (c-2)*sin(theta)", {"b": 0, "a": 0, "c": 0.5})
    link = parent_of(parent)
    start = warm_start(child, [link], cohort(), 100)
    np.testing.assert_allclose(start[:, 1], link.fit(None, 0)[:, 1])
    np.testing.assert_allclose(start[:, 2], 0.5)


GAUSS = "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2)"
GAUSS_START = {"b": 0.3, "a1": 2.0, "a2": 1.0, "m": 1.0, "s": 0.4}
SHIFTED = "b + a1*exp(-0.5*(wrap(theta-(m+d1))/s)**2) + a2*exp(-0.5*(wrap(theta-(m+pi+d2))/s)**2)"
WIDTHS = "b + a1*exp(-0.5*(wrap(theta-m)/s1)**2)*(1+tanh(k1*wrap(theta-m))) + a2*exp(-0.5*(wrap(theta-m-pi)/s2)**2)"


def gauss_parent():
    equation = compile_(GAUSS, GAUSS_START)
    return Parent("double_gaussian", tuple(equation.initial), equation.model, declared_probes(equation), None)


def test_a_map_that_leaves_out_new_parameters_is_completed_with_values_that_reproduce_the_parent():
    declared = {"double_gaussian": {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s"}}
    child = compile_(SHIFTED, {**GAUSS_START, "d1": 0.2, "d2": -0.1}, declared)

    found = complete_map(child, gauss_parent())

    assert found is not None
    assert found.bindings == {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s", "d1": 0.0, "d2": 0.0}
    assert not found.declared


def test_with_no_map_numbered_copies_of_a_parent_parameter_and_neutral_values_are_found():
    child = compile_(WIDTHS, {"b": 0.3, "a1": 2.0, "a2": 1.0, "m": 1.0, "s1": 0.3, "s2": 0.5, "k1": 0.4})

    found = complete_map(child, gauss_parent())

    assert found is not None
    assert found.bindings["s1"] == "s" and found.bindings["s2"] == "s" and found.bindings["k1"] == 0.0


def test_a_wrong_declared_binding_is_repaired():
    declared = {
        "double_gaussian": {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s", "d1": 0.3, "d2": 0.0}
    }
    child = compile_(SHIFTED, {**GAUSS_START, "d1": 0.2, "d2": -0.1}, declared)

    assert complete_map(child, gauss_parent()).bindings["d1"] == 0.0


def test_a_correct_declared_map_is_kept_as_declared():
    declared = {
        "double_gaussian": {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s", "d1": 0.0, "d2": 0.0}
    }
    child = compile_(SHIFTED, {**GAUSS_START, "d1": 0.2, "d2": -0.1}, declared)

    assert complete_map(child, gauss_parent()).declared


def test_a_child_that_does_not_contain_its_parent_gets_no_map():
    laplace = compile_("b + a1*exp(-abs(wrap(theta-m))/s) + a2*exp(-abs(wrap(theta-m-pi))/s)", GAUSS_START)
    assert complete_map(laplace, gauss_parent()) is None


def test_a_parent_sharing_no_names_with_the_child_gives_no_warm_start():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("p + q*cos(theta)", {"p": 0, "q": 0})
    assert warm_start(child, [parent_of(parent)], cohort(), 100) is None
    assert warm_start(child, [], cohort(), 100) is None


def test_a_verified_nesting_map_carries_the_fit_across_renamed_parameters():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    nesting = {"parent": {"p": "b", "q": "a", "r": 0}}
    child = compile_("p + q*cos(theta) + r*sin(theta)", {"p": 0, "q": 0, "r": 0.5}, nesting)
    link = parent_of(parent)
    start = warm_start(child, [link], cohort(), 100)
    fitted = link.fit(None, 0)
    np.testing.assert_allclose(start[:, 0], fitted[:, 0], rtol=1e-6)
    np.testing.assert_allclose(start[:, 1], fitted[:, 1], rtol=1e-6)
    np.testing.assert_allclose(start[:, 2], 0)


def test_a_nesting_map_that_does_not_reproduce_the_parent_is_rejected():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    wrong = {"parent": {"p": "a", "q": "b", "r": 0}}
    child = compile_("p + q*cos(theta) + r*sin(theta)", {"p": 0, "q": 0, "r": 0.5}, wrong)
    assert warm_start(child, [parent_of(parent)], cohort(), 100) is None


@pytest.mark.parametrize(
    "bindings",
    [
        {"p": "b", "q": "a"},
        {"p": "b", "q": "a", "r": "missing"},
        {"p": "b", "q": "a", "r": [1, 2]},
        {"p": "b", "q": "a", "r": 1000},
    ],
)
def test_malformed_or_missing_entries_are_dropped_and_the_rest_of_the_map_is_completed(bindings):
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("p + q*cos(theta) + r*sin(theta)", {"p": 0, "q": 0, "r": 0.5}, {"parent": bindings})
    link = parent_of(parent)
    start = warm_start(child, [link], cohort(), 100)
    np.testing.assert_allclose(start[:, :2], link.fit(None, 0), rtol=1e-6)
    np.testing.assert_allclose(start[:, 2], 0)


def test_a_map_that_is_not_an_object_and_shares_no_names_gives_no_warm_start():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("p + q*cos(theta) + r*sin(theta)", {"p": 0, "q": 0, "r": 0.5}, {"parent": "not a map"})
    assert warm_start(child, [parent_of(parent)], cohort(), 100) is None


def test_with_several_parents_each_neuron_takes_the_start_with_the_lowest_training_error():
    child = compile_("b + a*cos(theta) + c*sin(theta)", {"b": 0, "a": 0, "c": 0})
    x, y = cohort()
    train, _ = split_trials(x)
    good = np.stack([np.mean(y[:, train], axis=1), np.ones(CELLS), np.zeros(CELLS)], axis=1)
    poor = np.full((CELLS, 3), 9.0)
    first = Parent("one", ("b", "a", "c"), child.model, np.ones((1, 3)), lambda *_: poor)
    second = Parent("two", ("b", "a", "c"), child.model, np.ones((1, 3)), lambda *_: good)
    for order in ([first, second], [second, first]):
        np.testing.assert_allclose(warm_start(child, order, cohort(), 100), good)
    mixed = good.copy()
    mixed[:3] = 9.0
    third = Parent("three", ("b", "a", "c"), child.model, np.ones((1, 3)), lambda *_: mixed)
    chosen = warm_start(child, [first, third, second], cohort(), 100)
    np.testing.assert_allclose(chosen, good)


def test_a_child_that_contains_its_parent_never_fits_worse_than_the_parent_under_a_warm_start():
    parent = compile_("b + a*cos(theta)", {"b": 2, "a": 1})
    child = compile_("b + a*cos(theta) + c*sin(theta)", {"b": 9, "a": -9, "c": 0.5})
    link = parent_of(parent)
    reference = fit_equation(parent, cohort(), 100)
    warm = warm_start(child, [link], cohort(), 100)
    result = fit_equation(child, cohort(), 100, warm=warm)
    assert result.training_mse <= reference.training_mse + 1e-9

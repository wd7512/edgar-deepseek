import jax
import numpy as np
import pytest

from edgar_study.frontier.grammar import InvalidEquation, compile_equation
from edgar_study.frontier.seed import BASE


def test_equation_grammar_rejects_code_and_deduplicates_parameter_renames():
    one = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 2}})
    two = compile_equation({"equation": "q*cos(theta) + c", "initial": {"q": 3, "c": 4}})
    assert one.key == two.key
    np.testing.assert_allclose(
        one.model(np.array([0, np.pi]), *one.estimator(None, None)), [3, -1], atol=1e-6
    )
    with pytest.raises(ValueError):
        compile_equation({"equation": "__import__('os').system('false')", "initial": {}})


def test_equation_grammar_accepts_periodic_cutoffs_and_width_drift():
    window = compile_equation(
        {
            "equation": "b + a*(0.5+0.5*cos(wrap(theta-m)))*(abs(wrap(theta-m))<w)",
            "initial": {"b": 1, "a": 2, "m": 0, "w": 1},
        }
    )
    angles = np.array([0, np.pi, 2 * np.pi])
    np.testing.assert_allclose(window.model(angles, *window.estimator(None, None)), [3, 1, 3], atol=1e-6)
    combination = compile_equation(
        {
            "equation": BASE.replace("b + ", "b + r*cos(theta - phi) + ", 1).replace(
                "(wrap(theta-m)/s)**2", "(wrap(theta-m)/(s*(1 + k*tanh(wrap(theta-m)/w))))**2"
            ),
            "initial": {"b": 1, "m": 0, "a1": 2, "a2": 1, "s": 0.5, "r": 0, "phi": 0, "k": 0, "w": 1},
        }
    )
    np.testing.assert_allclose(combination.model(angles, *combination.estimator(None, None)), [3, 2, 3])


def test_cutoff_width_has_a_gradient_at_its_edge():
    equation = compile_equation(
        {
            "equation": "b + a*(abs(wrap(theta-m))<w)",
            "initial": {"b": 1, "a": 2, "m": 0, "w": 1},
        }
    )
    assert jax.grad(lambda w: equation.model(1.0, 1.0, 2.0, 0.0, w))(1.0) > 1
    edge = compile_equation(
        {
            "equation": "b+a*(abs(wrap(theta))<w)",
            "initial": {"b": 1, "a": 2, "w": 2.9},
        }
    )
    np.testing.assert_allclose(edge.model(-2.9, 1, 2, 2.9), edge.model(2 * np.pi - 2.9, 1, 2, 2.9), atol=1e-5)


def test_cutoffs_accept_all_comparisons_and_need_real_inputs():
    for left, op, right in (
        ("wrap(theta)", "<", "0"),
        ("wrap(theta)", "<=", "0"),
        ("wrap(theta)", ">", "0"),
        ("wrap(theta)", ">=", "0"),
        ("0", "<", "wrap(theta)"),
        ("0", ">=", "wrap(theta)"),
    ):
        equation = compile_equation({"equation": f"b+a*({left}{op}{right})", "initial": {"b": 1, "a": 2}})
        assert 0 < equation.model(0.0, 1, 2) < 3
    with pytest.raises(ValueError, match="real-valued"):
        compile_equation({"equation": "b+a*((theta*(-1)**0.5)>=0)", "initial": {"b": 1, "a": 2}})


def test_undeclared_names_have_their_own_reason():
    with pytest.raises(ValueError, check=lambda exc: getattr(exc, "code", None) == "unknown_name"):
        compile_equation({"equation": "b+a*cos(w1)", "initial": {"b": 1, "a": 2, "m1": 3}})


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("a*cos(theta) + 1/0", "nonfinite_expression"),
        ("a*cos(theta)*(-1)**0.5", "nonreal_expression"),
        ("a*cos(theta) + (cos(theta)*(-1)**0.5 < 1)", "nonreal_expression"),
        ("a*theta", "nonperiodic_expression"),
        ("a*cos(theta)+(0<theta<1)", "unsupported_operator"),
        ("a*cos(theta)" + "+1" * 85, "equation_too_complex"),
    ],
)
def test_invalid_equation_has_stable_reason(expression, reason):
    with pytest.raises(ValueError, check=lambda exc: getattr(exc, "code", None) == reason):
        compile_equation({"equation": expression, "initial": {"a": 1}})


@pytest.mark.parametrize(
    ("text", "initial", "code"),
    [
        ("a*log(theta)", {"a": 1}, "unsupported_operator"),
        ("__import__('os').system('false')", {"a": 1}, "unsupported_operator"),
        ("a*cos(theta", {"a": 1}, "invalid_syntax"),
        ("a*cos(x)", {"a": 1}, "unknown_name"),
        ("cos(theta)", {"a": 1}, "parameter_mismatch"),
        ("a*cos(theta)", {"a": 1, "c": 1}, "parameter_mismatch"),
        ("a*theta", {"a": 1}, "nonperiodic_expression"),
        ("a/(theta-theta)", {"a": 1}, "nonfinite_expression"),
        ("a*cos(theta)", {"a": 500}, "invalid_parameter"),
        ("a*cos(theta)", {"theta": 1}, "invalid_parameter"),
        ("a*cos(theta)", {}, "invalid_parameter_count"),
        ("a*cos(theta)" + " " * 600, {"a": 1}, "equation_too_complex"),
        ("a*cos(theta)", [], "invalid_equation_object"),
    ],
)
def test_equations_are_rejected_with_a_stable_reason_code(text, initial, code):
    with pytest.raises(InvalidEquation) as caught:
        compile_equation({"equation": text, "initial": initial})
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("text", "initial"),
    [
        ("b + a*cos(theta)", {"a": 1, "b": 1}),
        ("b + a*exp(-0.5*(wrap(theta-m)/s)**2)", {"a": 1, "b": 1, "m": 1, "s": 1}),
        ("b + a*(cos(theta) < c)", {"a": 1, "b": 1, "c": 0}),
        ("b + a*(c >= cos(theta))", {"a": 1, "b": 1, "c": 0}),
        ("-a*sin(2*theta) + b", {"a": 1, "b": 1}),
    ],
)
def test_equations_in_the_grammar_are_accepted(text, initial):
    assert compile_equation({"equation": text, "initial": initial}).text == text


def test_equations_that_differ_only_by_renaming_or_ordering_share_a_key():
    one = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 2}})
    two = compile_equation({"equation": "q*cos(theta) + c", "initial": {"q": 3, "c": 4}})
    three = compile_equation({"equation": "b + a*cos(2*theta)", "initial": {"b": 1, "a": 2}})
    assert one.key == two.key
    assert one.key != three.key

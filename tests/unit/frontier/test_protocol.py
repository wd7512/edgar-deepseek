import numpy as np
import pytest

from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import STARTS, fit_equation, fit_model, solve
from edgar_study.frontier.seed import BASE

SIGMA = 0.05
DOUBLE_GAUSSIAN_START = {"m": 3.0, "b": 0.3, "a1": 1.5, "a2": 1.0, "s": 0.6}


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def double_gaussian_cohort(cells=24, trials=72, seed=1, noise=SIGMA):
    rng = np.random.default_rng(seed)
    theta = np.tile(np.linspace(0, 2 * np.pi, trials, endpoint=False), (cells, 1)).astype(np.float32)
    m, s = rng.uniform(0, 2 * np.pi, cells)[:, None], rng.uniform(0.3, 0.9, cells)[:, None]
    b, a1, a2 = (rng.uniform(lo, hi, cells)[:, None] for lo, hi in ((0.2, 0.5), (1, 2.5), (0.2, 1.5)))
    clean = (
        b
        + a1 * np.exp(-0.5 * (wrap(theta - m) / s) ** 2)
        + a2 * np.exp(-0.5 * (wrap(theta - m - np.pi) / s) ** 2)
    )
    return theta, (clean + noise * rng.standard_normal(clean.shape)).astype(np.float32)


def equation(text=BASE, initial=None):
    return compile_equation({"equation": text, "initial": initial or DOUBLE_GAUSSIAN_START})


def test_a_constant_cold_start_reaches_the_noise_floor_on_nearly_every_cell():
    result = fit_equation(equation(), double_gaussian_cohort(), 100)
    assert result.mse < 1.5 * SIGMA**2
    assert result.training_mse < 1.5 * SIGMA**2


def test_more_starts_never_give_a_worse_training_fit():
    cohort = double_gaussian_cohort()
    costs = [fit_equation(equation(), cohort, 100, starts=k).training_mse for k in (1, 2, 4, 8)]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(costs, costs[1:], strict=False))


def test_the_same_inputs_give_the_same_fit_and_the_seed_only_moves_the_jitter():
    cohort = double_gaussian_cohort()
    one = fit_equation(equation(), cohort, 100)
    again = fit_equation(equation(), cohort, 100)
    np.testing.assert_array_equal(one.params, again.params)
    other = fit_equation(equation(), cohort, 100, seed=5)
    assert other.params.shape == one.params.shape
    assert other.training_mse < 1.5 * SIGMA**2


def test_equivalent_equations_get_identical_fits_whatever_their_parameter_names_and_order():
    cohort = double_gaussian_cohort()
    renamed = (
        BASE.replace("a1", "p")
        .replace("a2", "q")
        .replace("m", "mu")
        .replace("s)", "w)")
        .replace("b +", "c +")
    )
    one = fit_equation(equation(), cohort, 100)
    two = fit_equation(
        equation(renamed, {"w": 0.6, "q": 1.0, "mu": 3.0, "c": 0.3, "p": 1.5}),
        cohort,
        100,
    )
    assert equation().key == equation(renamed, {"w": 0.6, "q": 1.0, "mu": 3.0, "c": 0.3, "p": 1.5}).key
    assert two.mse == pytest.approx(one.mse, rel=1e-5)
    assert two.training_mse == pytest.approx(one.training_mse, rel=1e-5)
    assert two.count == one.count


def test_held_out_responses_never_influence_the_fit():
    theta, y = double_gaussian_cohort()
    perturbed = y.copy()
    from edgar_study.frontier.protocol import split_trials

    _, held = split_trials(theta)
    perturbed[:, held] += 10
    clean = fit_equation(equation(), (theta, y), 100)
    changed = fit_equation(equation(), (theta, perturbed), 100)
    np.testing.assert_array_equal(clean.params, changed.params)
    assert changed.training_mse == clean.training_mse
    assert changed.mse > clean.mse + 50


def test_a_fit_is_independent_of_which_other_equations_were_fitted_first():
    cohort = double_gaussian_cohort()
    other = equation("b + a*cos(theta)", {"b": 1, "a": 1})
    alone = fit_equation(equation(), cohort, 100)
    fit_equation(other, cohort, 100)
    after = fit_equation(equation(), cohort, 100)
    np.testing.assert_array_equal(alone.params, after.params)


def test_a_model_supplies_its_own_start_and_gets_the_same_budget_as_any_other():
    cohort = double_gaussian_cohort()
    smart = equation()
    calls = []

    def estimator(theta, response):
        calls.append(len(theta))
        return tuple(DOUBLE_GAUSSIAN_START.values())

    result = fit_model(smart.model, estimator, cohort, 100)
    assert result.mse < 1.5 * SIGMA**2
    assert len(calls) == len(cohort[0])
    assert max(calls) == cohort[0].shape[1] // 2


def test_solve_returns_the_score_mse_count_and_parameters_the_runner_records():
    score, mse, count, params = solve(equation(), double_gaussian_cohort(), 100)
    assert count == 5 == params.shape[1]
    assert score == pytest.approx(mse * 10 + count * 0.01, rel=1e-4)


def test_zero_iterations_return_the_estimator_start_unchanged():
    cohort = double_gaussian_cohort(cells=3)
    result = fit_equation(equation(), cohort, 0)
    np.testing.assert_allclose(
        result.params, np.tile(list(DOUBLE_GAUSSIAN_START.values()), (3, 1)), rtol=1e-6
    )


def test_an_equation_that_cannot_fit_is_reported_not_silently_scored():
    cohort = double_gaussian_cohort(cells=3)
    poisoned = (cohort[0], np.full_like(cohort[1], np.nan))
    with pytest.raises(ValueError):
        fit_equation(equation(), poisoned, 100)


def test_the_default_number_of_starts_is_a_documented_constant():
    assert STARTS == 5


def test_a_warm_start_takes_the_last_slot_and_never_leaves_the_fit_worse_than_the_cold_starts():
    cohort = double_gaussian_cohort()
    cold = fit_equation(equation(), cohort, 100, starts=4)
    poor = np.full((24, 5), 0.1, dtype=np.float32)
    warm = fit_equation(equation(), cohort, 100, starts=5, warm=poor)
    assert warm.training_mse <= cold.training_mse + 1e-9


def test_a_warm_start_from_a_better_fit_is_used():
    theta, y = double_gaussian_cohort()
    first = fit_equation(equation(), (theta, y), 100, starts=1)
    shifted = equation(initial={"m": 0.1, "b": 5.0, "a1": 0.1, "a2": 0.1, "s": 3.0})
    cold = fit_equation(shifted, (theta, y), 100, starts=1)
    seeded = fit_equation(shifted, (theta, y), 100, starts=2, warm=first.params)
    assert seeded.training_mse <= first.training_mse + 1e-6
    assert seeded.training_mse < cold.training_mse


def test_a_warm_start_is_given_in_the_equations_declared_parameter_order():
    cohort = double_gaussian_cohort(cells=4)
    renamed = equation(BASE.replace("m", "mu"), {"s": 0.6, "mu": 3.0, "a1": 1.5, "a2": 1.0, "b": 0.3})
    names = list(renamed.initial)
    first = fit_equation(equation(), cohort, 100, starts=1)
    reference = dict(zip(equation().initial, first.params.T, strict=True))
    warm = np.stack([reference["m" if name == "mu" else name] for name in names], axis=1)
    result = fit_equation(renamed, cohort, 0, starts=2, warm=warm)
    assert result.training_mse == pytest.approx(first.training_mse, rel=1e-4)


def test_a_warm_start_is_ignored_when_only_one_start_is_allowed_and_must_match_the_cells():
    cohort = double_gaussian_cohort(cells=4)
    nonsense = np.zeros((4, 5), dtype=np.float32)
    assert (
        fit_equation(equation(), cohort, 0, starts=1, warm=nonsense).params.tolist()
        == fit_equation(equation(), cohort, 0, starts=1).params.tolist()
    )
    with pytest.raises(ValueError):
        fit_equation(equation(), cohort, 0, starts=2, warm=np.zeros((3, 5), dtype=np.float32))

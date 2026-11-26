import numpy as np
import pytest

from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.protocol import (
    STARTS,
    fit_equation,
    fit_equation_training,
    fit_model,
    predict,
    solve,
    split_trials,
)
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
    assert other.training_mse < 4 * SIGMA**2


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


FAMILIES = {
    "generalised_gaussian": (
        "b + a1*exp(-(abs(wrap(theta-m))/s)**p) + a2*exp(-(abs(wrap(theta-m-pi))/s)**p)",
        {"m": 3.0, "b": 0.3, "a1": 1.5, "a2": 1.0, "s": 0.6, "p": 2.0},
        {"b": (0.2, 0.5), "a1": (1, 2.5), "a2": (0.2, 1.5), "s": (0.4, 1.0), "p": (1.2, 3.0)},
    ),
    "von_mises": (
        "b + a*exp(k*(cos(theta-m)-1))",
        {"m": 3.0, "b": 0.3, "a": 1.5, "k": 2.0},
        {"b": (0.2, 0.5), "a": (1, 2.5), "k": (1.0, 8.0)},
    ),
    "skewed_width": (
        "b + a*exp(-0.5*(wrap(theta-m)/(s*(1 + k*(wrap(theta-m) > 0))))**2)",
        {"m": 3.0, "b": 0.3, "a": 1.5, "s": 0.5, "k": 0.0},
        {"b": (0.2, 0.5), "a": (1, 2.5), "s": (0.3, 0.7), "k": (0.2, 1.0)},
    ),
    "harmonics": (
        "b + a1*cos(theta-p1) + a2*cos(2*theta-p2)",
        {"b": 1.0, "a1": 1.0, "p1": 1.0, "a2": 0.5, "p2": 1.0},
        {"b": (0.5, 2), "a1": (0.5, 2), "p1": (0, 6.28), "a2": (0.2, 1), "p2": (0, 6.28)},
    ),
}


def family_cohort(text, start, ranges, cells=24, seed=3):
    rng = np.random.default_rng(seed)
    equation = compile_equation({"equation": text, "initial": start})
    truth = {
        name: rng.uniform(*ranges[name], cells) if name in ranges else rng.uniform(0, 2 * np.pi, cells)
        for name in start
    }
    theta = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False), (cells, 1)).astype(np.float32)
    columns = np.stack([truth[name] for name in equation.initial], axis=1)
    import jax

    clean = np.asarray(jax.vmap(equation.model)(theta, *columns.T))
    return theta, (clean + SIGMA * rng.standard_normal(clean.shape)).astype(np.float32)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_of_equation_the_model_proposes_is_recovered_from_a_cold_start(family):
    import jax

    text, start, ranges = FAMILIES[family]
    theta, y = family_cohort(text, start, ranges)
    fitted = compile_equation({"equation": text, "initial": start})
    result = fit_equation(fitted, (theta, y), 100)
    errors = np.mean((np.asarray(jax.vmap(fitted.model)(theta, *result.params.T)) - y) ** 2, axis=1)
    assert np.mean(errors < 2 * SIGMA**2) >= 0.75
    assert np.median(errors) < 1.5 * SIGMA**2


def curves(params):
    import jax

    grid = np.tile(np.linspace(0, 2 * np.pi, 90, endpoint=False), (len(params), 1)).astype(np.float32)
    return np.asarray(jax.vmap(equation().model)(grid, *np.asarray(params).T))


def test_a_cells_fitted_curve_does_not_depend_on_which_other_cells_are_fitted_with_it():
    theta, y = double_gaussian_cohort(cells=12)
    together = curves(fit_equation(equation(), (theta, y), 100).params)
    order = np.random.default_rng(0).permutation(12)
    shuffled = curves(fit_equation(equation(), (theta[order], y[order]), 100).params)
    np.testing.assert_allclose(shuffled, together[order], atol=0.15)
    alone = curves(fit_equation(equation(), (theta[3:4], y[3:4]), 100).params)
    np.testing.assert_allclose(alone, together[3:4], atol=0.15)


def test_a_parent_fit_full_of_nonfinite_values_cannot_poison_the_child():
    cohort = double_gaussian_cohort(cells=6)
    cold = fit_equation(equation(), cohort, 100, starts=4)
    bad = np.full((6, 5), np.nan, dtype=np.float32)
    bad[2] = np.inf
    result = fit_equation(equation(), cohort, 100, starts=5, warm=bad)
    assert np.isfinite(result.params).all()
    assert result.training_mse <= cold.training_mse + 1e-9


def test_starts_that_jitter_into_an_invalid_domain_are_discarded_not_fatal():
    theta, y = double_gaussian_cohort(cells=8)
    text = "b + a*exp(-0.5*(wrap(theta-m)/sqrt(w))**2)"
    sqrt_equation = compile_equation({"equation": text, "initial": {"b": 0.3, "a": 1.5, "m": 3.0, "w": 0.05}})
    result = fit_equation(sqrt_equation, (theta, y), 100, starts=8)
    assert np.isfinite(result.params).all()
    assert np.isfinite(result.training_mse)


def test_one_unfittable_cell_makes_the_whole_equation_unfittable_instead_of_hiding_it():
    theta, y = double_gaussian_cohort(cells=6)
    y[4] = np.nan
    with pytest.raises(ValueError):
        fit_equation(equation(), (theta, y), 100)


@pytest.mark.parametrize("trials", [4, 5, 7, 11])
def test_odd_and_tiny_trial_counts_still_split_and_fit(trials):
    theta, y = double_gaussian_cohort(cells=4, trials=trials)
    result = fit_equation(equation(), (theta, y), 50)
    assert np.isfinite([result.mse, result.training_mse]).all()
    assert result.params.shape == (4, 5)


def test_constant_responses_are_fitted_without_dividing_by_zero():
    theta, _ = double_gaussian_cohort(cells=4)
    flat = np.full_like(theta, 1.5)
    result = fit_equation(equation(), (theta, flat), 100)
    assert np.isfinite(result.params).all()
    assert result.mse < 1e-2


def test_a_single_iteration_still_returns_finite_parameters_no_worse_than_the_start():
    cohort = double_gaussian_cohort(cells=4)
    start = fit_equation(equation(), cohort, 0)
    one = fit_equation(equation(), cohort, 1)
    assert np.isfinite(one.params).all()
    assert one.training_mse <= start.training_mse + 1e-9


@pytest.mark.parametrize(
    "scale",
    [
        1e-3,
        50,
    ],
)
def test_responses_on_very_different_scales_are_fitted_to_the_same_relative_accuracy(scale):
    theta, y = double_gaussian_cohort()
    initial = {"m": 3.0, "b": 0.3 * scale, "a1": 1.5 * scale, "a2": 1.0 * scale, "s": 0.6}
    result = fit_equation(equation(initial=initial), (theta, y * scale), 100)
    assert result.training_mse < 3 * (SIGMA * scale) ** 2


def test_a_declared_amplitude_far_below_the_data_still_reaches_it():
    theta, y = double_gaussian_cohort()
    result = fit_equation(equation(), (theta, 40 * y), 100)
    assert result.training_mse < 3 * (40 * SIGMA) ** 2


def test_parameters_far_from_the_truth_in_declared_values_do_not_stop_a_good_fit():
    theta, y = double_gaussian_cohort()
    bad = {"m": 0.0, "b": 5.0, "a1": 0.1, "a2": 0.1, "s": 2.0}
    result = fit_equation(equation(initial=bad), (theta, y), 100)
    assert result.training_mse < 20 * SIGMA**2


def test_fitting_only_the_training_trials_gives_the_protocols_parameters():
    x, y = double_gaussian_cohort(cells=6)
    train, _ = split_trials(x)
    shifted = equation(initial={**DOUBLE_GAUSSIAN_START, "m": 1.0})
    warm = np.tile(np.array([3.0, 0.3, 1.5, 1.0, 0.6], dtype=np.float32), (6, 1))

    whole = fit_equation(shifted, (x, y), 100, warm=warm)
    alone = fit_equation_training(shifted, x[:, train], y[:, train], 100, warm=warm)

    np.testing.assert_array_equal(alone, whole.params)


def test_the_protocols_relative_step_cap_reaches_a_far_start_that_an_absolute_cap_cannot():
    x, y = double_gaussian_cohort(cells=4)
    far = equation(initial={**DOUBLE_GAUSSIAN_START, "a1": 40.0})

    capped = fit_equation_training(far, x, y, 100, relative_step=False)
    relative = fit_equation_training(far, x, y, 100)

    def error(params):
        return float(np.mean((np.asarray(predict(far.model, x, params)) - y) ** 2))

    assert error(relative) < error(capped)


def test_a_start_that_spikes_between_training_angles_is_never_chosen_for_its_lower_training_error():
    theta, y = double_gaussian_cohort(cells=4, trials=72)
    train, _ = split_trials(theta)
    spiky = equation(
        "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2) "
        "+ h*exp(-0.5*(wrap(theta-c)/0.0005)**2)",
        {**DOUBLE_GAUSSIAN_START, "h": 0.0, "c": 0.0},
    )
    good = fit_equation(equation(), (theta, y), 100, starts=1).params
    gaps = np.sort(np.asarray(theta)[0, np.asarray(train)])
    middle = float(gaps[np.argmax(np.diff(gaps))] + np.max(np.diff(gaps)) / 2)
    names = list(spiky.initial)
    reference = dict(zip(equation().initial, good.T, strict=True))
    needle = np.stack([reference.get(n, np.full(4, {"h": 1e6, "c": middle}.get(n, 0.0))) for n in names], 1)
    params = fit_equation_training(
        spiky, np.asarray(theta)[:, train], np.asarray(y)[:, train], 0, warm=needle, starts=2
    )
    assert np.all(params[:, names.index("h")] < 1e3)

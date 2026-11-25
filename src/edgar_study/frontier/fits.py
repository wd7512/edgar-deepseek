"""Fit candidate equations, warm-started from fitted parents."""

import math
from functools import cache, partial
from itertools import permutations
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import TRIAL_SEED
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.images import ParentFit
from edgar_study.frontier.seed import BASE, BASELINE_ID
from edgar_study.upstream import MSE_SCALE, PENALTY, score_program


def double_gaussian_parent(up):
    seeds = up["seed_programs"]
    return ParentFit(BASELINE_ID, seeds.neuron_model_2_jax, seeds.parameter_estimator_2)


def split_trials(x):
    order = jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(x.shape[1]))
    half = x.shape[1] // 2
    return order[:half], order[half:]


def predict(model, x, params):
    return jax.vmap(model)(x, *params.T)


BASE_NAMES = ("m", "b", "a1", "a2", "s")
BASE_LOWER = (0, 0, 0, 0, 0.01)
BASE_UPPER = (2 * np.pi, np.inf, np.inf, np.inf, np.inf)
BASE_PROBES = ((0.73, 0.24, 2.1, 0.4, 0.38), (3.8, 0.36, 0.8, 1.7, 0.91), (5.1, 1.4, 1.2, 2.4, 0.22))


@cache
def baseline_templates():
    shared: dict[str, Any] = {name: name for name in BASE_NAMES}
    templates = []
    for power, neutral in (
        ("abs(wrap(theta-m))**p/s**p", 2),
        ("(abs(wrap(theta-m))/s)**p", 2),
        ("abs(wrap(theta-m)/s)**p", 2),
        ("(wrap(theta-m)**2/(s**2))**p", 1),
    ):
        templates.append((BASE.replace("(wrap(theta-m)/s)**2", power), {**shared, "p": neutral}))
    for term, extra in (
        ("c*cos(theta)", {"c": 0}),
        ("c*sin(theta)", {"c": 0}),
        ("cc*cos(theta) + ss*sin(theta)", {"cc": 0, "ss": 0}),
        ("ss*sin(theta) + cc*cos(theta)", {"cc": 0, "ss": 0}),
        ("c*cos(theta-phi)", {"c": 0, "phi": 0}),
        ("c*cos(wrap(theta-phi))", {"c": 0, "phi": 0}),
        ("c*sin(theta-phi)", {"c": 0, "phi": 0}),
        ("c*tanh(k*cos(theta-phi))", {"c": 0, "phi": 0, "k": 1}),
    ):
        templates.extend(
            (text, {**shared, **extra})
            for text in (BASE + " + " + term, BASE.replace("b + ", "b + " + term + " + ", 1))
        )
    templates.append((BASE.replace("b + ", "b*(1+k*cos(theta-phi)) + ", 1), {**shared, "k": 0, "phi": 0}))
    first = "exp(-0.5*(wrap(theta-m)/s)**2)"
    shoulder = "exp(-0.5*(wrap(theta-m)/t)**2)"
    second = "a2*exp(-0.5*(wrap(theta-q)/u)**2)"
    independent = {**shared, "t": ["s", 2, 0], "u": "s", "q": ["m", 1, np.pi]}
    templates.extend(
        (
            (f"b + a1*{first} + c*{shoulder} + {second}", {**independent, "c": 0}),
            (f"b + a1*((1-w)*{first} + w*{shoulder}) + {second}", {**independent, "w": 0}),
        )
    )
    result = {}
    for text, bindings in templates:
        template = compile_equation({"equation": text, "initial": dict.fromkeys(bindings, 0.8)})
        result[template.key] = tuple(bindings[name] for name in template.canonical_names)
    return result


def _map_parameters(params, indices, scales, shifts):
    return jnp.asarray(params)[:, indices] * scales + shifts


def offset_bounds(embed):
    options = embed.keywords
    index, scale, shift = (options[key] for key in ("indices", "scales", "shifts"))
    lo, hi = np.array(BASE_LOWER)[index], np.array(BASE_UPPER)[index]
    mapped = scale != 0
    factor = np.where(mapped, scale, 1)
    low = np.where(mapped, np.where(scale < 0, hi, lo) * factor + shift, -np.inf)
    high = np.where(mapped, np.where(scale < 0, lo, hi) * factor + shift, np.inf)
    return jnp.asarray(low), jnp.asarray(high)


def verified_mapping(equation, parent, parent_names, probes, bindings):
    if not isinstance(bindings, dict) or bindings.keys() != equation.initial.keys():
        return None
    indices, scales, shifts = [], [], []
    for value in (bindings[name] for name in equation.initial):
        if type(value) in (float, int) and -100 <= value <= 100:
            name, scale, shift = parent_names[0], 0, value
        elif isinstance(value, str):
            name, scale, shift = value, 1, 0
        elif isinstance(value, list) and len(value) == 3:
            name, scale, shift = value
        else:
            return None
        if name not in parent_names or any(
            type(v) not in (int, float) or not -100 <= v <= 100 for v in (scale, shift)
        ):
            return None
        indices.append(parent_names.index(name))
        scales.append(scale)
        shifts.append(shift)
    embed = partial(
        _map_parameters, indices=np.array(indices), scales=np.array(scales), shifts=np.array(shifts)
    )
    grid = jnp.broadcast_to(jnp.linspace(0.007, 2 * np.pi + 0.007, 72, endpoint=False), (len(probes), 72))
    expected = predict(parent.model, grid, jnp.asarray(probes, dtype=grid.dtype))
    actual = predict(equation.model, grid, embed(jnp.asarray(probes, dtype=grid.dtype)))
    if np.isfinite(expected).all() and np.allclose(actual, expected, rtol=2e-5, atol=3e-6):
        return embed
    return None


def offset_embedding(equation, baseline_model):
    if len(equation.initial) != 6:
        raise ValueError("Expected five baseline parameters and one free offset.")
    probe = jnp.linspace(0, 2 * np.pi, 32, endpoint=False)
    seeds = np.array([[0.73, 0.24, 2.1, 0.4, 0.38], [3.8, 0.36, 0.8, 1.7, 0.91]], dtype=np.float32)
    target = predict(baseline_model, jnp.broadcast_to(probe, (len(seeds), probe.size)), seeds)
    orders = np.array(list(permutations(range(6))))
    for neutral in (0.0, np.pi, 1.0, -np.pi, -1.0, *equation.initial.values()):
        params = np.column_stack((seeds, np.full(len(seeds), neutral, dtype=seeds.dtype)))[:, orders]
        x = jnp.broadcast_to(probe, (params.shape[0] * params.shape[1], probe.size))
        fit = predict(equation.model, x, params.reshape(-1, 6)).reshape(*params.shape[:2], probe.size)
        matches = np.asarray(jnp.all(jnp.isclose(fit, target[:, None, :], rtol=2e-6, atol=2e-6), axis=(0, 2)))
        if matches.any():
            return orders[np.flatnonzero(matches)[0]], neutral
    raise ValueError("Could not reproduce the double Gaussian at a neutral offset.")


def _fit_from_starts(equation, cohort, max_iter, starts, *, bounds=None):
    x, y = cohort
    train, held = split_trials(x)
    xt, yt = x[:, train], y[:, train]

    def errors(params):
        return jnp.mean((predict(equation.model, xt, params) - yt) ** 2, axis=1)

    def project(params):
        return jnp.clip(params, *bounds) if bounds is not None else params

    def bounded(theta, *params):
        return equation.model(theta, *project(jnp.stack(params)))

    params = None
    for start, inherited in starts:
        if not np.allclose(predict(equation.model, xt, start), inherited, rtol=3e-4, atol=3e-5):
            raise ValueError("Nested start does not match the fitted parent on training angles.")
        if params is None:
            params = start
        else:
            params = jnp.where((errors(start) < errors(params))[:, None], start, params)
        fit = fitting.fit_lm_bounded(
            bounded, start, xt, yt, max_iterations=min(max_iter, fitting.LM_ITERATIONS)
        )
        fitted = project(jnp.asarray(fit.params))
        fit_errors = errors(fitted)
        params = jnp.where(
            (jnp.isfinite(fit_errors) & (fit_errors < errors(params)))[:, None], fitted, params
        )
    if params is None:
        raise ValueError("No nested start available.")
    mse = float(jnp.mean((predict(equation.model, x[:, held], params) - y[:, held]) ** 2))
    count = params.shape[1]
    return mse * MSE_SCALE + count * PENALTY, mse, count, params


def _parent_start(up, parent, embed, cohort, max_iter):
    x, y = cohort
    if parent.solve is None:
        score, _, _, params = score_program(up, parent.model, parent.estimator, x, y, max_iter)
    else:
        score, _, _, params = parent.solve(cohort, max_iter)
    if not math.isfinite(score):
        raise ValueError(f"Could not fit {parent.node_id}.")
    params = jnp.asarray(params)
    if parent.node_id == BASELINE_ID:
        params = jnp.clip(params, jnp.asarray(BASE_LOWER), jnp.asarray(BASE_UPPER))
    train, _ = split_trials(x)
    return embed(params), predict(parent.model, x[:, train], params)


def fit_nested(up, equation, cohort, max_iter, *, mappings, bounds=None):
    starts = (_parent_start(up, parent, embed, cohort, max_iter) for parent, embed in mappings)
    return _fit_from_starts(equation, cohort, max_iter, starts, bounds=bounds)


def fit_offset(up, equation, cohort, max_iter, *, embedding=None):
    baseline = double_gaussian_parent(up)
    order, neutral = offset_embedding(equation, baseline.model) if embedding is None else embedding
    embed = partial(
        _map_parameters,
        indices=np.minimum(order, 4),
        scales=order != 5,
        shifts=np.where(order == 5, neutral, 0),
    )
    start = _parent_start(up, baseline, embed, cohort, max_iter)
    lo = jnp.asarray((*BASE_LOWER, -np.inf))[order]
    hi = jnp.asarray((*BASE_UPPER, np.inf))[order]
    return _fit_from_starts(equation, cohort, max_iter, (start,), bounds=(lo, hi))


def measure(up, model, cohort, max_iter, *, nested_offset=False, solve=None, initialization="estimator"):
    x, y = cohort
    if nested_offset:
        solve = partial(fit_offset, up, model)
        initialization = "fitted_double_gaussian"
    if solve is not None:
        score, mse, count, params = solve(cohort, max_iter)
    else:
        score, mse, count, params = score_program(up, model.model, model.estimator, x, y, max_iter)
    if not math.isfinite(score):
        raise ValueError("Nonfinite fit.")
    train, held = split_trials(x)
    angles = x[:, held]
    fitted = predict(model.model, angles, params)
    errors = np.asarray((fitted - y[:, held]) ** 2)
    training_mse = float(jnp.mean((predict(model.model, x[:, train], params) - y[:, train]) ** 2))
    if not math.isfinite(training_mse) or not np.isfinite(errors).all():
        raise ValueError("Nonfinite predictions.")
    bins = np.floor((np.rad2deg(np.asarray(angles)) % 360) / 30).astype(int)
    return {
        "score": score,
        "mse": mse,
        "training_mse": training_mse,
        "initialization": initialization,
        "parameters": count,
        "angle_mse": [float(errors[bins == i].mean()) if (bins == i).any() else None for i in range(12)],
    }

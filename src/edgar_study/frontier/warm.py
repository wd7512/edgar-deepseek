"""Warm starts: carry a parent's fitted parameters into a child equation.

A candidate start comes from each parent, by a verified model-declared map when there is one and by
matching parameter names otherwise. Each neuron then takes the candidate with the lowest training error.
A start is only a start: the fit still picks its best start on training error, so a poor map costs one
start and cannot bias the held-out score.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from edgar_study.frontier.protocol import predict, split_trials


@dataclass(frozen=True)
class Parent:
    node_id: str
    names: tuple[str, ...]
    model: Callable
    probes: np.ndarray
    fit: Callable


def _binding(value, parent_names):
    if type(value) in (float, int):
        name, scale, shift = parent_names[0], 0, value
    elif isinstance(value, str):
        name, scale, shift = value, 1, 0
    elif isinstance(value, list) and len(value) == 3:
        name, scale, shift = value
    else:
        return None
    numbers = (scale, shift) if name in parent_names else ()
    if name not in parent_names or any(type(v) not in (int, float) or not -100 <= v <= 100 for v in numbers):
        return None
    return parent_names.index(name), scale, shift


def verified_map(equation, parent):
    """Return a function from the parent's parameters to the child's, if the model's map reproduces the parent."""
    nesting = equation.nesting if isinstance(equation.nesting, dict) else {}
    bindings = nesting.get(parent.node_id)
    if not isinstance(bindings, dict) or bindings.keys() != equation.initial.keys():
        return None
    parts = [_binding(bindings[name], parent.names) for name in equation.initial]
    if None in parts:
        return None
    columns = list(zip(*parts, strict=True))
    indices = np.array(columns[0], dtype=int)
    scales, shifts = np.array(columns[1], dtype=float), np.array(columns[2], dtype=float)

    def embed(params):
        return jnp.asarray(params)[:, indices] * scales + shifts

    probes = jnp.asarray(parent.probes, dtype=float)
    grid = jnp.broadcast_to(jnp.linspace(0.007, 2 * np.pi + 0.007, 72, endpoint=False), (len(probes), 72))
    expected = np.asarray(predict(parent.model, grid, probes))
    actual = np.asarray(predict(equation.model, grid, embed(probes)))
    if (
        np.isfinite(expected).all()
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, rtol=2e-5, atol=3e-6)
    ):
        return embed
    return None


def named_map(equation, parent):
    """Return a function that copies the parent's value for every parameter name the child shares."""
    shared = [
        (i, parent.names.index(name)) for i, name in enumerate(equation.initial) if name in parent.names
    ]
    if not shared:
        return None
    base = np.array(list(equation.initial.values()), dtype=float)

    def embed(params):
        out = np.tile(base, (len(params), 1))
        for child, source in shared:
            out[:, child] = np.asarray(params)[:, source]
        return jnp.asarray(out)

    return embed


def warm_start(equation, parents, cohort, max_iter):
    """Return the child's best candidate start per neuron in its declared order, or None if there is none."""
    candidates = []
    for parent in parents:
        embed = verified_map(equation, parent) or named_map(equation, parent)
        if embed is not None:
            candidates.append(np.asarray(embed(parent.fit(cohort, max_iter)), dtype=np.float32))
    if not candidates:
        return None
    x, y = cohort
    train, _ = split_trials(x)
    xt, yt = jnp.asarray(x)[:, train], jnp.asarray(y)[:, train]
    costs = np.stack(
        [
            np.asarray(jnp.mean((predict(equation.model, xt, jnp.asarray(c)) - yt) ** 2, axis=1))
            for c in candidates
        ]
    )
    costs = np.where(np.isfinite(costs), costs, np.inf)
    best = np.argmin(costs, axis=0)
    return np.stack(candidates)[best, np.arange(len(best))]


def declared_probes(equation):
    """Parameter vectors around the equation's declared start, used to verify nesting maps against it."""
    initial = np.array(tuple(equation.initial.values()), dtype=float)
    perturb = np.random.default_rng(0).uniform(-0.3, 0.3, size=(4, len(initial)))
    return np.concatenate((initial[None, :], initial + perturb * (0.2 + np.abs(initial))))

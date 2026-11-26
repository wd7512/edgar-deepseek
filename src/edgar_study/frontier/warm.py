"""Warm starts: carry a parent's fitted parameters into a child equation.

A candidate start comes from each parent through a map that reproduces the parent: the model's declared
map when it does, otherwise a completed or repaired one found by search, otherwise parameters matched by
name. Each neuron then takes the candidate with the lowest training error. A start is only a start: the
fit still picks its best start on training error, so a poor map costs one start and cannot bias the
held-out score.
"""

import heapq
import itertools
import re
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


SEARCH_LIMIT = 64
NEUTRAL = (0.0, 1.0)


@dataclass(frozen=True)
class Nesting:
    """A map from a parent's parameters to a child's that reproduces the parent's predictions."""

    bindings: dict
    embed: Callable
    declared: bool


def _embedding(parts):
    columns = list(zip(*parts, strict=True))
    indices = np.array(columns[0], dtype=int)
    scales, shifts = np.array(columns[1], dtype=float), np.array(columns[2], dtype=float)

    def embed(params):
        return jnp.asarray(params)[:, indices] * scales + shifts

    return embed


def _reproducing(equation, parent, embeds):
    """Index of the first embedding whose child predictions match the parent's on its probes, or None."""
    probes = jnp.asarray(parent.probes, dtype=float)
    grid = jnp.broadcast_to(jnp.linspace(0.007, 2 * np.pi + 0.007, 72, endpoint=False), (len(probes), 72))
    expected = np.asarray(predict(parent.model, grid, probes))
    if not np.isfinite(expected).all():
        return None
    stacked = jnp.concatenate([embed(probes) for embed in embeds])
    actual = np.asarray(predict(equation.model, jnp.tile(grid, (len(embeds), 1)), stacked))
    actual = actual.reshape(len(embeds), *expected.shape)
    matches = np.isfinite(actual).all(axis=(1, 2)) & np.isclose(actual, expected, rtol=2e-5, atol=3e-6).all(
        axis=(1, 2)
    )
    found = np.flatnonzero(matches)
    return int(found[0]) if len(found) else None


def verified_map(equation, parent):
    """Return a function from the parent's parameters to the child's, if the model's map reproduces the parent."""
    nesting = equation.nesting if isinstance(equation.nesting, dict) else {}
    bindings = nesting.get(parent.node_id)
    if not isinstance(bindings, dict) or bindings.keys() != equation.initial.keys():
        return None
    parts = [_binding(bindings[name], parent.names) for name in equation.initial]
    if None in parts:
        return None
    embed = _embedding(parts)
    return embed if _reproducing(equation, parent, [embed]) is not None else None


def _options(name, declared, parent_names):
    options = []
    if name in declared and _binding(declared[name], parent_names) is not None:
        options.append(declared[name])
    stem = re.sub(r"_?\d+$", "", name)
    options += [candidate for candidate in (name, stem) if candidate in parent_names]
    options += list(NEUTRAL)
    unique = []
    for option in options:
        if option not in unique:
            unique.append(option)
    return unique


def _cheapest_first(sizes):
    """Index tuples into option lists, in order of total index, so the likeliest maps come first."""
    start = (0,) * len(sizes)
    heap, seen = [(0, start)], {start}
    while heap:
        _, combo = heapq.heappop(heap)
        yield combo
        for i, size in enumerate(sizes):
            if combo[i] + 1 < size:
                step = (*combo[:i], combo[i] + 1, *combo[i + 1 :])
                if step not in seen:
                    seen.add(step)
                    heapq.heappush(heap, (sum(step), step))


def complete_map(equation, parent, limit=SEARCH_LIMIT):
    """Find a map that reproduces the parent: the declared one if it does, else the first of up to `limit`
    candidates built from declared bindings, same or numbered names (s1, s2 from s), and 0 or 1."""
    embed = verified_map(equation, parent)
    nesting = equation.nesting if isinstance(equation.nesting, dict) else {}
    declared = nesting.get(parent.node_id)
    declared = declared if isinstance(declared, dict) else {}
    if embed is not None:
        return Nesting(dict(declared), embed, declared=True)
    names = tuple(equation.initial)
    options = [_options(name, declared, parent.names) for name in names]
    combos = list(itertools.islice(_cheapest_first([len(o) for o in options]), limit))
    bindings = [
        {name: options[i][pick] for i, (name, pick) in enumerate(zip(names, combo, strict=True))}
        for combo in combos
    ]
    embeds = [_embedding([_binding(b[name], parent.names) for name in names]) for b in bindings]
    found = _reproducing(equation, parent, embeds)
    if found is None:
        return None
    return Nesting(bindings[found], embeds[found], declared=False)


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
        nesting = complete_map(equation, parent)
        embed = nesting.embed if nesting is not None else named_map(equation, parent)
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

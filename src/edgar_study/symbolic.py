"""Expression trees for a symbolic-regression search that needs no LLM.

A tree is nested tuples: ("theta",), ("const", value), ("param", name, kind), or (op, *children).
Every parameter leaf is fitted per neuron, like the keyword arguments of an LLM-written program, so a
tree's parameter count is its number of distinct parameter names. A parameter's kind sets its clip
range and how the generic estimator picks its starting value.
"""

import math
import re
from typing import NamedTuple

import numpy as np

MAXSIZE = 45
MAX_DEPTH = 14
MAX_PARAMETERS = 16


class Kind(NamedTuple):
    bounds: tuple[float, float]
    default: float


KINDS = {
    "pref": Kind((0.0, 2 * math.pi), math.pi),
    "level": Kind((0.0, 100.0), 0.0),
    "scale": Kind((0.0, 100.0), 1.0),
    "opposite": Kind((0.0, 100.0), 0.0),
    "width": Kind((0.01, 10.0), 1.0),
    "exponent": Kind((0.1, 8.0), 2.0),
    "free": Kind((-100.0, 100.0), 1.0),
    "shift": Kind((-100.0, 100.0), 0.0),
}
NEUTRAL = {"add": "shift", "sub": "shift", "mul": "free", "div": "free"}
PROMOTABLE = {0.0: "shift", 1.0: "free", 2.0: "exponent"}
MOVES = {
    "point": 0.18,
    "insert": 0.22,
    "promote": 0.10,
    "leaf": 0.10,
    "subtree": 0.10,
    "shrink": 0.10,
    "hoist": 0.03,
    "share": 0.05,
    "split": 0.05,
    "retag": 0.07,
}
RANDOM_KINDS = ("free", "level", "scale", "pref", "width", "exponent")
CONSTANTS = (0.5, 1.0, 2.0, math.pi, -0.5)

OPS = {
    "add": (2, "({0} + {1})"),
    "sub": (2, "({0} - {1})"),
    "mul": (2, "({0} * {1})"),
    "div": (2, "_div({0}, {1})"),
    "pow": (2, "_pow({0}, {1})"),
    "exp": (1, "_exp({0})"),
    "cos": (1, "{m}.cos({0})"),
    "sin": (1, "{m}.sin({0})"),
    "abs": (1, "{m}.abs({0})"),
    "square": (1, "({0}) ** 2"),
    "wrap": (1, "_wrap({0})"),
    "ifneg": (3, "{m}.where({0} < 0, {1}, {2})"),
}
INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}
BY_ARITY = {n: [op for op, (arity, _) in OPS.items() if arity == n] for n in (1, 2, 3)}

THETA = ("theta",)


def _param(kind, k=1):
    return ("param", f"{kind}_{k}", kind)


def _gaussian(offset):
    distance = ("abs", ("wrap", ("sub", THETA, offset)))
    return ("exp", ("mul", ("const", -0.5), ("pow", ("div", distance, _param("width")), ("const", 2.0))))


SINGLE_GAUSSIAN = ("add", _param("level"), ("mul", _param("scale"), _gaussian(_param("pref"))))
DOUBLE_GAUSSIAN = (
    "add",
    SINGLE_GAUSSIAN,
    ("mul", _param("opposite"), _gaussian(("add", _param("pref"), ("const", math.pi)))),
)
SEEDS = (SINGLE_GAUSSIAN, DOUBLE_GAUSSIAN)


def nodes(tree, path=()):
    """Every (path, subtree) pair, root first."""
    found = [(path, tree)]
    if tree[0] in OPS:
        for i, child in enumerate(tree[1:]):
            found += nodes(child, (*path, i))
    return found


def replace(tree, path, new):
    if not path:
        return new
    children = list(tree[1:])
    children[path[0]] = replace(children[path[0]], path[1:], new)
    return (tree[0], *children)


def size(tree):
    return len(nodes(tree))


def depth(tree):
    return 1 + max((depth(c) for c in tree[1:]), default=0) if tree[0] in OPS else 1


def uses_theta(tree):
    return any(node == THETA for _, node in nodes(tree))


def parameters(tree):
    """Distinct (name, kind) pairs in order of first appearance, which is the signature order."""
    seen = {}
    for _, node in nodes(tree):
        if node[0] == "param" and node[1] not in seen:
            seen[node[1]] = node[2]
    return list(seen.items())


def canonical(tree):
    """Rename parameters to kind_1, kind_2, ... in order of first appearance."""
    counts, names = {}, {}
    for name, kind in parameters(tree):
        counts[kind] = counts.get(kind, 0) + 1
        names[name] = f"{kind}_{counts[kind]}"

    def rename(node):
        if node[0] == "param":
            return ("param", names[node[1]], node[2])
        if node[0] in OPS:
            return (node[0], *(rename(c) for c in node[1:]))
        return node

    return rename(tree)


def valid(tree):
    return (
        size(tree) <= MAXSIZE
        and depth(tree) <= MAX_DEPTH
        and uses_theta(tree)
        and 1 <= len(parameters(tree)) <= MAX_PARAMETERS
    )


def random_leaf(rng, kinds=RANDOM_KINDS):
    roll = rng.random()
    if roll < 0.4:
        return THETA
    if roll < 0.85:
        return ("param", f"new_{rng.integers(1 << 30)}", str(rng.choice(kinds)))
    return ("const", float(CONSTANTS[rng.integers(len(CONSTANTS))]))


def _grow(rng, max_depth):
    if max_depth <= 1 or rng.random() < 0.3:
        return random_leaf(rng)
    op = str(rng.choice(list(OPS)))
    return (op, *(_grow(rng, max_depth - 1) for _ in range(OPS[op][0])))


def random_tree(rng, depth=4):
    """A random valid tree, grown to at most `depth` levels."""
    while True:
        tree = canonical(_grow(rng, depth))
        if valid(tree):
            return tree


def _fresh(tree, tag):
    """The same tree with parameter names made unique, so a grafted subtree does not share them by accident."""
    if tree[0] == "param":
        return ("param", f"{tree[1]}_{tag}", tree[2])
    if tree[0] in OPS:
        return (tree[0], *(_fresh(c, tag) for c in tree[1:]))
    return tree


def _mutate_once(rng, tree, moves):
    spots = nodes(tree)
    path, node = spots[rng.integers(len(spots))]
    weights = np.array([MOVES[m] for m in moves])
    move = moves[rng.choice(len(moves), p=weights / weights.sum())]
    if move == "point":
        arity = OPS[node[0]][0] if node[0] in OPS else 0
        new = (str(rng.choice(BY_ARITY[arity])), *node[1:]) if arity else random_leaf(rng)
    elif move == "insert":
        op = str(rng.choice(list(NEUTRAL)))
        new = (op, node, ("param", f"new_{rng.integers(1 << 30)}", NEUTRAL[op]))
    elif move == "promote":
        constants = [(p, n) for p, n in spots if n[0] == "const" and n[1] in PROMOTABLE]
        if not constants:
            return tree
        path, node = constants[rng.integers(len(constants))]
        new = ("param", f"new_{rng.integers(1 << 30)}", PROMOTABLE[node[1]])
    elif move == "leaf":
        new = random_leaf(rng)
    elif move == "subtree":
        new = _grow(rng, 2)
    elif move == "hoist":
        inner = [n for _, n in nodes(node) if n[0] in OPS and uses_theta(n)]
        if not inner:
            return tree
        return inner[rng.integers(len(inner))]
    elif move == "shrink":
        if node[0] not in OPS:
            return tree
        new = node[1 + rng.integers(len(node) - 1)]
    elif move == "share":
        if node[0] != "param":
            return tree
        same = [name for name, kind in parameters(tree) if kind == node[2] and name != node[1]]
        if not same:
            return tree
        new = ("param", str(rng.choice(same)), node[2])
    elif move == "split":
        if node[0] != "param":
            return tree
        new = ("param", f"split_{rng.integers(1 << 30)}", node[2])
    else:
        if node[0] != "param":
            return tree
        return canonical(_retag(tree, node[1], str(rng.choice(RANDOM_KINDS))))
    return replace(tree, path, new)


def _retag(tree, name, kind):
    if tree[0] == "param" and tree[1] == name:
        return ("param", f"retag_{name}", kind)
    if tree[0] in OPS:
        return (tree[0], *(_retag(c, name, kind) for c in tree[1:]))
    return tree


def mutate(rng, tree, steps=1, moves=tuple(MOVES)):
    """Apply `steps` random edits drawn from `moves` (weighted by MOVES), each retried until the result
    is valid; the output is canonical. `insert` and `promote` are neutral: the child starts as the parent."""
    for _ in range(steps):
        for _ in range(30):
            candidate = canonical(_mutate_once(rng, tree, moves))
            if valid(candidate) and candidate != tree:
                tree = candidate
                break
    return tree


def crossover(rng, a, b):
    """Graft a random subtree of `b` into a random place in `a`; returns `a` if no valid child is found."""
    for attempt in range(30):
        targets, donors = nodes(a), nodes(b)
        path = targets[rng.integers(len(targets))][0]
        donor = _fresh(donors[rng.integers(len(donors))][1], f"x{attempt}")
        child = canonical(replace(a, path, donor))
        if valid(child):
            return child
    return a


def _expr(tree, m):
    if tree == THETA:
        return "theta"
    if tree[0] == "const":
        return repr(float(tree[1]))
    if tree[0] == "param":
        return tree[1]
    return OPS[tree[0]][1].format(*(_expr(c, m) for c in tree[1:]), m=m)


def _source(tree, m, module):
    params = parameters(tree)
    signature = ", ".join(f"{name}={KINDS[kind].default!r}" for name, kind in params)
    lines = [f"import {module}\n", "", f"def neuron_model(theta, {signature}):"]
    for name, kind in params:
        low, high = KINDS[kind].bounds
        lines.append(f"    {name} = {m}.clip({name}, {low!r}, {high!r})")
    lines += [
        "",
        "    def _div(a, b):",
        f"        return a / {m}.where({m}.abs(b) < 1e-6, {m}.where(b < 0, -1e-6, 1e-6), b)",
        "",
        "    def _pow(a, b):",
        f"        return ({m}.abs(a) + 1e-12) ** {m}.clip(b, -8.0, 8.0)",
        "",
        "    def _exp(a):",
        f"        return {m}.exp({m}.clip(a, -50.0, 50.0))",
        "",
        "    def _wrap(a):",
        f"        return {m}.arctan2({m}.sin(a), {m}.cos(a))",
        "",
        f"    return {_expr(tree, m)} + 0.0 * theta",
        "",
    ]
    return "\n".join(lines)


def to_numpy_source(tree):
    """The tree as an upstream-style NumPy `neuron_model` with keyword defaults and clipped parameters."""
    return _source(tree, "np", "numpy as np")


def to_jax_source(tree):
    """The same program in JAX, as the search fits it."""
    return _source(tree, "jnp", "jax.numpy as jnp")


def to_infix(tree):
    """A readable formula that `parse` reads back."""
    if tree == THETA:
        return "theta"
    if tree[0] == "const":
        return repr(float(tree[1]))
    if tree[0] == "param":
        return tree[1]
    args = [to_infix(c) for c in tree[1:]]
    if tree[0] in INFIX:
        return f"({args[0]} {INFIX[tree[0]]} {args[1]})"
    return f"{tree[0]}({', '.join(args)})"


ESTIMATOR = """import numpy as np


def parameter_estimator(theta, spike_counts):
    n_bins = 20
    bins = np.clip((np.mod(theta, 2 * np.pi) * n_bins / (2 * np.pi)).astype(np.int32), 0, n_bins - 1)
    sums = np.bincount(bins, weights=spike_counts, minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    filled = counts > 0
    curve = np.full(n_bins, np.mean(sums[filled] / counts[filled]) if filled.any() else 0.0)
    curve[filled] = sums[filled] / counts[filled]
    peak = int(np.argmax(curve))
    floor = float(np.min(curve))
    span = float(np.max(curve) - floor)
    above = curve[(np.arange(-5, 6) + peak) % n_bins] >= floor + span / 2
    starts = {{
        "pref": (peak + 0.5) * 2 * np.pi / n_bins,
        "level": floor,
        "scale": span,
        "opposite": float(curve[(peak + n_bins // 2) % n_bins] - floor),
        "width": 2 * np.pi * np.sum(above) / n_bins / 2.3548,
        "exponent": 2.0,
        "free": 1.0,
        "shift": 0.0,
    }}
    kinds = {kinds!r}
    low = np.array({lows!r})
    high = np.array({highs!r})
    return np.clip(np.array([starts[k] for k in kinds], dtype=float), low, high)
"""


def estimator_source(tree):
    """A generic `parameter_estimator` that starts each parameter from its kind, using a binned tuning curve."""
    kinds = [kind for _, kind in parameters(tree)]
    return ESTIMATOR.format(
        kinds=kinds,
        lows=[KINDS[k].bounds[0] for k in kinds],
        highs=[KINDS[k].bounds[1] for k in kinds],
    )


TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|([A-Za-z_]\w*)|(.))")
FUNCTIONS = {"exp", "cos", "sin", "abs", "square", "wrap", "ifneg", "sqrt", "neg", "cube"}
KIND_NAME = re.compile(r"(" + "|".join(KINDS) + r")_\d+$")


def parse(text):
    """Read an infix formula (PySR style or `to_infix`) into a canonical tree.

    x, x0 and theta are the stimulus angle; kind_N names keep their kind and any other name is a
    free parameter. `^` is `pow`, which raises the magnitude of its base.
    """
    tokens = [(n, i, o) for n, i, o in TOKEN.findall(text) if n or i or o.strip()]
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else ("", "", "")

    def take(symbol=None):
        nonlocal pos
        token = peek()
        if symbol is not None and token[2] != symbol:
            raise ValueError(f"expected {symbol!r} at token {pos} in {text!r}")
        pos += 1
        return token

    def expr():
        node = term()
        while peek()[2] in "+-" and peek()[2]:
            op = "add" if take()[2] == "+" else "sub"
            node = (op, node, term())
        return node

    def term():
        node = unary()
        while peek()[2] in "*/" and peek()[2]:
            op = "mul" if take()[2] == "*" else "div"
            node = (op, node, unary())
        return node

    def unary():
        if peek()[2] == "-":
            take()
            return ("mul", ("const", -1.0), unary())
        if peek()[2] == "+":
            take()
            return unary()
        return power()

    def power():
        node = atom()
        if peek()[2] == "^":
            take()
            node = ("pow", node, unary())
        return node

    def atom():
        number, name, symbol = take()
        if number:
            return ("const", float(number))
        if symbol == "(":
            node = expr()
            take(")")
            return node
        if not name:
            raise ValueError(f"unexpected {symbol!r} in {text!r}")
        if name in FUNCTIONS and peek()[2] == "(":
            take("(")
            args = [expr()]
            while peek()[2] == ",":
                take()
                args.append(expr())
            take(")")
            if name == "sqrt":
                return ("pow", args[0], ("const", 0.5))
            if name == "cube":
                return ("mul", args[0], ("square", args[0]))
            if name == "neg":
                return ("mul", ("const", -1.0), args[0])
            return (name, *args)
        if name in ("x", "x0", "theta"):
            return THETA
        if name == "pi":
            return ("const", math.pi)
        found = KIND_NAME.match(name)
        return ("param", name, found.group(1) if found else "free")

    tree = expr()
    if pos != len(tokens):
        raise ValueError(f"unexpected trailing input in {text!r}")
    return canonical(tree)


def _phases(tree):
    """Names of parameters added to or subtracted from theta directly, which act as preferred angles."""
    found = set()
    for _, node in nodes(tree):
        if node[0] in ("add", "sub") and THETA in node[1:]:
            found |= {c[1] for c in node[1:] if c[0] == "param"}
    return found


def from_pysr(text):
    """A tree from a PySR template equation `f = ...; q1 = [...]`, where #1 is theta and #k+1 is the
    k-th per-neuron parameter. Global constants stay fixed. Parameters added to theta become `pref`
    parameters, so the generic estimator starts them at the preferred angle; the rest are `free`."""
    body = text.split(";")[0].split("=", 1)[1]
    body = re.sub(r"#(\d+)", lambda m: "theta" if m.group(1) == "1" else f"q{int(m.group(1)) - 1}", body)
    tree = parse(body)
    for name in _phases(tree):
        tree = _retag(tree, name, "pref")
    return canonical(tree)

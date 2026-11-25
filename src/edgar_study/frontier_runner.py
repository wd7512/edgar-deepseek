"""Run score-blind equation exploration and one independent confirmation pass."""

import ast
import asyncio
import fcntl
import json
import logging
import math
import operator
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache, partial
from itertools import permutations
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import TRIAL_SEED, load_dataset, prepare
from edgar_study.frontier_images import ParentFit, prepare_images
from edgar_study.frontier_seed import BASE, BASELINE_ID, baseline_node
from edgar_study.go_client import GoClient, Ledger, fetch_models, is_free, load_key
from edgar_study.upstream import MSE_SCALE, PENALTY, import_upstream, score_program

MODEL = "hy4-flash-free"
LANES = {"frontier": 0.6, "detour": 0.3, "root": 0.1}
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
FUNCTIONS = {name: getattr(jnp, name) for name in ("sin", "cos", "exp", "tanh", "abs", "sqrt")}
FUNCTIONS["wrap"] = lambda x: jnp.arctan2(jnp.sin(x), jnp.cos(x))
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Idea:
    node: str
    branch: str
    mechanism: str
    text: str
    routes: tuple[tuple[str, ...], ...] = ()
    lane: str = "frontier"
    label: str = ""


DEFAULT_IDEAS = (
    Idea(
        "width",
        "flanks",
        "asymmetry",
        "Give the main peak separate left and right widths; leave the second peak and baseline unchanged.",
        label="Asymmetric main-peak widths",
    ),
    Idea(
        "tip",
        "peaks",
        "sharpness",
        "Free the exponent on the main peak's wrapped angular distance inside its exponential; "
        "keep its center fixed to the same parameter and leave the second peak and baseline unchanged.",
        label="Free the main-peak shape exponent",
    ),
    Idea(
        "offset",
        "second_peak",
        "offset",
        "Replace the fixed 180° separation with a fitted angular offset; keep both peak kernels and baseline.",
        label="Free second peak from 180°",
    ),
    Idea(
        "drift",
        "baseline",
        "drift",
        "Add one full-turn sinusoid to the constant baseline; leave both peak kernels unchanged.",
        label="Add one baseline harmonic",
    ),
    Idea(
        "harmonic",
        "harmonics",
        "harmonic",
        "Replace both localized peaks with a finite circular Fourier sum and a constant baseline.",
        lane="root",
        label="Replace peaks with a Fourier sum",
    ),
    Idea(
        "window",
        "windows",
        "window",
        "Replace both Gaussian kernels with periodic raised-cosine windows that are zero outside their support.",
        lane="root",
        label="Replace Gaussians with cosine windows",
    ),
    Idea(
        "blend",
        "flanks",
        "blend",
        "Replace the main Gaussian with a smooth blend of two co-centered kernels to test for shoulders; "
        "retain the second peak and constant baseline.",
        lane="detour",
        label="Blend two co-centered main-peak kernels",
    ),
    Idea(
        "width-drift",
        "baseline",
        "interaction",
        "Combine separate left and right main-peak widths with one baseline harmonic; "
        "keep their parameters independent and leave the second peak unchanged.",
        (("width", "drift"),),
        "detour",
        "Asymmetric main-peak widths + baseline harmonic",
    ),
    Idea(
        "peak-pair",
        "peaks",
        "interaction",
        "Combine the main peak's free shape exponent with a fitted offset for the second peak; "
        "leave the baseline and second peak kernel unchanged.",
        (("tip", "offset"),),
        label="Main-peak shape exponent + second-peak offset",
    ),
)


@dataclass(frozen=True)
class Equation:
    text: str
    initial: dict[str, float]
    key: str
    model: object
    estimator: object
    canonical_names: tuple[str, ...]
    nesting: object


def compile_equation(proposal):
    """Compile a small numeric expression tree without executing generated Python."""
    try:
        text, initial = proposal["equation"], proposal["initial"]
        if not isinstance(text, str) or len(text) > 600 or not isinstance(initial, dict):
            raise ValueError("Expected a short equation and a parameter object.")
        if not 1 <= len(initial) <= 12:
            raise ValueError("Expected one to twelve free parameters.")
        for name, value in initial.items():
            if (
                not name.isidentifier()
                or not name.isascii()
                or name in {"pi", "theta", *FUNCTIONS}
                or type(value) not in (float, int)
                or not -100 <= value <= 100
            ):
                raise ValueError("Invalid parameter name or initial value.")
        expr = ast.parse(text, mode="eval").body
    except (KeyError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid equation object.") from exc
    if sum(1 for _ in ast.walk(expr)) > 100:
        raise ValueError("Equation is too large.")
    used = set()

    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int) and -100 <= node.value <= 100:
            return ("number", float(node.value))
        if isinstance(node, ast.Name) and node.id in {"pi", "theta", *initial}:
            used.add(node.id)
            return ("name", node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return (type(node.op), build(node.left), build(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return ("neg", build(node.operand)) if isinstance(node.op, ast.USub) else build(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return ("call", node.func.id, build(node.args[0]))
        raise ValueError("Expression uses an unsupported operation.")

    tree = build(expr)
    if set(initial) != used - {"pi", "theta"} or "theta" not in used:
        raise ValueError("Equation must use theta and exactly the listed parameters.")

    def shape(tree):
        op, *parts = tree
        if op == "name" and parts[0] in initial:
            return "parameter"
        if op in OPS:
            children = [shape(part) for part in parts]
            if op in (ast.Add, ast.Mult):
                children.sort()
            return repr((op.__name__, children))
        if op in {"call", "neg"}:
            return repr((op, *parts[:-1], shape(parts[-1])))
        return repr(tree)

    seen = {}

    def canonical(tree):
        op, *parts = tree
        if op == "name" and parts[0] in initial:
            return ("parameter", seen.setdefault(parts[0], len(seen)))
        if op in OPS:
            children = sorted(parts, key=shape) if op in (ast.Add, ast.Mult) else parts
            return (op.__name__, *(canonical(part) for part in children))
        if op in {"call", "neg"}:
            return (op, *parts[:-1], canonical(parts[-1]))
        return tree

    def evaluate(tree, values):
        op, *parts = tree
        if op == "number":
            return parts[0]
        if op == "name":
            return values[parts[0]]
        if op == "call":
            return FUNCTIONS[parts[0]](evaluate(parts[1], values))
        if op == "neg":
            return -evaluate(parts[0], values)
        return OPS[op](*(evaluate(part, values) for part in parts))

    names = tuple(initial)

    def model(theta, *params):
        values = {"theta": theta, "pi": jnp.pi, **dict(zip(names, params, strict=True))}
        return jnp.broadcast_to(evaluate(tree, values), jnp.shape(theta))

    def estimator(_theta, _response):
        return tuple(float(initial[name]) for name in names)

    grid = jnp.linspace(-2.9, 2.9, 37)
    values = jnp.asarray(estimator(None, None), dtype=float)
    for params in (values, values + 0.071):
        a, b = np.asarray(model(grid, *params)), np.asarray(model(grid + 2 * jnp.pi, *params))
        if not np.isfinite(a).all() or not np.allclose(a, b, atol=2e-4, rtol=2e-4):
            raise ValueError("Equation failed the finite periodicity probe.")
    key = repr(canonical(tree))
    return Equation(text, dict(initial), key, model, estimator, tuple(seen), proposal.get("nesting"))


def validate_graph(ideas):
    """Reject invalid routes and cycles in the expanded prerequisite graph."""
    nodes = {idea.node: idea for idea in ideas}
    if len(nodes) != len(ideas):
        raise ValueError("Duplicate idea IDs.")
    if BASELINE_ID in nodes:
        raise ValueError(f"{BASELINE_ID!r} is reserved for the starting equation.")
    active, complete = set(), set()

    def visit(node):
        if node in active:
            raise ValueError("Prerequisite cycle.")
        if node in complete:
            return
        if node not in nodes:
            raise ValueError(f"Unknown prerequisite {node!r}.")
        idea = nodes[node]
        if idea.lane not in LANES or (idea.lane == "root" and idea.routes):
            raise ValueError("Invalid exploration lane.")
        active.add(node)
        for route in idea.routes:
            if not route or len(set(route)) != len(route):
                raise ValueError("A route needs distinct prerequisites.")
            for parent in route:
                visit(parent)
        active.remove(node)
        complete.add(node)

    for node in nodes:
        visit(node)


def _pick(weighted, rng):
    keys = list(weighted)
    weights = np.array(list(weighted.values()), dtype=float)
    probabilities = dict(zip(keys, map(float, weights / weights.sum()), strict=True))
    return keys[int(rng.choice(len(keys), p=list(probabilities.values())))], probabilities


def draw(ideas, references, attempts, rng, depth=None):
    """Sample an eligible lane, branch, idea and complete route without seeing fit scores."""
    depth = depth or {}
    eligible = {}
    for idea in ideas:
        routes = [route for route in idea.routes or ((),) if all(parent in references for parent in route)]
        if routes:
            eligible[idea.node] = (idea, routes)
    if not eligible:
        return None
    lane, lane_p = _pick(
        {lane: weight for lane, weight in LANES.items() if any(i.lane == lane for i, _ in eligible.values())},
        rng,
    )
    pool = [i for i, _ in eligible.values() if i.lane == lane]
    if lane == "root":
        pool = [i for i in pool if not attempts.get(i.node)] or pool
    branch, branch_p = _pick({i.branch: 1 / (1 + depth.get(i.branch, 0)) for i in pool}, rng)
    tried = Counter(i.mechanism for i in ideas if attempts.get(i.node))
    node, node_p = _pick(
        {
            i.node: 1
            / ((1 + attempts.get(i.node, 0)) * (1 + tried[i.mechanism] - bool(attempts.get(i.node))))
            for i in pool
            if i.branch == branch
        },
        rng,
    )
    routes = eligible[node][1]
    parents = routes[int(rng.integers(len(routes)))]
    return {
        "lane": lane,
        "branch": branch,
        "node": node,
        "parents": list(parents),
        "lane_probabilities": lane_p,
        "branch_probabilities": branch_p,
        "node_probabilities": node_p,
        "route_probability": 1 / len(routes),
        "eligible": [
            {"node": i.node, "parents": list(route)} for i, routes in eligible.values() for route in routes
        ],
    }


def _mode(idea, parents):
    if parents:
        return "combination"
    return {"root": "independent_root", "detour": "anomaly_detour", "frontier": "single_idea"}[idea.lane]


def _prompt(idea, parents, definitions, references, k):
    if parents:
        lines = [f"- Idea 1 — Base equation: {BASE}"]
        for parent in parents:
            lines.append(
                f"- Idea {len(lines) + 1} — {parent}: {definitions[parent].text}"
                f" Equation: {references[parent].text}"
            )
        lines.append(f"- Idea {len(lines) + 1} — {idea.text}")
        brief = (
            "Combine the ideas in this thread:\n"
            + "\n".join(lines)
            + f"\nGive {k} structurally distinct equations that combine these ideas"
            " and could fit the data shown in the attached images."
        )
    elif idea.lane == "root":
        brief = (
            f"New standalone idea: {idea.text}\nGive {k} structurally distinct full equations based on"
            " this idea that could fit the attached data. The plotted fit is a reference only."
        )
    elif idea.lane == "detour":
        brief = (
            f"Idea to try: {idea.text}\nGive {k} structurally distinct full equations that express this"
            " idea and could fit the attached data. You may change equation family."
        )
    else:
        brief = (
            f"Base equation: {BASE}\nIdea to test: {idea.text}\nGive {k} structurally distinct full"
            " equations that test this idea, keeping other terms unchanged."
        )
    return (
        brief + "\nDo not add an unrelated mechanism. Respect the stated fixed parts. "
        "Use theta in radians, periodic with period 2*pi. Use + - * / ** and "
        "sin, cos, exp, tanh, abs, sqrt or wrap. Return only a JSON array of objects with "
        "'equation' (expression), 'initial' (parameter names mapped to numeric starting values), "
        "and 'parameters' (free parameter definitions). If a simpler equation is an exact special case,"
        " optionally include 'nesting': an object keyed by the parent idea ID (or 'double_gaussian' for the base)."
        " Each value maps every new equation parameter to a parent parameter name, a numeric constant, or"
        " [parent parameter name, scale, shift] meaning scale * parent + shift. Do not fit or rank."
    )


def _baseline(up):
    seeds = up["seed_programs"]
    return ParentFit(BASELINE_ID, seeds.neuron_model_2_jax, seeds.parameter_estimator_2)


def _trials(x):
    order = jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(x.shape[1]))
    half = x.shape[1] // 2
    return order[:half], order[half:]


def _predictions(model, x, params):
    return jax.vmap(model)(x, *params.T)


BASE_NAMES = ("m", "b", "a1", "a2", "s")
BASE_LOWER = (0, 0, 0, 0, 0.01)
BASE_UPPER = (2 * np.pi, np.inf, np.inf, np.inf, np.inf)
BASE_PROBES = ((0.73, 0.24, 2.1, 0.4, 0.38), (3.8, 0.36, 0.8, 1.7, 0.91), (5.1, 1.4, 1.2, 2.4, 0.22))


@cache
def _baseline_templates():
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


def _verified_mapping(equation, parent, parent_names, probes, bindings):
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
    expected = _predictions(parent.model, grid, jnp.asarray(probes, dtype=grid.dtype))
    actual = _predictions(equation.model, grid, embed(jnp.asarray(probes, dtype=grid.dtype)))
    if np.isfinite(expected).all() and np.allclose(actual, expected, rtol=2e-5, atol=3e-6):
        return embed
    return None


def _offset_embedding(equation, baseline_model):
    if len(equation.initial) != 6:
        raise ValueError("Expected five baseline parameters and one free offset.")
    probe = jnp.linspace(0, 2 * np.pi, 32, endpoint=False)
    seeds = np.array([[0.73, 0.24, 2.1, 0.4, 0.38], [3.8, 0.36, 0.8, 1.7, 0.91]], dtype=np.float32)
    target = _predictions(baseline_model, jnp.broadcast_to(probe, (len(seeds), probe.size)), seeds)
    orders = np.array(list(permutations(range(6))))
    for neutral in (0.0, np.pi, 1.0, -np.pi, -1.0, *equation.initial.values()):
        params = np.column_stack((seeds, np.full(len(seeds), neutral, dtype=seeds.dtype)))[:, orders]
        x = jnp.broadcast_to(probe, (params.shape[0] * params.shape[1], probe.size))
        fit = _predictions(equation.model, x, params.reshape(-1, 6)).reshape(*params.shape[:2], probe.size)
        matches = np.asarray(jnp.all(jnp.isclose(fit, target[:, None, :], rtol=2e-6, atol=2e-6), axis=(0, 2)))
        if matches.any():
            return orders[np.flatnonzero(matches)[0]], neutral
    raise ValueError("Could not reproduce the double Gaussian at a neutral offset.")


def _fit_from_starts(equation, cohort, max_iter, starts, *, bounds=None):
    x, y = cohort
    train, held = _trials(x)
    xt, yt = x[:, train], y[:, train]

    def errors(params):
        return jnp.mean((_predictions(equation.model, xt, params) - yt) ** 2, axis=1)

    def project(params):
        return jnp.clip(params, *bounds) if bounds is not None else params

    def bounded(theta, *params):
        return equation.model(theta, *project(jnp.stack(params)))

    params = None
    for start, inherited in starts:
        if not np.allclose(_predictions(equation.model, xt, start), inherited, rtol=3e-4, atol=3e-5):
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
    mse = float(jnp.mean((_predictions(equation.model, x[:, held], params) - y[:, held]) ** 2))
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
    train, _ = _trials(x)
    return embed(params), _predictions(parent.model, x[:, train], params)


def _fit_nested(up, equation, cohort, max_iter, *, mappings):
    starts = (_parent_start(up, parent, embed, cohort, max_iter) for parent, embed in mappings)
    return _fit_from_starts(equation, cohort, max_iter, starts)


def _fit_offset(up, equation, cohort, max_iter, *, embedding=None):
    baseline = _baseline(up)
    order, neutral = _offset_embedding(equation, baseline.model) if embedding is None else embedding
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


def _measure(up, model, cohort, max_iter, *, nested_offset=False, solve=None, initialization="estimator"):
    x, y = cohort
    if nested_offset:
        solve = partial(_fit_offset, up, model)
        initialization = "fitted_double_gaussian"
    if solve is not None:
        score, mse, count, params = solve(cohort, max_iter)
    else:
        score, mse, count, params = score_program(up, model.model, model.estimator, x, y, max_iter)
    if not math.isfinite(score):
        raise ValueError("Nonfinite fit.")
    train, held = _trials(x)
    angles = x[:, held]
    fitted = _predictions(model.model, angles, params)
    errors = np.asarray((fitted - y[:, held]) ** 2)
    training_mse = float(jnp.mean((_predictions(model.model, x[:, train], params) - y[:, train]) ** 2))
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


def _write(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _decode(reply, k):
    if not reply:
        raise ValueError("No proposal text returned.")
    text = reply.strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        text = text.split("\n", 1)[1][:-3]

    def reject_number(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    values = json.loads(text, parse_constant=reject_number, parse_float=lambda value: _finite_float(value))
    if not isinstance(values, list) or len(values) != k:
        raise ValueError(f"Expected exactly {k} proposals.")
    return values


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number.")
    return number


async def execute(
    up,
    proposer,
    explore,
    confirm,
    output,
    *,
    ideas=DEFAULT_IDEAS,
    batches=6,
    k=4,
    seed=0,
    max_iter=2000,
    final=None,
    waves=1,
    resume=False,
):
    """Explore rolling waves, then confirm a frozen shortlist; optionally select and test once."""
    validate_graph(ideas)
    if min(batches, k, waves) < 1:
        raise ValueError("Wave count, batch count and equations per batch must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "wave.json"
    if path.exists() and not resume:
        raise FileExistsError(path)
    nodes = {i.node: i for i in ideas}
    references, metrics, attempts, depth, seen, exact = {}, {}, Counter(), Counter(), set(), {BASE}
    lineage = {}
    rng = np.random.default_rng(seed)
    born = set()
    wave: dict[str, Any] = {
        "version": 3,
        "seed": seed,
        "fit_protocol": "nested_parent_v1",
        "idea_catalog": [asdict(i) for i in ideas],
        "graph": [baseline_node()],
        "planned_batches": batches * waves,
        "episodes": [],
        "confirmation": [],
        "status": "exploring",
    }
    config = {"batches": batches, "waves": waves, "k": k, "max_iter": max_iter, "finalize": final is not None}
    if resume:
        saved = json.loads(path.read_text())
        fields = ("version", "seed", "idea_catalog", "planned_batches", "fit_protocol")
        if any(saved.get(key) != json.loads(json.dumps(wave[key])) for key in fields):
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if saved.get("config", config) != config:
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if saved["status"] in {"complete", "no_fitted_equations"}:
            return saved
        if "selection" in saved:
            raise ValueError("Cannot resume after selection has started; locked tests are one-shot.")
        for event in saved["episodes"]:
            choice = draw(ideas, references, attempts, rng, depth)
            node = event["node"]
            if not choice or (choice["node"], choice["parents"]) != (node, event["parents"]):
                raise ValueError("Checkpoint does not match the scheduler replay.")
            attempts[node] += 1
            for record in event["candidates"]:
                if record["status"] not in {"fitted", "fit_failed"}:
                    continue
                candidate = compile_equation(record["proposal"])
                seen.add(candidate.key)
                exact.add(candidate.text.strip())
                born.add(node)
                if record["status"] == "fitted":
                    depth[nodes[node].branch] += 1
                    if node not in references:
                        references[node], metrics[node] = candidate, record["metrics"]
                        lineage[node] = event["parents"]
        wave = saved
        wave.pop("active_probe", None)
        wave["confirmation"] = []
        wave["status"] = "exploring"
    wave["config"] = config
    _write(path, wave)
    if "baseline_exploration" not in wave:
        wave["baseline_exploration"] = _measure(up, _baseline(up), explore, max_iter)

    baseline_fit = _baseline(up)

    @cache
    def reference_fit(node):
        candidate = references[node]
        solve, _ = fit_strategy(node, candidate, tuple(lineage[node]))
        return ParentFit(node, candidate.model, candidate.estimator, solve)

    def fit_strategy(node, equation, parents):
        if nodes[node].mechanism == "offset" and not nodes[node].routes:
            start = _offset_embedding(equation, baseline_fit.model)
            return partial(_fit_offset, up, equation, embedding=start), "fitted_double_gaussian"
        mappings = []
        supplied = equation.nesting if isinstance(equation.nesting, dict) else {}
        template = (
            _baseline_templates().get(equation.key) if len(equation.initial) > len(BASE_NAMES) else None
        )
        if template is not None or BASELINE_ID in supplied:
            bindings = (
                dict(zip(equation.canonical_names, template, strict=True))
                if template is not None
                else supplied[BASELINE_ID]
            )
            embed = _verified_mapping(equation, baseline_fit, BASE_NAMES, BASE_PROBES, bindings)
            if embed is not None:
                mappings.append((baseline_fit, embed))
        for name in parents:
            parent = references[name]
            if name not in supplied or len(equation.initial) <= len(parent.initial):
                continue
            initial = np.array(tuple(parent.initial.values()))
            perturb = np.random.default_rng(0).uniform(-0.3, 0.3, size=(4, len(initial)))
            probes = np.concatenate((initial[None, :], initial + perturb * (0.2 + np.abs(initial))))
            fit = reference_fit(name)
            embed = _verified_mapping(equation, fit, tuple(parent.initial), probes, supplied[name])
            if embed is not None:
                mappings.append((fit, embed))
        if not mappings:
            return None, "estimator"
        source = (
            "fitted_parent"
            if any(parent.node_id != BASELINE_ID for parent, _ in mappings)
            else "fitted_double_gaussian"
        )
        return partial(_fit_nested, up, equation, mappings=tuple(mappings)), source

    def measure_node(node, equation, cohort, *, parents=None):
        solve, source = fit_strategy(node, equation, tuple(lineage[node] if parents is None else parents))
        return _measure(up, equation, cohort, max_iter, solve=solve, initialization=source)

    def evaluate(proposal, event, step, duplicates, *, retry=False):
        node, parents = event["node"], event["parents"]
        idea = nodes[node]
        record = {"proposal": proposal, **({"source": "retry"} if retry else {})}
        event["candidates"].append(record)
        try:
            candidate = compile_equation(proposal)
        except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
            record.update(status="invalid", error=str(exc))
            return
        record["equation"] = candidate.text
        text = candidate.text.strip()
        if text in exact or candidate.key in seen:
            record["status"] = "duplicate"
            if text in exact and not retry:
                duplicates.append(text)
            return
        seen.add(candidate.key)
        exact.add(text)
        if node not in born:
            born.add(node)
            routes = idea.routes or (() if idea.lane == "root" else ((BASELINE_ID,),))
            wave["graph"].append({**asdict(idea), "routes": routes, "introduced_at": step + 1})
        try:
            result = measure_node(node, candidate, explore, parents=parents)
            record.update(status="fitted", metrics=result)
            depth[idea.branch] += 1
            if node not in references:
                references[node], metrics[node] = candidate, result
                lineage[node] = parents
        except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
            record.update(status="fit_failed", error=str(exc))

    for step in range(len(wave["episodes"]), batches * waves):
        choice = draw(ideas, references, attempts, rng, depth)
        if choice is None:
            break
        node, parents = choice["node"], choice["parents"]
        idea = nodes[node]
        attempts[node] += 1
        fitted = tuple(reference_fit(p) for p in parents)
        prompt = _prompt(idea, parents, nodes, references, k)
        wave_number = step // batches
        event = {
            **choice,
            "mode": _mode(idea, parents),
            "wave": wave_number + 1,
            "image_seed": seed + wave_number,
            "prompt": prompt,
            "candidates": [],
        }
        wave["active_probe"] = {
            "batch": step + 1,
            "node": node,
            "label": idea.label or idea.text,
            "lane": idea.lane,
            "mode": event["mode"],
        }
        _write(path, wave)
        batch = None
        try:
            batch = prepare_images(up, *explore, seed=event["image_seed"], parents=fitted, max_iter=max_iter)
            event.update(cells=list(batch.cells), images=len(batch.images))
            event["reply"] = await proposer.submit(batch, prompt)
            proposals = _decode(event["reply"], k)
        except ValueError as exc:
            event["error"] = str(exc)
            proposals = []
        duplicates = []

        for proposal in proposals:
            evaluate(proposal, event, step, duplicates)
        if duplicates and batch is not None:
            instruction = (
                "These equations have already been tested (one entry per repeat): "
                f"{json.dumps(duplicates)}. Suggest one different equation for each repeat, "
                f"keeping the same idea. Return only a JSON array of exactly {len(duplicates)} replacements "
                "in the original format."
            )
            retry = event["retry"] = {"instruction": instruction}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"], instruction)
                replacements = _decode(retry["reply"], len(duplicates))
            except ValueError as exc:
                retry["error"] = str(exc)
            else:
                for replacement in replacements:
                    evaluate(replacement, event, step, duplicates, retry=True)
        log.info("Frontier episode %s: %s (%s images)", step + 1, node, event.get("images"))
        del wave["active_probe"]
        wave["episodes"].append(event)
        _write(path, wave)
    wave["shortlist"] = {
        node: {
            "equation": eq.text,
            "initial": eq.initial,
            "exploration": metrics[node],
            "parents": lineage[node],
        }
        for node, eq in references.items()
    }
    wave["confirmation_rule"] = (
        "Lower whole-curve MSE versus the double Gaussian and every used parent; "
        "no 30-degree bin may regress over 5% of the comparator's MSE."
    )
    wave["status"] = "confirming"
    _write(path, wave)
    if not references:
        wave["status"] = "no_fitted_equations"
        _write(path, wave)
        return json.loads(path.read_text())
    baseline = _measure(up, _baseline(up), confirm, max_iter)
    wave["confirmation_baseline"] = baseline
    measured = {}
    for node, eq in references.items():
        row = {"node": node}
        try:
            result = measure_node(node, eq, confirm)
            measured[node] = result
            comparators = {"double_gaussian": baseline, **{p: measured[p] for p in lineage[node]}}
            improved = all(
                result["mse"] < ref["mse"]
                and all(
                    a - b <= 0.05 * ref["mse"]
                    for a, b in zip(result["angle_mse"], ref["angle_mse"], strict=True)
                    if a is not None and b is not None
                )
                for ref in comparators.values()
            )
            row.update(metrics=result, outcome="confirmed_fit" if improved else "unresolved")
        except (KeyError, ValueError) as exc:
            row.update(outcome="inconclusive", error=str(exc))
        wave["confirmation"].append(row)
    supported = {row["node"] for row in wave["confirmation"] if row["outcome"] == "confirmed_fit"}

    def supported_claim(node):
        return node in supported and all(supported_claim(parent) for parent in lineage[node])

    wave["claims"] = {
        row["node"]: "supported_fit" if supported_claim(row["node"]) else "unresolved"
        for row in wave["graph"][1:]
    }
    if final is not None and supported:
        wave["selection"] = {"status": "started"}
        _write(path, wave)
        selection, locked = final
        scores = {"double_gaussian": _measure(up, _baseline(up), selection, max_iter)}
        for node in sorted(n for n in supported if supported_claim(n)):
            try:
                scores[node] = measure_node(node, references[node], selection)
            except ValueError:
                continue
        winner = min(scores, key=lambda name: (scores[name]["score"], name))
        wave["selection"] = {"metrics": scores, "winner": winner}
        _write(path, wave)
        locked_baseline = _measure(up, _baseline(up), locked, max_iter)
        wave["locked"] = {"double_gaussian": locked_baseline}
        if winner != "double_gaussian":
            wave["locked"][winner] = measure_node(winner, references[winner], locked)
    wave["status"] = "complete"
    _write(path, wave)
    return json.loads(path.read_text())


class _Live:
    def __init__(self, client):
        self.client = client

    async def submit(self, batch, prompt):
        return await batch.submit(self.client, prompt, effort="none", temperature=0.5)

    async def retry(self, batch, prompt, previous_reply, instruction):
        return await batch.submit(
            self.client, prompt, effort="none", temperature=0.5, followup=(previous_reply, instruction)
        )


def run(args):
    """Load the pinned recording and run one explicitly bounded, free-model wave."""
    output = Path(args.output)
    if args.resume:
        if not (output / "manifest.json").is_file() or not (output / "wave.json").is_file():
            raise SystemExit("Resume requires an existing run manifest and checkpoint.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("This run is already active.") from exc
        _run(args, output)


def _run(args, output):
    if min(args.waves, args.batches, args.k) < 1 or args.seed < 0:
        raise SystemExit("Use positive waves, batches and K and a nonnegative seed.")
    ideas = DEFAULT_IDEAS
    if args.graph is not None:
        ideas = tuple(
            Idea(
                node=row["node"],
                branch=row["branch"],
                mechanism=row["mechanism"],
                text=row["text"],
                routes=tuple(tuple(route) for route in row.get("routes", ())),
                lane=row.get("lane", "frontier"),
                label=row.get("label", ""),
            )
            for row in json.loads(args.graph.read_text())
        )
    validate_graph(ideas)
    entry = fetch_models(load_key()).get(MODEL)
    if not entry or not is_free(entry) or "image/png" not in entry["modalities"]["input"]:
        raise SystemExit(f"{MODEL} is not confirmed free with PNG support.")
    up = import_upstream(args.upstream.resolve())
    fitting.install(up["hypothesis_engine"], "lm-bounded")
    response, angles, sha = load_dataset(args.dataset, up["utils"].extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    discovery = np.random.default_rng(881).permutation(groups["discovery"])
    cut = len(discovery) * 3 // 4
    explore, confirm = discovery[:cut], discovery[cut:]
    manifest = {
        "dataset_sha256": sha,
        "model": MODEL,
        "seed": args.seed,
        "discovery_partition_seed": 881,
        "exploration_cells": indices["eligible_cell_ids"][explore].tolist(),
        "confirmation_cells": indices["eligible_cell_ids"][confirm].tolist(),
        "batches": args.batches,
        "k": args.k,
        "waves": args.waves,
        "finalize": args.finalize,
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("Resume requires the same dataset, split, and run options.")
    else:
        _write(manifest_path, manifest)

    def cohort(rows):
        return x[rows], y[rows]

    async def wave():
        go = GoClient(output, Ledger(output / "spend.jsonl"), output.name, [MODEL], max_tokens=8192)
        try:
            final = (cohort(groups["selection"]), cohort(groups["locked_test"])) if args.finalize else None
            return await execute(
                up,
                _Live(go),
                cohort(explore),
                cohort(confirm),
                output,
                batches=args.batches,
                k=args.k,
                seed=args.seed,
                final=final,
                waves=args.waves,
                ideas=ideas,
                resume=args.resume,
            )
        finally:
            await go.close()

    result = asyncio.run(wave())
    print(f"{result['status']}: {output / 'wave.json'}")
    if result["status"] != "complete":
        raise SystemExit(1)

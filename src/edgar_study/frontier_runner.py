"""Run score-blind equation exploration and one independent confirmation pass."""

import ast
import asyncio
import fcntl
import json
import logging
import math
import operator
import re
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


def _soft_cutoff(left, right):
    if jnp.iscomplexobj(left) or jnp.iscomplexobj(right):
        raise InvalidEquation("nonreal_expression", "Cutoffs require real-valued inputs.")
    return jax.nn.sigmoid(20 * (right - left))


def _soft_cutoff_reverse(left, right):
    return _soft_cutoff(right, left)


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Lt: _soft_cutoff,
    ast.LtE: _soft_cutoff,
    ast.Gt: _soft_cutoff_reverse,
    ast.GtE: _soft_cutoff_reverse,
}
CUTOFFS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
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


@dataclass(frozen=True)
class Equation:
    text: str
    initial: dict[str, float]
    key: str
    model: object
    estimator: object
    canonical_names: tuple[str, ...]
    nesting: object


class InvalidEquation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def compile_equation(proposal):
    """Compile a small numeric expression tree without executing generated Python."""
    if not isinstance(proposal, dict):
        raise InvalidEquation("invalid_equation_object", "Expected an equation object.")
    text, initial = proposal.get("equation"), proposal.get("initial")
    if not isinstance(text, str) or not isinstance(initial, dict):
        raise InvalidEquation(
            "invalid_equation_object", "Expected an equation string and a parameter object."
        )
    if len(text) > 600:
        raise InvalidEquation("equation_too_complex", "Equation exceeds 600 characters.")
    if not 1 <= len(initial) <= 12:
        raise InvalidEquation("invalid_parameter_count", "Expected one to twelve free parameters.")
    for name, value in initial.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or not name.isascii()
            or name in {"pi", "theta", *FUNCTIONS}
            or type(value) not in (float, int)
            or not -100 <= value <= 100
        ):
            raise InvalidEquation("invalid_parameter", "Invalid parameter name or initial value.")
    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise InvalidEquation("invalid_syntax", "Equation is not a valid expression.") from exc
    except RecursionError as exc:
        raise InvalidEquation("equation_too_complex", "Equation is nested too deeply.") from exc
    if sum(1 for _ in ast.walk(expr)) > 260:
        raise InvalidEquation("equation_too_complex", "Equation exceeds the expression complexity limit.")
    used = set()

    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int) and -100 <= node.value <= 100:
            return ("number", float(node.value))
        if isinstance(node, ast.Name):
            if node.id not in {"pi", "theta", *initial}:
                raise InvalidEquation(
                    "unknown_name",
                    f"Undefined name {node.id!r}; declare it in initial or use theta, pi and the declared parameters.",
                )
            used.add(node.id)
            return ("name", node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return (type(node.op), build(node.left), build(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CUTOFFS:
            return (type(node.ops[0]), build(node.left), build(node.comparators[0]))
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
        raise InvalidEquation("unsupported_operator", "Expression uses an unsupported operation or name.")

    tree = build(expr)
    if set(initial) != used - {"pi", "theta"} or "theta" not in used:
        raise InvalidEquation(
            "parameter_mismatch", "Equation must use theta and exactly the listed parameters."
        )

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
        try:
            a, b = np.asarray(model(grid, *params)), np.asarray(model(grid + 2 * jnp.pi, *params))
        except InvalidEquation:
            raise
        except ArithmeticError as exc:
            raise InvalidEquation("nonfinite_expression", "Equation has undefined arithmetic.") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidEquation(
                "invalid_numeric_expression", "Equation has an invalid numeric operation."
            ) from exc
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            raise InvalidEquation("nonreal_expression", "Equation must produce real-valued responses.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise InvalidEquation(
                "nonfinite_expression", "Equation produces nonfinite values at trial parameters."
            )
        if not np.allclose(a, b, atol=2e-4, rtol=2e-4):
            raise InvalidEquation("nonperiodic_expression", "Equation failed the 2*pi periodicity probe.")
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


def draw_generated(nodes, references, history, rng, new_probability, step):
    """Draw a new route or revisit using proposal validity, never fit quality."""
    types = {"new": new_probability, "revisit": 1 - new_probability} if nodes else {"new": 1}
    kind, kind_p = _pick({key: value for key, value in types.items() if value}, rng)
    choice: dict[str, Any] = {"kind": kind, "kind_probabilities": kind_p}
    if kind == "revisit":
        weights = {}
        for node in nodes:
            recent = history.get(node, [])[-16:]
            weights[node] = 1 - 0.85 * sum(recent) / max(1, len(recent))
        node, node_p = _pick(weights, rng)
        routes = [
            route
            for route in nodes[node].routes or ((),)
            if all(parent == BASELINE_ID or parent in references for parent in route)
        ]
        parents = routes[int(rng.integers(len(routes)))]
        return {
            **choice,
            "node": node,
            "parents": list(parents),
            "mode": "revisit",
            "lane": nodes[node].lane,
            "node_probabilities": node_p,
            "route_probability": 1 / len(routes),
        }
    pool = [BASELINE_ID, *references]
    weights = {"single_idea": 0.6, "anomaly_detour": 0.2, "independent_root": 0.1}
    if len(references) >= 2:
        weights["combination"] = 0.1
    mode, mode_p = _pick(weights, rng)
    if mode == "combination":
        pool = list(references)
    count = {"independent_root": 0, "combination": 2}.get(mode, 1)
    parents = sorted(rng.choice(pool, count, replace=False).tolist())
    return {
        **choice,
        "node": f"idea-{step + 1}",
        "mode": mode,
        "parents": parents,
        "lane": {"anomaly_detour": "detour", "independent_root": "root"}.get(mode, "frontier"),
        "mode_probabilities": mode_p,
        "eligible_parents": pool,
        "route_probability": 1 / math.comb(len(pool), count),
    }


def _idea_key(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def _generated_idea(value, choice, existing):
    if not isinstance(value, dict):
        raise InvalidReply("invalid_idea", "Return an idea object with a title, mechanism, and change.")
    parts = {}
    for field, limit in (("title", 64), ("change", 600)):
        text = value.get(field)
        if not isinstance(text, str) or not _idea_key(text) or len(text.strip()) > limit:
            raise InvalidReply(
                "invalid_idea", f"Idea {field} must be nonempty and at most {limit} characters."
            )
        parts[field] = " ".join(text.split())
    mechanism_text = value.get("mechanism") or parts["title"]
    if (
        not isinstance(mechanism_text, str)
        or not _idea_key(mechanism_text)
        or len(mechanism_text.strip()) > 64
    ):
        raise InvalidReply("invalid_idea", "Idea mechanism must be nonempty and at most 64 characters.")
    parents = tuple(choice["parents"])
    routes = ((*parents,),) if parents else ()
    mechanism = _idea_key(mechanism_text).replace(" ", "-")
    same_route = [idea for idea in existing.values() if idea.routes == routes]
    for idea in same_route:
        if _idea_key(parts["title"]) == _idea_key(idea.label) or _idea_key(parts["change"]) == _idea_key(
            idea.text
        ):
            titles = ", ".join(sorted(idea.label for idea in same_route))
            raise InvalidReply(
                "duplicate_idea",
                f"This repeats the idea {idea.node} on the same route. Choose a materially different change."
                f" Existing ideas on this route: {titles}.",
            )
    return Idea(
        choice["node"], choice["mode"], mechanism, parts["change"], routes, choice["lane"], parts["title"]
    )


def _generated_prompt(choice, nodes, references, k, tried=()):
    selected = choice["node"]
    if choice["kind"] == "revisit":
        idea = nodes[selected]
        task = (
            f"Revisit {idea.label}: {idea.text}. Keep the same idea and give new implementations. "
            f"Return only a JSON array with zero to {k} equation objects."
        )
        if tried:
            task += f" Already tried for this idea (do not repeat): {json.dumps(list(tried))}."
    else:
        mode = choice["mode"]
        brief = {
            "single_idea": "Invent one precise change to the selected parent equation.",
            "anomaly_detour": "Invent one precise change addressing a pattern in the residual plots.",
            "combination": "Invent one idea explicitly combining the selected parents' changes.",
            "independent_root": "Invent one independent model family. The plotted baseline is only a reference.",
        }[mode]
        known = [
            {"title": node.label, "mechanism": node.mechanism, "change": node.text}
            for node in nodes.values()
            if node.routes == ((tuple(choice["parents"])),) or (not node.routes and not choice["parents"])
        ]
        task = (
            f"{brief} Existing ideas on this route (avoid repeats): {json.dumps(known)}. "
            "Return only a JSON object with 'idea' and 'equations'. "
            "The idea has a short, concise 'title' you choose and a concrete 'change'. "
            "The optional 'mechanism' names the changed component, for example offset, width, amplitude, "
            "centre, baseline, shape, kernel, or mixture; it defaults to the title. "
            f"Provide zero to {k} structurally distinct equations in 'equations'."
        )
    parents = [
        f"{name}: {BASE}"
        if name == BASELINE_ID
        else f"{name}: {nodes[name].label} — {nodes[name].text}. Equation: {references[name].text}"
        for name in choice["parents"]
    ]
    return (
        f"{task}\nSelected parents: {json.dumps(parents)}.\n"
        "Use theta in radians, periodic with period 2*pi. Use + - * / **, "
        "sin, cos, exp, tanh, abs, sqrt, wrap, or a smooth cutoff using <, <=, >, or >=. "
        "Declare every symbol you use except theta and pi in 'initial'. "
        "Each equation object has 'equation', 'initial' (parameter names to numeric starts), and "
        "'parameters' (definitions). Optionally include 'nesting': parent ID to a map from each new "
        "parameter to a parent parameter name, a number, or [parent parameter name, scale, shift]. "
        "For double_gaussian use parent parameter names m, b, a1, a2, s. Do not fit or rank."
    )


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
        "sin, cos, exp, tanh, abs, sqrt or wrap. Comparisons with <, <=, >, or >= give smooth cutoffs."
        " Declare every symbol except theta and pi in initial."
        " Keep expressions short. Return only a JSON array of objects with "
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


def _offset_bounds(embed):
    options = embed.keywords
    index, scale, shift = (options[key] for key in ("indices", "scales", "shifts"))
    lo, hi = np.array(BASE_LOWER)[index], np.array(BASE_UPPER)[index]
    mapped = scale != 0
    factor = np.where(mapped, scale, 1)
    low = np.where(mapped, np.where(scale < 0, hi, lo) * factor + shift, -np.inf)
    high = np.where(mapped, np.where(scale < 0, lo, hi) * factor + shift, np.inf)
    return jnp.asarray(low), jnp.asarray(high)


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


def _fit_nested(up, equation, cohort, max_iter, *, mappings, bounds=None):
    starts = (_parent_start(up, parent, embed, cohort, max_iter) for parent, embed in mappings)
    return _fit_from_starts(equation, cohort, max_iter, starts, bounds=bounds)


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


class InvalidReply(InvalidEquation):
    pass


def _decode_json(reply):
    if not isinstance(reply, str) or not reply.strip():
        raise InvalidReply("empty_reply", "No proposal text returned.")
    text = reply.strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        text = text.split("\n", 1)[1][:-3]

    def reject_number(value):
        raise InvalidReply("nonfinite_reply_number", "Reply contains a nonfinite JSON number.")

    try:
        return json.loads(text, parse_constant=reject_number, parse_float=_finite_float)
    except InvalidReply:
        raise
    except (ValueError, RecursionError) as exc:
        raise InvalidReply("invalid_reply_json", "Reply is not readable JSON.") from exc


def _equation_list(values, k, *, exact=True):
    if not isinstance(values, list) or len(values) > k or (exact and len(values) != k):
        count = f"exactly {k}" if exact else f"at most {k}"
        raise InvalidReply("invalid_reply_count", f"Expected {count} proposals.")
    return values


def _decode(reply, k):
    return _equation_list(_decode_json(reply), k)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise InvalidReply("nonfinite_reply_number", "Reply contains a nonfinite JSON number.")
    return number


async def execute(
    up,
    proposer,
    explore,
    confirm,
    output,
    *,
    ideas=None,
    batches=6,
    k=4,
    seed=0,
    max_iter=2000,
    final=None,
    waves=1,
    resume=False,
    new_probability=0.8,
):
    """Explore rolling waves, then confirm a frozen shortlist; optionally select and test once."""
    discovery = ideas is None
    ideas = () if discovery else ideas
    validate_graph(ideas)
    if not 0 <= new_probability <= 1:
        raise ValueError("New-idea probability must be between zero and one.")
    if min(batches, k, waves) < 1:
        raise ValueError("Wave count, batch count and equations per batch must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "wave.json"
    if path.exists() and not resume:
        raise FileExistsError(path)
    nodes = {i.node: i for i in ideas}
    references, metrics, attempts, depth, seen, exact = {}, {}, Counter(), Counter(), set(), {BASE}
    lineage = {}
    history: dict[str, list[bool]] = {}
    tried: dict[str, list[str]] = {}
    rng = np.random.default_rng(seed)
    born = set()
    wave: dict[str, Any] = {
        "version": 3,
        "seed": seed,
        "fit_protocol": "nested_parent_v3",
        "idea_source": "generated" if discovery else "catalog",
        "idea_catalog": [asdict(i) for i in ideas],
        "graph": [baseline_node()],
        "planned_batches": batches * waves,
        "episodes": [],
        "confirmation": [],
        "status": "exploring",
    }
    config = {"batches": batches, "waves": waves, "k": k, "max_iter": max_iter, "finalize": final is not None}
    if discovery:
        config["new_probability"] = new_probability

    def next_choice(step):
        if discovery:
            return draw_generated(nodes, references, history, rng, new_probability, step)
        return draw(ideas, references, attempts, rng, depth)

    def track_proposals(event):
        if not discovery:
            return
        if event["node"] in nodes:
            history.setdefault(event["node"], []).extend(
                row["status"] in {"duplicate", "invalid"} for row in event["candidates"]
            )
        equations = tried.setdefault(event["node"], [])
        for record in event["candidates"]:
            equation = record.get("equation")
            if equation and equation not in equations:
                equations.append(equation)
        del equations[:-8]

    if resume:
        saved = json.loads(path.read_text())
        fields = ("version", "seed", "idea_source", "planned_batches", "fit_protocol")
        if any(saved.get(key) != json.loads(json.dumps(wave[key])) for key in fields):
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if saved.get("config", config) != config:
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if not discovery and saved["idea_catalog"] != json.loads(json.dumps(wave["idea_catalog"])):
            raise ValueError("Run checkpoint does not match the idea catalog.")
        if saved["status"] in {"complete", "no_fitted_equations"}:
            return saved
        if "selection" in saved:
            raise ValueError("Cannot resume after selection has started; locked tests are one-shot.")
        for step, event in enumerate(saved["episodes"]):
            choice = next_choice(step)
            node = event["node"]
            if not choice or any(event.get(key) != choice[key] for key in choice):
                raise ValueError("Checkpoint does not match the scheduler replay.")
            if discovery and "idea" in event:
                value = event["idea"]
                nodes[node] = Idea(
                    node,
                    value["branch"],
                    value["mechanism"],
                    value["text"],
                    tuple(map(tuple, value["routes"])),
                    value["lane"],
                    value["label"],
                )
                born.add(node)
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
            track_proposals(event)
        if discovery:
            replayed = {event["node"] for event in saved["episodes"]}
            saved["graph"] = [
                row for row in saved["graph"] if row["node"] == BASELINE_ID or row["node"] in replayed
            ]
            saved["idea_catalog"] = [row for row in saved["idea_catalog"] if row["node"] in replayed]
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
        if node == BASELINE_ID:
            return baseline_fit
        candidate = references[node]
        solve, _ = fit_strategy(node, candidate, tuple(lineage[node]))
        return ParentFit(node, candidate.model, candidate.estimator, solve)

    def fit_strategy(node, equation, parents):
        mappings = []
        bounds = None
        standalone_offset = tuple(parents) in {(), (BASELINE_ID,)}
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
                if standalone_offset:
                    bounds = _offset_bounds(embed)
        for name in parents:
            if name == BASELINE_ID:
                continue
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
            if standalone_offset and len(equation.initial) == 6:
                try:
                    start = _offset_embedding(equation, baseline_fit.model)
                except ValueError:
                    return None, "estimator"
                return partial(_fit_offset, up, equation, embedding=start), "fitted_double_gaussian"
            return None, "estimator"
        source = (
            "fitted_parent"
            if any(parent.node_id != BASELINE_ID for parent, _ in mappings)
            else "fitted_double_gaussian"
        )
        return partial(_fit_nested, up, equation, mappings=tuple(mappings), bounds=bounds), source

    def measure_node(node, equation, cohort, *, parents=None):
        solve, source = fit_strategy(node, equation, tuple(lineage[node] if parents is None else parents))
        return _measure(up, equation, cohort, max_iter, solve=solve, initialization=source)

    def evaluate(proposal, event, step, *, replaces=None, replacement=False):
        node, parents = event["node"], event["parents"]
        idea = nodes[node]
        record = {"proposal": proposal}
        if replacement:
            record["source"] = "retry"
            if replaces is not None:
                record["replaces"] = replaces
        event["candidates"].append(record)
        try:
            candidate = compile_equation(proposal)
        except InvalidEquation as exc:
            record.update(status="invalid", reason_code=exc.code, error=str(exc))
            return exc.code, str(exc)
        record["equation"] = candidate.text
        text = candidate.text.strip()
        if text in exact or candidate.key in seen:
            record["status"] = "duplicate"
            return ("exact_repeat", json.dumps(text)) if text in exact else None
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
        return None

    def decode_probe(reply, event, step):
        if not discovery:
            return _decode(reply, k)
        value = _decode_json(reply)
        if event["kind"] == "revisit":
            return _equation_list(value, k, exact=False)
        if not isinstance(value, dict):
            raise InvalidReply("invalid_idea_reply", "Return an object containing idea and equations.")
        equations = _equation_list(value.get("equations", []), k, exact=False)
        accepted = _generated_idea(value.get("idea"), event, nodes)
        nodes[accepted.node] = accepted
        born.add(accepted.node)
        record = asdict(accepted)
        event["idea"] = record
        wave["active_probe"].update(idea=record, label=accepted.label)
        _write(path, wave)
        wave["idea_catalog"].append(record)
        wave["graph"].append({**record, "introduced_at": step + 1})
        return equations

    for step in range(len(wave["episodes"]), batches * waves):
        choice = next_choice(step)
        if choice is None:
            break
        node, parents = choice["node"], choice["parents"]
        idea = nodes.get(node)
        attempts[node] += 1
        fitted = tuple(reference_fit(p) for p in parents)
        prompt = (
            _generated_prompt(choice, nodes, references, k, tried.get(node, ()))
            if discovery
            else _prompt(idea, parents, nodes, references, k)
        )
        wave_number = step // batches
        event = {
            **choice,
            "mode": choice["mode"] if discovery else _mode(idea, parents),
            "wave": wave_number + 1,
            "image_seed": seed + wave_number,
            "prompt": prompt,
            "candidates": [],
        }
        wave["active_probe"] = {
            "batch": step + 1,
            "node": node,
            "label": (idea.label or idea.text) if idea else "New idea",
            "lane": choice["lane"],
            "mode": event["mode"],
        }
        _write(path, wave)
        batch = None
        unreadable = None

        try:
            batch = prepare_images(up, *explore, seed=event["image_seed"], parents=fitted, max_iter=max_iter)
            event.update(cells=list(batch.cells), images=len(batch.images))
            event["reply"] = await proposer.submit(batch, prompt)
            proposals = decode_probe(event["reply"], event, step)
        except InvalidReply as exc:
            unreadable = exc
            event["reply_error"] = {"reason_code": exc.code, "error": str(exc)}
            proposals = []
        except ValueError as exc:
            event["error"] = str(exc)
            proposals = []
        targets = []
        for position, proposal in enumerate(proposals, start=1):
            rejected = evaluate(proposal, event, step)
            if rejected is not None:
                reason, detail = rejected
                targets.append({"position": position, "reason_code": reason, "detail": detail})
        retried = unreadable is not None
        if unreadable is not None and batch is not None:
            instruction = (
                f"{unreadable.code}: {unreadable} Return the entire reply again in the requested format."
            )
            retry = event["retry"] = {"instruction": instruction, "reason_code": unreadable.code}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"] or "", instruction)
                replacements = decode_probe(retry["reply"], event, step)
            except ValueError as exc:
                retry["error"] = str(exc)
                event["error"] = str(unreadable)
            else:
                for replacement in replacements:
                    evaluate(replacement, event, step, replacement=True)
        elif targets and batch is not None:
            retried = True
            reasons = "\n".join(
                f"- Proposal {target['position']}: {target['reason_code']} — {target['detail']}"
                for target in targets
            )
            instruction = (
                f"Replace these proposals from your original reply:\n{reasons}\n"
                "Suggest one different equation for each listed proposal, keeping the same idea. "
                f"Return only a JSON array of exactly {len(targets)} replacements in the listed order "
                "and original format."
            )
            retry = event["retry"] = {"instruction": instruction, "targets": targets}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"], instruction)
                replacements = _decode(retry["reply"], len(targets))
            except ValueError as exc:
                retry["error"] = str(exc)
            else:
                for replacement, target in zip(replacements, targets, strict=True):
                    evaluate(replacement, event, step, replaces=target["position"], replacement=True)
        fitted_here = any(
            record.get("status") == "fitted" for record in event["candidates"] if "status" in record
        )
        if batch is not None and not retried and not fitted_here:
            instruction = (
                "No equation from your last reply could be used. Return a fresh reply in the requested"
                " format with materially different equations. Do not repeat earlier equations."
            )
            retry = event["retry"] = {"instruction": instruction, "reason_code": "no_usable_equation"}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"] or "", instruction)
                if discovery and (event.get("idea") or event["kind"] == "revisit"):
                    replacements = _equation_list(_decode_json(retry["reply"]), k, exact=False)
                else:
                    replacements = _decode(retry["reply"], k)
            except ValueError as exc:
                retry["error"] = str(exc)
            else:
                for replacement in replacements:
                    evaluate(replacement, event, step, replacement=True)
        log.info("Frontier episode %s: %s (%s images)", step + 1, node, event.get("images"))
        del wave["active_probe"]
        track_proposals(event)
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
            comparators = {
                BASELINE_ID: baseline,
                **{p: measured[p] for p in lineage[node] if p != BASELINE_ID},
            }
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
        return node == BASELINE_ID or (
            node in supported and all(supported_claim(parent) for parent in lineage[node])
        )

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
    ideas = None
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
    if ideas is not None:
        validate_graph(ideas)
    if not 0 <= args.new_probability <= 1:
        raise SystemExit("--new-probability must be between zero and one.")
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
        "idea_source": "generated" if ideas is None else "catalog",
        "new_probability": args.new_probability if ideas is None else None,
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
                new_probability=args.new_probability,
                resume=args.resume,
            )
        finally:
            await go.close()

    result = asyncio.run(wave())
    print(f"{result['status']}: {output / 'wave.json'}")
    if result["status"] != "complete":
        raise SystemExit(1)

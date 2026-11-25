"""Run score-blind equation exploration and one independent confirmation pass."""

import ast
import asyncio
import json
import logging
import math
import operator
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import TRIAL_SEED, load_dataset, prepare
from edgar_study.frontier_images import ParentFit, prepare_images
from edgar_study.frontier_seed import BASE, BASELINE_ID, baseline_node
from edgar_study.go_client import GoClient, Ledger, fetch_models, is_free, load_key
from edgar_study.upstream import import_upstream, score_program

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
    return Equation(text, dict(initial), repr(canonical(tree)), model, estimator)


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
        "and 'parameters' (free parameter definitions). Do not fit or rank."
    )


def _baseline(up):
    seeds = up["seed_programs"]
    return ParentFit(BASELINE_ID, seeds.neuron_model_2_jax, seeds.parameter_estimator_2)


def _measure(up, model, cohort, max_iter):
    x, y = cohort
    score, mse, count, params = score_program(up, model.model, model.estimator, x, y, max_iter)
    if not math.isfinite(score):
        raise ValueError("Nonfinite fit.")
    held = jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(x.shape[1]))[x.shape[1] // 2 :]
    angles = x[:, held]
    fitted = jax.vmap(model.model)(angles, *params.T)
    errors = np.asarray((fitted - y[:, held]) ** 2)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite predictions.")
    bins = np.floor((np.rad2deg(np.asarray(angles)) % 360) / 30).astype(int)
    return {
        "score": score,
        "mse": mse,
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
):
    """Explore rolling waves, then confirm a frozen shortlist; optionally select and test once."""
    validate_graph(ideas)
    if min(batches, k, waves) < 1:
        raise ValueError("Wave count, batch count and equations per batch must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "wave.json"
    if path.exists():
        raise FileExistsError(path)
    nodes = {i.node: i for i in ideas}
    references, metrics, attempts, depth, seen = {}, {}, Counter(), Counter(), set()
    lineage = {}
    rng = np.random.default_rng(seed)
    born = set()
    wave = {
        "version": 3,
        "seed": seed,
        "idea_catalog": [asdict(i) for i in ideas],
        "graph": [baseline_node()],
        "planned_batches": batches * waves,
        "episodes": [],
        "confirmation": [],
        "status": "exploring",
    }
    _write(path, wave)
    wave["baseline_exploration"] = _measure(up, _baseline(up), explore, max_iter)
    for step in range(batches * waves):
        choice = draw(ideas, references, attempts, rng, depth)
        if choice is None:
            break
        node, parents = choice["node"], choice["parents"]
        idea = nodes[node]
        attempts[node] += 1
        fitted = tuple(ParentFit(p, references[p].model, references[p].estimator) for p in parents)
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
        try:
            batch = prepare_images(up, *explore, seed=event["image_seed"], parents=fitted, max_iter=max_iter)
            event.update(cells=list(batch.cells), images=len(batch.images))
            event["reply"] = await proposer.submit(batch, prompt)
            proposals = _decode(event["reply"], k)
        except ValueError as exc:
            event["error"] = str(exc)
            proposals = []
        for proposal in proposals:
            record = {"proposal": proposal}
            event["candidates"].append(record)
            try:
                candidate = compile_equation(proposal)
            except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
                record.update(status="invalid", error=str(exc))
                continue
            record["equation"] = candidate.text
            if candidate.key in seen:
                record["status"] = "duplicate"
                continue
            seen.add(candidate.key)
            if node not in born:
                born.add(node)
                routes = idea.routes or (() if idea.lane == "root" else ((BASELINE_ID,),))
                wave["graph"].append({**asdict(idea), "routes": routes, "introduced_at": step + 1})
            try:
                result = _measure(up, candidate, explore, max_iter)
                record.update(status="fitted", metrics=result)
                depth[idea.branch] += 1
                if node not in references:
                    references[node], metrics[node] = candidate, result
                    lineage[node] = parents
            except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
                record.update(status="fit_failed", error=str(exc))
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
            result = _measure(up, eq, confirm, max_iter)
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
        selection, locked = final
        scores = {"double_gaussian": _measure(up, _baseline(up), selection, max_iter)}
        for node in sorted(n for n in supported if supported_claim(n)):
            try:
                scores[node] = _measure(up, references[node], selection, max_iter)
            except ValueError:
                continue
        winner = min(scores, key=lambda name: (scores[name]["score"], name))
        wave["selection"] = {"metrics": scores, "winner": winner}
        _write(path, wave)
        locked_baseline = _measure(up, _baseline(up), locked, max_iter)
        wave["locked"] = {"double_gaussian": locked_baseline}
        if winner != "double_gaussian":
            wave["locked"][winner] = _measure(up, references[winner], locked, max_iter)
    wave["status"] = "complete"
    _write(path, wave)
    return json.loads(path.read_text())


class _Live:
    def __init__(self, client):
        self.client = client

    async def submit(self, batch, prompt):
        return await batch.submit(self.client, prompt, effort="none", temperature=0.5)


def run(args):
    """Load the pinned recording and run one explicitly bounded, free-model wave."""
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write(
        output / "manifest.json",
        {
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
        },
    )

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
            )
        finally:
            await go.close()

    result = asyncio.run(wave())
    print(f"{result['status']}: {output / 'wave.json'}")
    if result["status"] != "complete":
        raise SystemExit(1)

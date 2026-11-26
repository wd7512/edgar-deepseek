"""Ideas, the prerequisite graph and route draws.

Draws are score-blind by default. With a `Guide`, the scheduler (never the model) weights every choice of
a node (revisits, parents, catalog ideas and the routes made of parents) by exploration score, so the
search climbs while every eligible node stays possible. All such choices go through `pick_nodes`.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from edgar_study.frontier.replies import InvalidReply
from edgar_study.frontier.seed import BASELINE_ID

LANES = {"frontier": 0.6, "detour": 0.3, "root": 0.1}
BLIND_MODES = {"single_idea": 0.6, "anomaly_detour": 0.2, "independent_root": 0.1, "combination": 0.1}
GUIDED_MODES = {"single_idea": 0.5, "anomaly_detour": 0.15, "independent_root": 0.05, "combination": 0.3}
UNFITTED = 0.05


@dataclass(frozen=True)
class Guide:
    """Exploration scores (lower is better) and how strongly to favour low ones, as exactly one of:

    `temperature` (in score units): weight exp(-(score - lowest) / temperature).
    `rank_bias`: weight 1 / (1 + rank_bias * rank / worst_rank), by rank alone, so the best name has
    weight 1 and the worst 1 / (1 + rank_bias) however the scores are spread.
    """

    scores: dict
    temperature: float | None = None
    rank_bias: float | None = None

    def __post_init__(self):
        if (self.temperature is None) == (self.rank_bias is None):
            raise ValueError("Give a guide either a temperature or a rank bias.")
        if self.rank_bias is not None and self.rank_bias < 0:
            raise ValueError("Rank bias must be zero or more.")

    def weights(self, names):
        """A weight per name; names without a score get a small weight."""
        known = {n: self.scores[n] for n in names if n in self.scores}
        if self.rank_bias is not None:
            rank = {n: sum(other < score for other in known.values()) for n, score in known.items()}
            worst = max(rank.values(), default=0)
            return {
                n: 1 / (1 + self.rank_bias * rank[n] / worst)
                if n in known and worst
                else 1.0
                if n in known
                else UNFITTED
                for n in names
            }
        lowest = min(known.values(), default=0.0)
        return {
            n: math.exp(-(known[n] - lowest) / self.temperature) if n in known else UNFITTED for n in names
        }


@dataclass(frozen=True)
class Idea:
    node: str
    branch: str
    mechanism: str
    text: str
    routes: tuple[tuple[str, ...], ...] = ()
    lane: str = "frontier"
    label: str = ""


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


def pick_nodes(pool, count, rng, guide=None, uniform=False, base=None):
    """Draw `count` distinct names from `pool`; return them and each name's first-draw probability.

    A name's weight is its `base` weight (default 1) times the guide's weight. With no guide, or with
    `uniform`, the guide is ignored: `uniform` is how a caller asks for a totally random pick.
    """
    pool = list(pool)
    use = guide is not None and not uniform
    if not use and base is None:
        picks = rng.choice(len(pool), count, replace=False)
        return [pool[int(i)] for i in picks], {name: 1 / len(pool) for name in pool}
    fit = guide.weights(pool) if use else {}
    weights = np.array([(base or {}).get(name, 1.0) * fit.get(name, 1.0) for name in pool], dtype=float)
    probabilities = weights / weights.sum()
    if count == 1:
        picks = [int(rng.choice(len(pool), p=probabilities))]
    else:
        picks = [int(i) for i in rng.choice(len(pool), count, replace=False, p=probabilities)]
    return [pool[i] for i in picks], dict(zip(pool, map(float, probabilities), strict=True))


def pick_route(routes, rng, guide=None):
    """Draw one route, uniformly or, with a guide, by the mean weight of the parents on it."""
    if guide is None:
        return routes[int(rng.integers(len(routes)))], 1 / len(routes)
    fit = guide.weights({parent for route in routes for parent in route})
    weights = np.array([np.mean([fit[parent] for parent in route] or [1.0]) for route in routes])
    probabilities = weights / weights.sum()
    index = int(rng.choice(len(routes), p=probabilities))
    return routes[index], float(probabilities[index])


def draw(ideas, references, attempts, rng, depth=None, guide=None):
    """Sample an eligible lane, branch, idea and complete route; a guide weights ideas and routes by score."""
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
    base = {
        i.node: 1 / ((1 + attempts.get(i.node, 0)) * (1 + tried[i.mechanism] - bool(attempts.get(i.node))))
        for i in pool
        if i.branch == branch
    }
    picked, node_p = pick_nodes(list(base), 1, rng, guide, base=base)
    node = picked[0]
    parents, route_p = pick_route(eligible[node][1], rng, guide)
    return {
        "lane": lane,
        "branch": branch,
        "node": node,
        "parents": list(parents),
        "lane_probabilities": lane_p,
        "branch_probabilities": branch_p,
        "node_probabilities": node_p,
        "route_probability": route_p,
        "eligible": [
            {"node": i.node, "parents": list(route)} for i, routes in eligible.values() for route in routes
        ],
    }


def draw_generated(nodes, references, history, rng, new_probability, step, guide=None):
    """Draw a new route or revisit. Without a guide only proposal validity counts, never fit quality."""
    types = {"new": new_probability, "revisit": 1 - new_probability} if nodes else {"new": 1}
    kind, kind_p = _pick({key: value for key, value in types.items() if value}, rng)
    choice: dict[str, Any] = {"kind": kind, "kind_probabilities": kind_p}
    if kind == "revisit":
        validity = {}
        for node in nodes:
            recent = history.get(node, [])[-16:]
            validity[node] = 1 - 0.85 * sum(recent) / max(1, len(recent))
        picked, node_p = pick_nodes(list(nodes), 1, rng, guide, base=validity)
        node = picked[0]
        routes = [
            route
            for route in nodes[node].routes or ((),)
            if all(parent == BASELINE_ID or parent in references for parent in route)
        ]
        parents, route_p = pick_route(routes, rng, guide)
        return {
            **choice,
            "node": node,
            "parents": list(parents),
            "mode": "revisit",
            "lane": nodes[node].lane,
            "node_probabilities": node_p,
            "route_probability": route_p,
        }
    pool = [BASELINE_ID, *references]
    weights = dict(GUIDED_MODES if guide else BLIND_MODES)
    if len(references) < 2:
        del weights["combination"]
    mode, mode_p = _pick(weights, rng)
    if mode == "combination":
        pool = list(references)
    count = {"independent_root": 0, "combination": 2}.get(mode, 1)
    extra = {}
    if guide and count:
        picked, extra["parent_probabilities"] = pick_nodes(pool, count, rng, guide)
    else:
        picked, _ = pick_nodes(pool, count, rng)
    parents = sorted(picked)
    return {
        **extra,
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


def generated_idea(value, choice, existing):
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


def route_mode(idea, parents):
    if parents:
        return "combination"
    return {"root": "independent_root", "detour": "anomaly_detour", "frontier": "single_idea"}[idea.lane]

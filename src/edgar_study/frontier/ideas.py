"""Ideas, the prerequisite graph and score-blind route draws."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from edgar_study.frontier.replies import InvalidReply
from edgar_study.frontier.seed import BASELINE_ID

LANES = {"frontier": 0.6, "detour": 0.3, "root": 0.1}


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

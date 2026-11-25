from collections import Counter

import numpy as np
import pytest

from edgar_study.frontier.ideas import Idea, draw, draw_generated, generated_idea, validate_graph
from edgar_study.frontier.replies import InvalidReply


def test_generated_idea_mechanism_defaults_to_the_title():
    choice = {"node": "idea-1", "mode": "independent_root", "parents": [], "lane": "root"}
    accepted = generated_idea({"title": "Circular Skew", "change": "Skew each lobe."}, choice, {})
    assert accepted.mechanism == "circular-skew"


def test_draw_requires_a_complete_route_but_allows_alternatives():
    ideas = (
        Idea("a", "peak", "a", "a"),
        Idea("b", "peak", "b", "b"),
        Idea("c", "peak", "c", "c", (("a", "b"), ("b",))),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_graph((Idea("x", "peak", "x", "x", (("x",),)),))
    first = draw(ideas, {}, {}, np.random.default_rng(8))
    assert first["node"] in {"a", "b"}
    eligible = draw(ideas, {"b": object()}, {}, np.random.default_rng(8))
    c = next(row for row in eligible["eligible"] if row["node"] == "c")
    assert c["parents"] == ["b"]


def test_generated_idea_identity_uses_route_and_wording_not_mechanism():
    existing = {
        "idea-1": Idea(
            "idea-1",
            "single_idea",
            "offset",
            "Shift the m lobe only.",
            (("double_gaussian",),),
            "frontier",
            "Directional peak shift",
        )
    }
    choice = {"node": "idea-2", "mode": "single_idea", "parents": ["double_gaussian"], "lane": "frontier"}
    distinct = generated_idea(
        {"title": "Shared separation offset", "mechanism": "offset", "change": "Move both lobes."},
        choice,
        existing,
    )
    assert distinct.label == "Shared separation offset"
    with pytest.raises(InvalidReply, match="Existing ideas on this route: Directional peak shift"):
        generated_idea(
            {"title": "directional peak shift", "mechanism": "width", "change": "Move both lobes."},
            choice,
            existing,
        )


def test_idea_description_allows_six_hundred_characters():
    choice = {"node": "idea-2", "mode": "single_idea", "parents": [], "lane": "root"}
    accepted = generated_idea(
        {"title": "Long change", "mechanism": "mixture", "change": "a" * 600}, choice, {}
    )
    assert accepted.text == "a" * 600
    with pytest.raises(InvalidReply, match="at most 600"):
        generated_idea({"title": "Too long", "mechanism": "mixture", "change": "a" * 601}, choice, {})


def test_tried_independent_root_can_be_probed_again():
    root = Idea("root", "peaks", "harmonic", "Use circular harmonics.", lane="root")
    choice = draw((root,), {}, {"root": 1}, np.random.default_rng(8))
    assert choice["node"] == "root"


def test_draw_only_uses_fitted_idea_parents_and_downweights_rejected_revisits():
    def idea(node):
        return Idea(node, "root", node, node, lane="root")

    nodes = {key: idea(key) for key in ("bad", "good", "unfitted")}
    history = {"bad": [True] * 16, "good": [False] * 16}
    revisit = draw_generated(nodes, {"bad", "good"}, history, np.random.default_rng(0), 0, 5)
    weights = revisit["node_probabilities"]
    assert 0 < weights["bad"] < weights["good"]
    assert weights["unfitted"] > 0
    combinations = 0
    for seed in range(60):
        new = draw_generated(nodes, {"good": None, "bad": None}, history, np.random.default_rng(seed), 1, 5)
        assert set(new["parents"]) <= {"double_gaussian", "good", "bad"}
        if new["mode"] == "combination":
            combinations += 1
            assert set(new["parents"]) == {"good", "bad"}
    assert combinations > 0


@pytest.mark.parametrize(
    "ideas",
    [
        (Idea("a", "p", "m", "t"), Idea("a", "p", "m", "t")),
        (Idea("double_gaussian", "p", "m", "t"),),
        (Idea("a", "p", "m", "t", (("b",),)), Idea("b", "p", "m", "t", (("a",),))),
        (Idea("a", "p", "m", "t", (("ghost",),)),),
        (Idea("a", "p", "m", "t", (("b", "b"),)), Idea("b", "p", "m", "t")),
        (Idea("a", "p", "m", "t", ((),)),),
        (Idea("a", "p", "m", "t", lane="sideways"),),
        (Idea("a", "p", "m", "t", (("b",),), lane="root"), Idea("b", "p", "m", "t")),
    ],
)
def test_invalid_idea_graphs_are_rejected(ideas):
    with pytest.raises(ValueError):
        validate_graph(ideas)


def test_valid_graphs_with_alternative_routes_are_accepted():
    validate_graph(
        (
            Idea("a", "p", "m", "t"),
            Idea("b", "p", "m", "t"),
            Idea("c", "p", "m", "t", (("a",), ("a", "b"))),
        )
    )


def test_generated_draws_follow_the_documented_split_and_mode_weights():
    nodes = {"x": Idea("x", "new", "m", "t", lane="root"), "y": Idea("y", "new", "m", "t", lane="root")}
    references = {"x": None, "y": None}
    rng = np.random.default_rng(3)
    draws = [draw_generated(nodes, references, {}, rng, 0.8, 5) for _ in range(4000)]
    kinds = Counter(d["kind"] for d in draws)
    assert 0.77 < kinds["new"] / 4000 < 0.83
    modes = Counter(d["mode"] for d in draws if d["kind"] == "new")
    total = sum(modes.values())
    for mode, share in {
        "single_idea": 0.6,
        "anomaly_detour": 0.2,
        "independent_root": 0.1,
        "combination": 0.1,
    }.items():
        assert abs(modes[mode] / total - share) < 0.04, mode
    for d in draws:
        if d["kind"] == "new":
            assert d["node"] == "idea-6"
            expected = {"independent_root": 0, "combination": 2}.get(d["mode"], 1)
            assert len(d["parents"]) == expected == len(set(d["parents"]))
        if d["mode"] == "combination":
            assert set(d["parents"]) == {"x", "y"}


def test_first_generated_draw_is_always_new_and_new_probability_extremes_are_respected():
    rng = np.random.default_rng(0)
    assert draw_generated({}, {}, {}, rng, 0, 0)["kind"] == "new"
    nodes = {"x": Idea("x", "new", "m", "t", lane="root")}
    assert {draw_generated(nodes, {}, {}, rng, 1, 1)["kind"] for _ in range(50)} == {"new"}
    assert {draw_generated(nodes, {}, {}, rng, 0, 1)["kind"] for _ in range(50)} == {"revisit"}


def test_catalog_draw_returns_nothing_until_a_complete_route_is_fitted():
    ideas = (Idea("a", "p", "m", "t"), Idea("b", "p", "m2", "t", (("a",),), lane="frontier"))
    rng = np.random.default_rng(0)
    assert draw((), {}, {}, rng) is None
    first = draw(ideas, {}, {}, rng)
    assert first["node"] == "a"
    assert all(row["node"] == "a" for row in first["eligible"])
    assert {row["node"] for row in draw(ideas, {"a": object()}, {}, rng)["eligible"]} == {"a", "b"}

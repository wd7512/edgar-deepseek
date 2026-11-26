from collections import Counter

import numpy as np
import pytest

from edgar_study.frontier.ideas import (
    Guide,
    Idea,
    draw,
    draw_generated,
    generated_idea,
    pick_nodes,
    validate_graph,
)
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


def _guided_nodes():
    return {
        name: Idea(name, "single_idea", name, f"Idea {name}.", (("double_gaussian",),), "frontier", name)
        for name in ("good", "poor")
    }


def test_a_guide_draws_better_fitting_parents_more_often_but_never_excludes_one():
    guide = Guide({"double_gaussian": 28.6, "good": 28.5, "poor": 29.5}, temperature=0.1)
    references = {"good": None, "poor": None}
    counts = Counter()
    for seed in range(400):
        choice = draw_generated(_guided_nodes(), references, {}, np.random.default_rng(seed), 1, 5, guide)
        if choice["mode"] in ("single_idea", "anomaly_detour"):
            counts.update(choice["parents"])
            probabilities = choice["parent_probabilities"]
    assert counts["good"] > 2 * counts["double_gaussian"] > 0
    assert probabilities["poor"] > 0 and counts["poor"] < counts["good"] / 20


def test_a_guided_combination_draws_two_different_parents():
    guide = Guide({"double_gaussian": 28.6, "good": 28.5, "poor": 28.55}, temperature=0.1)
    pairs = [
        choice["parents"]
        for seed in range(300)
        if (
            choice := draw_generated(
                _guided_nodes(), {"good": None, "poor": None}, {}, np.random.default_rng(seed), 1, 5, guide
            )
        )["mode"]
        == "combination"
    ]
    assert pairs and all(len(set(pair)) == 2 for pair in pairs)


def test_a_guide_favours_combinations_over_unrelated_roots():
    guide = Guide({"double_gaussian": 28.6, "good": 28.5, "poor": 28.55}, temperature=0.1)
    modes = Counter(
        draw_generated(
            _guided_nodes(), {"good": None, "poor": None}, {}, np.random.default_rng(s), 1, 5, guide
        )["mode"]
        for s in range(2000)
    )
    assert modes["combination"] > 4 * modes["independent_root"] > 0


def test_guided_revisits_favour_better_fitting_ideas():
    guide = Guide({"double_gaussian": 28.6, "good": 28.5, "poor": 29.5}, temperature=0.1)
    choice = draw_generated(
        _guided_nodes(), {"good": None, "poor": None}, {}, np.random.default_rng(0), 0, 5, guide
    )
    weights = choice["node_probabilities"]
    assert weights["good"] > 10 * weights["poor"] > 0


SCORES = {"a": 28.0, "b": 28.4, "c": 29.0, "d": 29.0}


def test_rank_weights_depend_only_on_rank_with_the_worst_at_one_over_one_plus_bias():
    weights = Guide(SCORES, rank_bias=1.0).weights(["a", "b", "c", "d", "new"])
    assert weights["a"] == 1.0
    assert weights["a"] > weights["b"] > weights["c"]
    assert weights["c"] == weights["d"] == pytest.approx(0.5)
    assert weights["new"] == pytest.approx(0.05)
    shifted = Guide({k: v * 10 + 3 for k, v in SCORES.items()}, rank_bias=1.0).weights(["a", "b", "c", "d"])
    assert shifted == {k: weights[k] for k in "abcd"}


def test_zero_rank_bias_and_a_single_scored_name_weigh_everything_equally():
    assert set(Guide(SCORES, rank_bias=0.0).weights(["a", "b", "c"]).values()) == {1.0}
    assert Guide(SCORES, rank_bias=3.0).weights(["a"]) == {"a": 1.0}


def test_a_guide_needs_exactly_one_of_temperature_and_rank_bias():
    with pytest.raises(ValueError):
        Guide(SCORES)
    with pytest.raises(ValueError):
        Guide(SCORES, temperature=1.0, rank_bias=1.0)
    with pytest.raises(ValueError):
        Guide(SCORES, rank_bias=-0.1)


def test_pick_nodes_returns_distinct_names_and_probabilities_that_sum_to_one():
    guide = Guide(SCORES, rank_bias=1.0)
    names, probabilities = pick_nodes(["a", "b", "c"], 2, np.random.default_rng(0), guide)
    assert len(set(names)) == 2 and set(names) <= {"a", "b", "c"}
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_rank_bias_makes_the_best_node_about_twice_as_likely_as_the_worst_and_never_excludes_one():
    guide = Guide({"a": 1.0, "b": 2.0, "c": 3.0}, rank_bias=1.0)
    rng = np.random.default_rng(0)
    counts = Counter(pick_nodes(["a", "b", "c"], 1, rng, guide)[0][0] for _ in range(6000))
    assert counts["c"] > 0
    assert 1.8 < counts["a"] / counts["c"] < 2.2


def test_a_uniform_pick_ignores_the_guide():
    guide = Guide({"a": 1.0, "b": 2.0}, rank_bias=50.0)
    rng = np.random.default_rng(0)
    counts = Counter(pick_nodes(["a", "b"], 1, rng, guide, uniform=True)[0][0] for _ in range(4000))
    assert 0.9 < counts["a"] / counts["b"] < 1.1


def test_a_base_weight_is_multiplied_by_the_rank_weight():
    guide = Guide({"a": 1.0, "b": 2.0}, rank_bias=1.0)
    _, probabilities = pick_nodes(["a", "b"], 1, np.random.default_rng(0), guide, base={"a": 1.0, "b": 2.0})
    assert probabilities["a"] == pytest.approx(probabilities["b"])


def test_a_rank_guide_biases_revisits_parents_and_routes_alike():
    guide = Guide({"double_gaussian": 28.6, "good": 28.0, "poor": 29.5}, rank_bias=1.0)
    nodes = _guided_nodes()
    revisit = draw_generated(nodes, {"good": None, "poor": None}, {}, np.random.default_rng(0), 0, 5, guide)
    assert revisit["node_probabilities"]["good"] == pytest.approx(2 * revisit["node_probabilities"]["poor"])
    new = draw_generated(nodes, {"good": None, "poor": None}, {}, np.random.default_rng(1), 1, 5, guide)
    while new["mode"] not in ("single_idea", "anomaly_detour"):
        new = draw_generated(nodes, {"good": None, "poor": None}, {}, np.random.default_rng(2), 1, 5, guide)
    assert new["parent_probabilities"]["good"] > new["parent_probabilities"]["poor"] > 0
    routed = {
        "x": Idea("x", "p", "m", "t", (("good",), ("poor",)), "frontier", "x"),
        "good": nodes["good"],
        "poor": nodes["poor"],
    }
    counts = Counter()
    for seed in range(1500):
        choice = draw_generated(
            routed, {"good": None, "poor": None}, {}, np.random.default_rng(seed), 0, 5, guide
        )
        if choice["node"] == "x":
            counts[tuple(choice["parents"])] += 1
    assert counts[("good",)] > 1.5 * counts[("poor",)] > 0


def test_a_rank_guide_biases_the_catalog_draw_among_ideas_of_one_branch():
    ideas = (Idea("good", "peak", "m1", "t"), Idea("poor", "peak", "m2", "t"))
    guide = Guide({"good": 1.0, "poor": 2.0}, rank_bias=3.0)
    counts = Counter(
        draw(ideas, {"good": None, "poor": None}, {}, np.random.default_rng(s), None, guide)["node"]
        for s in range(3000)
    )
    assert counts["good"] > 2 * counts["poor"] > 0

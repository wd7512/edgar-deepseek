import asyncio
import fcntl
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgar_study.frontier_runner import (
    Idea,
    InvalidEquation,
    compile_equation,
    draw,
    draw_generated,
    execute,
    run,
    validate_graph,
)
from edgar_study.upstream import import_upstream

ROOT = Path(__file__).resolve().parents[2]
SCORE_WORDS = ("mse", "score", "loss", "ranking", "best")


def cohort():
    x = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    return x, 2 + np.cos(x)


class Scripted:
    """A proposer whose replies depend only on the batch number and the kind of prompt."""

    def __init__(self, output, interrupt_at=None):
        self.output = output
        self.interrupt_at = interrupt_at
        self.prompts = []

    def batch(self):
        return json.loads((self.output / "wave.json").read_text())["active_probe"]["batch"]

    async def submit(self, batch, prompt):
        self.prompts.append(prompt)
        step = self.batch()
        if step == self.interrupt_at:
            raise InterruptedError
        equation = {"equation": f"b + a*cos({step}*theta)", "initial": {"b": 2, "a": 1}}
        if prompt.startswith("Revisit"):
            return json.dumps([equation])
        idea = {"title": f"Harmonic {step}", "mechanism": f"harmonic-{step}", "change": f"Use harmonic {step}."}
        return json.dumps({"idea": idea, "equations": [equation]})

    async def retry(self, batch, prompt, previous_reply, instruction):
        self.prompts.append(instruction)
        return "[]"


def explore(output, *, seed=0, batches=4, resume=False, interrupt_at=None, **kwargs):
    x, y = cohort()
    proposer = Scripted(output, interrupt_at)
    result = asyncio.run(
        execute(
            import_upstream(ROOT),
            proposer,
            (x, y),
            (x, y),
            output,
            seed=seed,
            batches=batches,
            k=2,
            max_iter=0,
            resume=resume,
            **kwargs,
        )
    )
    return result, proposer


def choices(result):
    return [(e["kind"], e["node"], e["parents"], e["mode"], e["lane"]) for e in result["episodes"]]


def outcomes(result):
    return (
        choices(result),
        [[(c.get("status"), c.get("equation")) for c in e["candidates"]] for e in result["episodes"]],
        [(row["node"], row["outcome"]) for row in result["confirmation"]],
        result["claims"],
    )


@pytest.mark.parametrize(
    ("text", "initial", "code"),
    [
        ("a*log(theta)", {"a": 1}, "unsupported_operator"),
        ("__import__('os').system('false')", {"a": 1}, "unsupported_operator"),
        ("a*cos(theta", {"a": 1}, "invalid_syntax"),
        ("a*cos(x)", {"a": 1}, "unknown_name"),
        ("cos(theta)", {"a": 1}, "parameter_mismatch"),
        ("a*cos(theta)", {"a": 1, "c": 1}, "parameter_mismatch"),
        ("a*theta", {"a": 1}, "nonperiodic_expression"),
        ("a/(theta-theta)", {"a": 1}, "nonfinite_expression"),
        ("a*cos(theta)", {"a": 500}, "invalid_parameter"),
        ("a*cos(theta)", {"theta": 1}, "invalid_parameter"),
        ("a*cos(theta)", {}, "invalid_parameter_count"),
        ("a*cos(theta)" + " " * 600, {"a": 1}, "equation_too_complex"),
        ("a*cos(theta)", [], "invalid_equation_object"),
    ],
)
def test_equations_are_rejected_with_a_stable_reason_code(text, initial, code):
    with pytest.raises(InvalidEquation) as caught:
        compile_equation({"equation": text, "initial": initial})
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("text", "initial"),
    [
        ("b + a*cos(theta)", {"a": 1, "b": 1}),
        ("b + a*exp(-0.5*(wrap(theta-m)/s)**2)", {"a": 1, "b": 1, "m": 1, "s": 1}),
        ("b + a*(cos(theta) < c)", {"a": 1, "b": 1, "c": 0}),
        ("b + a*(c >= cos(theta))", {"a": 1, "b": 1, "c": 0}),
        ("-a*sin(2*theta) + b", {"a": 1, "b": 1}),
    ],
)
def test_equations_in_the_grammar_are_accepted(text, initial):
    assert compile_equation({"equation": text, "initial": initial}).text == text


def test_equations_that_differ_only_by_renaming_or_ordering_share_a_key():
    one = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 2}})
    two = compile_equation({"equation": "q*cos(theta) + c", "initial": {"q": 3, "c": 4}})
    three = compile_equation({"equation": "b + a*cos(2*theta)", "initial": {"b": 1, "a": 2}})
    assert one.key == two.key
    assert one.key != three.key


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
    for mode, share in {"single_idea": 0.6, "anomaly_detour": 0.2, "independent_root": 0.1, "combination": 0.1}.items():
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


def test_a_seeded_run_is_reproducible_and_the_seed_matters(tmp_path):
    first, _ = explore(tmp_path / "one", seed=7, batches=6)
    again, _ = explore(tmp_path / "two", seed=7, batches=6)
    other, _ = explore(tmp_path / "three", seed=8, batches=6)
    assert choices(first) == choices(again)
    assert choices(first) != choices(other)
    assert [e["kind"] for e in first["episodes"]][0] == "new"
    assert first["seed"] == 7


def test_a_run_writes_one_episode_per_batch_and_a_consistent_checkpoint(tmp_path):
    result, proposer = explore(tmp_path, batches=3, new_probability=1)
    assert result["status"] == "complete"
    assert json.loads((tmp_path / "wave.json").read_text()) == result
    assert len(result["episodes"]) == 3
    assert [row["node"] for row in result["graph"]] == ["double_gaussian", "idea-1", "idea-2", "idea-3"]
    assert [row["introduced_at"] for row in result["graph"]] == [0, 1, 2, 3]
    assert set(result["shortlist"]) == {"idea-1", "idea-2", "idea-3"}
    assert all(e["candidates"][0]["status"] == "fitted" for e in result["episodes"])
    assert not (tmp_path / "wave.tmp").exists()
    assert len(proposer.prompts) == 3


def test_prompts_never_reveal_fit_quality(tmp_path):
    _, proposer = explore(tmp_path, batches=5)
    for prompt in proposer.prompts:
        assert [word for word in SCORE_WORDS if word in prompt.lower()] == []


def test_a_perfect_fit_is_confirmed_and_an_unfittable_one_stays_unresolved(tmp_path):
    result, _ = explore(tmp_path, batches=2, new_probability=1)
    rows = {row["node"]: row for row in result["confirmation"]}
    assert rows["idea-1"]["outcome"] == "confirmed_fit"
    assert rows["idea-1"]["metrics"]["mse"] < result["confirmation_baseline"]["mse"]
    assert rows["idea-2"]["outcome"] == "unresolved"
    assert result["claims"] == {"idea-1": "supported_fit", "idea-2": "unresolved"}


def test_a_run_with_no_fitted_equation_reports_it_and_stops(tmp_path):
    x, y = cohort()

    class Nothing:
        async def submit(self, batch, prompt):
            return json.dumps({"idea": {"title": "Nothing", "change": "Nothing."}, "equations": []})

        async def retry(self, batch, prompt, previous_reply, instruction):
            return "[]"

    result = asyncio.run(
        execute(import_upstream(ROOT), Nothing(), (x, y), (x, y), tmp_path, batches=2, k=2, max_iter=0)
    )
    assert result["status"] == "no_fitted_equations"
    assert result["shortlist"] == {}
    assert result["confirmation"] == []


def test_an_interrupted_run_resumes_to_the_same_result_as_an_uninterrupted_one(tmp_path):
    whole, _ = explore(tmp_path / "whole", seed=11, batches=5)
    with pytest.raises(InterruptedError):
        explore(tmp_path / "split", seed=11, batches=5, interrupt_at=4)
    paused = json.loads((tmp_path / "split" / "wave.json").read_text())
    assert len(paused["episodes"]) == 3
    assert paused["status"] == "exploring"
    resumed, _ = explore(tmp_path / "split", seed=11, batches=5, resume=True)
    assert outcomes(resumed) == outcomes(whole)
    assert [e["prompt"] for e in resumed["episodes"]] == [e["prompt"] for e in whole["episodes"]]


def test_resuming_a_finished_run_returns_it_unchanged(tmp_path):
    done, _ = explore(tmp_path, batches=2)
    again, proposer = explore(tmp_path, batches=2, resume=True)
    assert again == done
    assert proposer.prompts == []


def test_a_checkpoint_is_not_overwritten_and_only_resumes_with_the_same_configuration(tmp_path):
    explore(tmp_path, seed=1, batches=2)
    with pytest.raises(FileExistsError):
        explore(tmp_path, seed=1, batches=2)
    with pytest.raises(ValueError):
        explore(tmp_path, seed=2, batches=2, resume=True)
    with pytest.raises(ValueError):
        explore(tmp_path, seed=1, batches=3, resume=True)
    with pytest.raises(ValueError):
        explore(tmp_path, seed=1, batches=2, resume=True, new_probability=0.5)


@pytest.mark.parametrize(
    "options",
    [{"batches": 0}, {"waves": 0}, {"new_probability": 1.5}, {"new_probability": -0.1}],
)
def test_invalid_run_options_are_refused_before_any_output(tmp_path, options):
    with pytest.raises(ValueError):
        explore(tmp_path / "run", **options)
    assert not (tmp_path / "run" / "wave.json").exists()


def test_finalizing_selects_once_and_tests_the_winner_on_locked_cells(tmp_path):
    x, y = cohort()
    final = ((x, y), (x, y))
    result, _ = explore(tmp_path, batches=2, new_probability=1, final=final)
    assert set(result["selection"]["metrics"]) == {"double_gaussian", "idea-1"}
    assert result["selection"]["winner"] == "idea-1"
    assert set(result["locked"]) == {"double_gaussian", "idea-1"}
    again, proposer = explore(tmp_path, batches=2, resume=True, new_probability=1, final=final)
    assert again == result
    assert proposer.prompts == []


def test_a_fixed_catalog_run_only_probes_ideas_whose_routes_are_fitted(tmp_path):
    ideas = (
        Idea("harmonic", "peak", "harmonic", "Use one harmonic."),
        Idea("second", "peak", "second", "Use a second harmonic.", (("harmonic",),)),
    )
    x, y = cohort()

    class Catalog:
        async def submit(self, batch, prompt):
            return json.dumps([{"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}}])

        async def retry(self, batch, prompt, previous_reply, instruction):
            return "[]"

    result = asyncio.run(
        execute(
            import_upstream(ROOT),
            Catalog(),
            (x, y),
            (x, y),
            tmp_path,
            ideas=ideas,
            batches=3,
            k=1,
            max_iter=0,
        )
    )
    assert result["idea_source"] == "catalog"
    assert result["episodes"][0]["node"] == "harmonic"
    assert result["episodes"][0]["parents"] == []
    for event in result["episodes"]:
        assert set(event["parents"]) <= {"harmonic"}


def args(output, **overrides):
    values = {
        "output": output,
        "resume": False,
        "waves": 1,
        "batches": 1,
        "k": 1,
        "seed": 0,
        "graph": None,
        "new_probability": 0.8,
        "finalize": False,
    }
    return SimpleNamespace(**{**values, **overrides})


def test_resuming_needs_an_existing_manifest_and_checkpoint(tmp_path):
    with pytest.raises(SystemExit, match="Resume requires"):
        run(args(tmp_path, resume=True))


def test_a_run_refuses_to_start_twice_on_one_directory(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "wave.json").write_text("{}")
    with (tmp_path / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="already active"):
            run(args(tmp_path, resume=True))


def test_a_new_run_refuses_an_existing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        run(args(tmp_path))


@pytest.mark.parametrize(
    "overrides", [{"waves": 0}, {"batches": 0}, {"k": 0}, {"seed": -1}, {"new_probability": 2}]
)
def test_run_rejects_unusable_options_before_touching_the_network(tmp_path, overrides):
    with pytest.raises(SystemExit):
        run(args(tmp_path / "out", **overrides))

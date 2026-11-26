import asyncio
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.runner import execute, run
from edgar_study.frontier.seed import BASE
from edgar_study.frontier.viewer import render
from edgar_study.upstream import import_upstream


def test_every_proposal_that_cannot_be_used_is_reported_back_and_replaced_without_scores(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False, dtype=np.float32), (10, 1))
    responses = 1 + 0.4 * np.cos(angles)
    invalid = {"equation": "a*log(theta)", "initial": {"a": 1}}
    reply = json.dumps(
        [
            {"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 1}},
            {"equation": BASE, "initial": {"m": 0, "s": 1, "a1": 1, "b": 1, "a2": 1}},
            invalid,
            {"equation": "a*cos(theta) + b", "initial": {"a": 1, "b": 1}},
            {"equation": "a*exp(60*sin(theta)) + b", "initial": {"b": 7, "a": 1}},
        ]
    )
    harmonic = [{"equation": f"b + a*cos({n}*theta)", "initial": {"b": 1, "a": 1}} for n in (2, 3, 4)]
    replies = [
        json.dumps([invalid, *harmonic]),
        json.dumps([{"equation": "b + a*sin(theta)", "initial": {"b": 1, "a": 1}}]),
    ]

    class Stub:
        async def submit(self, batch, prompt):
            return reply

        async def retry(self, batch, prompt, history):
            return replies.pop(0)

    result = asyncio.run(
        execute(
            up,
            Stub(),
            (angles, responses),
            (angles, responses),
            tmp_path,
            ideas=(Idea("a", "peak", "harmonic", "Try a harmonic."),),
            batches=1,
            k=5,
        )
    )
    event = result["episodes"][0]
    first, second = (turn["instruction"] for turn in event["turns"])
    assert BASE in first
    assert "Proposal 3: unsupported_operator" in first
    assert "Proposal 4: duplicate" in first
    assert "Proposal 5: fit_failed" in first
    assert "exactly 4 replacement" in first
    assert "Proposal 2: unsupported_operator" in second
    assert all(word not in (first + second).lower() for word in ("score", "mse", "loss"))
    assert [row["status"] for row in event["candidates"]] == [
        "fitted",
        "duplicate",
        "invalid",
        "duplicate",
        "fit_failed",
        "invalid",
        "fitted",
        "fitted",
        "fitted",
        "fitted",
    ]
    retried = [row for row in event["candidates"] if row.get("source") == "retry"]
    assert [(row["replaces"], row["turn"]) for row in retried] == [(2, 1), (3, 1), (4, 1), (5, 1), (2, 2)]
    assert event["ended"] == "done"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("not json", "invalid_reply_json"),
        ("[]", "invalid_reply_count"),
        ('[{"initial":{"b":NaN}}]', "nonfinite_reply_number"),
    ],
)
def test_an_unreadable_reply_is_asked_for_again_with_its_reason(tmp_path, reply, reason):
    up = import_upstream(Path(__file__).resolve().parents[3])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = 2 + np.cos(angles)

    class Stub:
        async def submit(self, batch, prompt):
            return reply

        async def retry(self, batch, prompt, history):
            return '[{"equation":"b+a*cos(theta)","initial":{"b":2,"a":1}}]'

    run = asyncio.run(
        execute(
            up,
            Stub(),
            (angles, responses),
            (angles, responses),
            tmp_path,
            ideas=(Idea("harmonic", "peak", "harmonic", "Try a harmonic."),),
            batches=1,
            k=1,
            max_iter=0,
        )
    )
    event = run["episodes"][0]
    assert event["turns"][0]["kind"] == "unreadable" and reason in event["turns"][0]["instruction"]
    assert event["candidates"][0]["status"] == "fitted"
    assert event["candidates"][0]["source"] == "retry"


def test_offline_wave_writes_audit_and_never_exposes_scores(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
    up["hypothesis_engine"] = SimpleNamespace(
        objective=fitting.make_objective(up["hypothesis_engine"].objective, "lm-bounded", lm_iterations=8)
    )
    responses, angles = synthetic_recording()
    x, y, _, _ = prepare(responses, angles, "strict-holdout-v1")
    x, y = np.tile(np.asarray(x), (4, 1)), np.tile(np.asarray(y), (4, 1))
    seen = []
    snapshots = []
    retry_inputs = []

    def reply(n):
        return json.dumps([{"equation": f"b + a*cos({n}*theta)", "initial": {"b": 3, "a": 1}}])

    class Stub:
        interrupt = True

        async def submit(self, batch, prompt):
            if len(seen) == 2 and self.interrupt:
                self.interrupt = False
                raise InterruptedError
            seen.append((batch.count, prompt))
            snapshots.append(json.loads((tmp_path / "wave.json").read_text()))
            return reply(1 if len(seen) == 3 else len(seen))

        async def retry(self, batch, prompt, history):
            retry_inputs.append((batch.count, prompt, *history[-1]))
            return reply(3)

    stub = Stub()

    def run(**kwargs):
        return asyncio.run(
            execute(
                up,
                stub,
                (x[:10], y[:10]),
                (x[10:20], y[10:20]),
                tmp_path,
                ideas=(
                    Idea("a", "peak", "harmonic", "Use one circular harmonic.", lane="root"),
                    Idea("b", "peak", "harmonic", "Try a higher harmonic.", lane="root"),
                    Idea("c", "peak", "harmonic", "Combine the patterns.", (("a", "b"),)),
                ),
                batches=3,
                k=1,
                seed=2,
                max_iter=8,
                **kwargs,
            )
        )

    with pytest.raises(InterruptedError):
        run()
    partial = json.loads((tmp_path / "wave.json").read_text())
    assert len(partial["episodes"]) == 2 and partial["active_probe"]["batch"] == 3
    result = run(resume=True)
    assert result["episodes"][:2] == partial["episodes"]
    assert retry_inputs == [(40, seen[2][1], reply(1), result["episodes"][2]["turns"][0]["instruction"])]
    assert "b + a*cos(1*theta)" in retry_inputs[0][3]
    assert "score" not in retry_inputs[0][3].lower()
    assert [row["status"] for row in result["episodes"][2]["candidates"]] == ["duplicate", "fitted"]
    assert run(resume=True) == result
    assert [n for n, _ in seen] == [20, 20, 40]
    assert any("Use one circular harmonic." in prompt for _, prompt in seen)
    assert all("score" not in prompt.lower() for _, prompt in seen)
    assert seen[2][1].startswith("Combine the ideas in this thread:")
    assert "Do not add an unrelated mechanism" in seen[2][1]
    assert result["version"] == 4
    assert result["baseline_exploration"]["score"] > 0
    assert [row["node"] for row in snapshots[0]["graph"]] == ["double_gaussian"]
    assert [row["node"] for row in snapshots[2]["graph"]] == [
        "double_gaussian",
        result["episodes"][0]["node"],
        result["episodes"][1]["node"],
    ]
    assert snapshots[0]["graph"][0]["equation"].startswith("b + a1*exp")
    assert snapshots[2]["active_probe"]["mode"] == "combination"
    assert [row["introduced_at"] for row in result["graph"]] == [0, 1, 2, 3]
    assert [row["routes"] for row in result["graph"]] == [[], [], [], [["a", "b"]]]
    assert "active_probe" not in result
    assert result["episodes"][0]["candidates"][0]["status"] == "fitted"
    assert {row["node"] for row in result["confirmation"]} == {"a", "b", "c"}
    assert result["shortlist"]["c"]["parents"] == ["a", "b"]
    assert json.loads((tmp_path / "wave.json").read_text()) == result


def test_an_equation_that_cannot_be_fitted_is_reported_back_and_replaced(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = 2 + np.cos(angles)
    calls = []

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(
                {
                    "idea": {"title": "Harmonic", "mechanism": "harmonic", "change": "Add a cosine."},
                    "equations": [{"equation": "a*exp(60*sin(theta)) + b", "initial": {"b": 7, "a": 1}}],
                }
            )

        async def retry(self, batch, prompt, history):
            calls.append(history[-1][1])
            return '[{"equation":"b+a*sin(theta+phi)","initial":{"b":2,"a":1,"phi":0}}]'

    result = asyncio.run(
        execute(up, Stub(), (angles, responses), (angles, responses), tmp_path, batches=1, max_iter=0)
    )
    event = result["episodes"][0]
    assert len(calls) == 1
    assert event["turns"][0]["kind"] == "replace" and "fit_failed" in calls[0]
    assert [row["status"] for row in event["candidates"]] == ["fit_failed", "fitted"]
    assert "score" not in calls[0].lower()


def test_generated_idea_is_visible_without_equations_and_can_be_revisited_after_resume(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
    x = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    y = 2 + np.cos(x)
    proposal = {"equation": "b+a*cos(theta)", "initial": {"b": 2, "a": 1}}
    idea = {
        "title": "Single circular harmonic",
        "mechanism": "single-harmonic",
        "change": "Use one cosine term.",
    }

    class Stub:
        async def submit(self, batch, prompt):
            state = json.loads((tmp_path / "wave.json").read_text())
            if state["active_probe"]["batch"] == 1:
                return json.dumps({"idea": idea, "equations": []})
            if not resume:
                raise InterruptedError
            return json.dumps([proposal])

        async def retry(self, batch, prompt, history):
            return "[]"

    def run():
        return asyncio.run(
            execute(
                up,
                Stub(),
                (x, y),
                (x, y),
                tmp_path,
                batches=2,
                k=4,
                new_probability=0,
                resume=resume,
                max_iter=0,
            )
        )

    resume = False
    with pytest.raises(InterruptedError):
        run()
    paused = json.loads((tmp_path / "wave.json").read_text())
    assert [node["introduced_at"] for node in paused["graph"]] == [0, 1]
    assert paused["graph"][1]["label"] == idea["title"]
    assert paused["episodes"][0]["candidates"] == []
    orphan = {**paused["graph"][1], "node": "idea-orphan", "introduced_at": 2}
    paused["graph"].append(orphan)
    paused["idea_catalog"].append(orphan)
    (tmp_path / "wave.json").write_text(json.dumps(paused))
    resume = True
    result = run()
    assert result["episodes"][0] == paused["episodes"][0]
    assert [event["kind"] for event in result["episodes"]] == ["new", "revisit"]
    assert len(result["graph"]) == 2
    assert "idea-orphan" not in {row["node"] for row in result["idea_catalog"]}
    assert result["episodes"][1]["node"] == paused["graph"][1]["node"]
    assert result["episodes"][1]["candidates"][0]["status"] == "fitted"
    assert result["confirmation"][0]["node"] == paused["graph"][1]["node"]


def test_generated_idea_is_visible_during_equation_retry(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
    x = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    y = 2 + np.cos(x)
    snapshots = []

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(
                {
                    "idea": {
                        "title": "Circular shoulder",
                        "mechanism": "shoulder",
                        "change": "Add a cosine shoulder.",
                    },
                    "equations": [{"equation": "b + a*log(theta)", "initial": {"b": 2, "a": 1}}],
                }
            )

        async def retry(self, batch, prompt, history):
            snapshots.append(json.loads((tmp_path / "wave.json").read_text()))
            return '[{"equation":"b+a*cos(theta)","initial":{"b":2,"a":1}}]'

    result = asyncio.run(execute(up, Stub(), (x, y), (x, y), tmp_path, batches=1, max_iter=0))
    assert len(snapshots) == 1
    assert 'data-node="idea-1" data-equations="0"' in render(snapshots[0])
    assert 'data-node="idea-1"' not in render(snapshots[0], 0)
    event = result["episodes"][0]
    assert [row["status"] for row in event["candidates"]] == ["invalid", "fitted"]
    assert "unsupported_operator" in event["turns"][0]["instruction"]
    assert result["graph"][1]["label"] == "Circular shoulder"


ROOT = Path(__file__).resolve().parents[3]


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
        idea = {
            "title": f"Harmonic {step}",
            "mechanism": f"harmonic-{step}",
            "change": f"Use harmonic {step}.",
        }
        return json.dumps({"idea": idea, "equations": [equation]})

    async def retry(self, batch, prompt, history):
        self.prompts.append(history[-1][1])
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

        async def retry(self, batch, prompt, history):
            return "[]"

    result = asyncio.run(
        execute(import_upstream(ROOT), Nothing(), (x, y), (x, y), tmp_path, batches=2, k=2, max_iter=0)
    )
    assert result["status"] == "no_fitted_equations"
    assert result["shortlist"] == {}
    assert result["confirmation"] == []


@pytest.mark.parametrize("guided", [{}, {"temperature": 0.1}])
def test_an_interrupted_run_resumes_to_the_same_result_as_an_uninterrupted_one(tmp_path, guided):
    whole, _ = explore(tmp_path / "whole", seed=11, batches=5, **guided)
    with pytest.raises(InterruptedError):
        explore(tmp_path / "split", seed=11, batches=5, interrupt_at=4, **guided)
    paused = json.loads((tmp_path / "split" / "wave.json").read_text())
    assert len(paused["episodes"]) == 3
    assert paused["status"] == "exploring"
    resumed, _ = explore(tmp_path / "split", seed=11, batches=5, resume=True, **guided)
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

        async def retry(self, batch, prompt, history):
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
        "starts": 4,
        "patience": 3,
        "temperature": None,
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
    "overrides",
    [
        {"waves": 0},
        {"batches": 0},
        {"k": 0},
        {"seed": -1},
        {"new_probability": 2},
        {"patience": -1},
        {"temperature": 0},
    ],
)
def test_run_rejects_unusable_options_before_touching_the_network(tmp_path, overrides):
    with pytest.raises(SystemExit):
        run(args(tmp_path / "out", **overrides))


def test_the_number_of_starts_is_part_of_the_run_configuration(tmp_path):
    result, _ = explore(tmp_path, batches=1, starts=2)
    assert result["fit_protocol"] == "warm_multistart_v2"
    assert result["config"]["starts"] == 2
    with pytest.raises(ValueError):
        explore(tmp_path, batches=1, starts=3, resume=True)
    with pytest.raises(ValueError):
        explore(tmp_path / "none", batches=1, starts=0)


def test_a_child_that_contains_its_parent_is_never_fitted_worse_than_the_parent(tmp_path):
    ideas = (
        Idea("base", "peak", "harmonic", "Use one harmonic."),
        Idea("child", "peak", "sine", "Add a sine term.", (("base",),)),
    )
    x = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False), (10, 1))
    y = 0.3 + 2 * np.exp(-0.5 * (np.arctan2(np.sin(x - 3.5), np.cos(x - 3.5)) / 0.3) ** 2)
    parent = {
        "equation": "b + a*exp(-0.5*(wrap(theta-m)/s)**2)",
        "initial": {"b": 0.3, "a": 1.5, "m": 3.3, "s": 0.4},
    }
    child = {
        "equation": "b + a*exp(-0.5*(wrap(theta-m)/s)**2) + c*cos(theta)",
        "initial": {"b": 0.3, "a": 1.5, "m": 0.2, "s": 0.3, "c": 0.0},
    }

    class Catalog:
        async def submit(self, batch, prompt):
            return json.dumps([child if "Add a sine term" in prompt else parent])

        async def retry(self, batch, prompt, history):
            return "[]"

    result = asyncio.run(
        execute(
            import_upstream(Path(__file__).resolve().parents[3]),
            Catalog(),
            (x, y),
            (x, y),
            tmp_path,
            ideas=ideas,
            batches=4,
            k=1,
            max_iter=100,
            starts=2,
        )
    )
    fitted = {}
    for event in result["episodes"]:
        for record in event["candidates"]:
            if record.get("status") == "fitted":
                fitted.setdefault(event["node"], record["metrics"])
    assert {"base", "child"} <= set(fitted)
    assert fitted["child"]["training_mse"] <= fitted["base"]["training_mse"] + 1e-9
    assert fitted["child"]["mse"] <= fitted["base"]["mse"] + 1e-9


def test_the_model_can_check_equations_before_answering_and_only_its_answer_is_fitted(tmp_path):
    x, y = cohort()
    statement = {"equation": "y = b + a*cos(theta)", "initial": {"b": 2, "a": 1}}
    good = {"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}}
    histories = []

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps({"check": [statement, good]})

        async def retry(self, batch, prompt, history):
            histories.append(history)
            return json.dumps([good])

    result = asyncio.run(
        execute(
            import_upstream(ROOT),
            Stub(),
            (x, y),
            (x, y),
            tmp_path,
            ideas=(Idea("a", "peak", "harmonic", "Try a harmonic."),),
            batches=1,
            k=1,
            max_iter=0,
            patience=2,
        )
    )
    event = result["episodes"][0]
    report = event["turns"][0]["instruction"]
    assert event["turns"][0]["kind"] == "check"
    assert "Equation 1: cannot be used: invalid_syntax" in report
    assert "Equation 2: usable." in report
    assert [(row["status"], row["turn"]) for row in event["candidates"]] == [("fitted", 1)]
    assert len(histories) == 1 and result["config"]["patience"] == 2
    assert '{"check": [' in event["prompt"] and "2 follow-up turns" in event["prompt"]


@pytest.mark.parametrize(
    ("temperature", "reference"), [(None, "b + a*cos(2*theta)"), (0.1, "b + a*cos(theta)")]
)
def test_with_a_temperature_a_better_revisit_becomes_the_ideas_reference(tmp_path, temperature, reference):
    x, y = cohort()
    idea = {"title": "Harmonic", "mechanism": "harmonic", "change": "Use a harmonic."}

    class Stub:
        async def submit(self, batch, prompt):
            if prompt.startswith("Revisit"):
                return json.dumps([{"equation": "b + a*cos(theta)", "initial": {"b": 2, "a": 1}}])
            return json.dumps(
                {"idea": idea, "equations": [{"equation": "b + a*cos(2*theta)", "initial": {"b": 2, "a": 1}}]}
            )

        async def retry(self, batch, prompt, history):
            return "[]"

    result = asyncio.run(
        execute(
            import_upstream(ROOT),
            Stub(),
            (x, y),
            (x, y),
            tmp_path,
            batches=2,
            new_probability=0,
            max_iter=0,
            temperature=temperature,
        )
    )
    assert [e["kind"] for e in result["episodes"]] == ["new", "revisit"]
    assert result["shortlist"]["idea-1"]["equation"] == reference
    assert result["config"].get("temperature") == temperature

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.frontier_runner import (
    BASE,
    Idea,
    InvalidReply,
    _baseline,
    _generated_idea,
    _generated_prompt,
    _measure,
    _trials,
    compile_equation,
    draw,
    draw_generated,
    execute,
    validate_graph,
)
from edgar_study.frontier_viewer import render
from edgar_study.upstream import import_upstream


def test_equation_grammar_rejects_code_and_deduplicates_parameter_renames():
    one = compile_equation({"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 2}})
    two = compile_equation({"equation": "q*cos(theta) + c", "initial": {"q": 3, "c": 4}})
    assert one.key == two.key
    np.testing.assert_allclose(
        one.model(np.array([0, np.pi]), *one.estimator(None, None)), [3, -1], atol=1e-6
    )
    with pytest.raises(ValueError):
        compile_equation({"equation": "__import__('os').system('false')", "initial": {}})


def test_equation_grammar_accepts_periodic_cutoffs_and_width_drift():
    window = compile_equation(
        {
            "equation": "b + a*(0.5+0.5*cos(wrap(theta-m)))*(abs(wrap(theta-m))<w)",
            "initial": {"b": 1, "a": 2, "m": 0, "w": 1},
        }
    )
    angles = np.array([0, np.pi, 2 * np.pi])
    np.testing.assert_allclose(window.model(angles, *window.estimator(None, None)), [3, 1, 3], atol=1e-6)
    combination = compile_equation(
        {
            "equation": BASE.replace("b + ", "b + r*cos(theta - phi) + ", 1).replace(
                "(wrap(theta-m)/s)**2", "(wrap(theta-m)/(s*(1 + k*tanh(wrap(theta-m)/w))))**2"
            ),
            "initial": {"b": 1, "m": 0, "a1": 2, "a2": 1, "s": 0.5, "r": 0, "phi": 0, "k": 0, "w": 1},
        }
    )
    np.testing.assert_allclose(combination.model(angles, *combination.estimator(None, None)), [3, 2, 3])


def test_cutoff_width_has_a_gradient_at_its_edge():
    equation = compile_equation(
        {
            "equation": "b + a*(abs(wrap(theta-m))<w)",
            "initial": {"b": 1, "a": 2, "m": 0, "w": 1},
        }
    )
    assert jax.grad(lambda w: equation.model(1.0, 1.0, 2.0, 0.0, w))(1.0) > 1
    edge = compile_equation(
        {
            "equation": "b+a*(abs(wrap(theta))<w)",
            "initial": {"b": 1, "a": 2, "w": 2.9},
        }
    )
    np.testing.assert_allclose(edge.model(-2.9, 1, 2, 2.9), edge.model(2 * np.pi - 2.9, 1, 2, 2.9), atol=1e-5)


def test_cutoffs_accept_all_comparisons_and_need_real_inputs():
    for left, op, right in (
        ("wrap(theta)", "<", "0"),
        ("wrap(theta)", "<=", "0"),
        ("wrap(theta)", ">", "0"),
        ("wrap(theta)", ">=", "0"),
        ("0", "<", "wrap(theta)"),
        ("0", ">=", "wrap(theta)"),
    ):
        equation = compile_equation({"equation": f"b+a*({left}{op}{right})", "initial": {"b": 1, "a": 2}})
        assert 0 < equation.model(0.0, 1, 2) < 3
    with pytest.raises(ValueError, match="real-valued"):
        compile_equation({"equation": "b+a*((theta*(-1)**0.5)>=0)", "initial": {"b": 1, "a": 2}})


def test_undeclared_names_have_their_own_reason():
    with pytest.raises(ValueError, check=lambda exc: getattr(exc, "code", None) == "unknown_name"):
        compile_equation({"equation": "b+a*cos(w1)", "initial": {"b": 1, "a": 2, "m1": 3}})


def test_generated_idea_mechanism_defaults_to_the_title():
    choice = {"node": "idea-1", "mode": "independent_root", "parents": [], "lane": "root"}
    accepted = _generated_idea({"title": "Circular Skew", "change": "Skew each lobe."}, choice, {})
    assert accepted.mechanism == "circular-skew"


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("a*cos(theta) + 1/0", "nonfinite_expression"),
        ("a*cos(theta)*(-1)**0.5", "nonreal_expression"),
        ("a*cos(theta) + (cos(theta)*(-1)**0.5 < 1)", "nonreal_expression"),
        ("a*theta", "nonperiodic_expression"),
        ("a*cos(theta)+(0<theta<1)", "unsupported_operator"),
        ("a*cos(theta)" + "+1" * 85, "equation_too_complex"),
    ],
)
def test_invalid_equation_has_stable_reason(expression, reason):
    with pytest.raises(ValueError, check=lambda exc: getattr(exc, "code", None) == reason):
        compile_equation({"equation": expression, "initial": {"a": 1}})


def test_free_offset_starts_at_the_fitted_baseline_and_improves_on_true_offsets():
    up = import_upstream(Path(__file__).resolve().parents[2])
    up["hypothesis_engine"] = SimpleNamespace(
        objective=fitting.make_objective(up["hypothesis_engine"].objective, "lm-bounded")
    )
    equation = compile_equation(
        {
            "equation": BASE.replace("m-pi", "m-pi*g"),
            "initial": {"g": 2, "m": 0, "s": 0.35, "a2": 1, "b": 0.5, "a1": 4},
        }
    )
    angles = np.tile(np.linspace(0, 2 * np.pi, 144, endpoint=False), (2, 1))
    responses = np.asarray(equation.model(angles, 0.86, 0.6, 0.35, 2, 0.2, 3))
    baseline = _measure(up, _baseline(up), (angles, responses), 2000)
    offset = _measure(up, equation, (angles, responses), 2000, nested_offset=True)
    assert offset["initialization"] == "fitted_double_gaussian"
    assert offset["training_mse"] <= baseline["training_mse"] + 1e-7
    assert offset["mse"] < baseline["mse"]


def test_free_offset_accepts_roundoff_at_the_minimum_baseline_width():
    up = import_upstream(Path(__file__).resolve().parents[2])
    params = np.array([[5.5, 0.1, 1, 20, 0.01]], dtype=np.float32)

    def objective(model, estimator, loss, x, y, **kwargs):
        return None, None, 0.05, params

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)
    equation = compile_equation(
        {
            "equation": BASE.replace("m-pi", "m-d"),
            "initial": {"m": 0, "b": 0.1, "a1": 1, "a2": 1, "s": 0.1, "d": np.pi},
        }
    )
    center = (5.5 + np.pi) % (2 * np.pi)
    angles = np.linspace(center - 0.02, center + 0.02, 40, dtype=np.float32)[None, :]
    responses = np.asarray(_baseline(up).model(angles, *params[0]))
    offset = _measure(up, equation, (angles, responses), 2000, nested_offset=True)
    assert offset["training_mse"] < 1e-7


@pytest.mark.parametrize("mechanism", ["shape", "offset"])
def test_frontier_warm_starts_nested_gaussian_families_without_selecting_on_held_out(tmp_path, mechanism):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (10, 1))
    base = _baseline(up)
    params = (0.7, 0.2, 2.5, 1.1, 0.6)
    responses = np.array(base.model(angles, *params))
    train, _ = _trials(angles)
    responses[:, train] += 0.4 * np.cos(angles[:, train])

    def objective(model, estimator, loss, x, y, **kwargs):
        start = params if model is base.model else estimator(None, None)
        return None, None, 1.0, np.tile(np.array(start, dtype=np.float32), (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)
    initial = {"m": 3, "b": 6, "a1": 0.2, "a2": 0.1, "s": 1}
    proposals = [
        {"equation": BASE + " + cc*cos(theta) + ss*sin(theta)", "initial": {**initial, "cc": 0, "ss": 0}},
        {
            "equation": BASE.replace("(wrap(theta-m)/s)**2", "(abs(wrap(theta-m))/s)**p"),
            "initial": {**initial, "p": 1},
        },
        {
            "equation": "b + A1*exp(-0.5*(wrap(theta-mu)/s1)**2) + A2*exp(-0.5*(wrap(theta-mu)/s2)**2)"
            " + A3*exp(-0.5*(wrap(theta-p2)/s3)**2)",
            "initial": {"b": 6, "A1": 1, "A2": 1, "A3": 1, "mu": 3, "p2": 2, "s1": 1, "s2": 2, "s3": 1},
        },
        {
            "equation": BASE.replace("b + ", "b + c*cos(theta) + ", 1),
            "initial": {**initial, "c": 0},
        },
        {
            "equation": BASE.replace("b + ", "b + c*sin(theta) + ", 1),
            "initial": {**initial, "c": 0},
        },
    ]
    if mechanism == "offset":
        proposals = [
            {
                "equation": BASE.replace("m-pi", "m-pi-k*sin(q)"),
                "initial": {**initial, "k": 2, "q": 1},
                "nesting": {"double_gaussian": {**{name: name for name in initial}, "k": 1, "q": 0}},
            }
        ]

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(proposals)

    result = asyncio.run(
        execute(
            up,
            Stub(),
            (angles, responses),
            (angles, responses),
            tmp_path,
            ideas=(Idea("changes", "peak", mechanism, "Modify the double Gaussian."),),
            batches=1,
            k=len(proposals),
            max_iter=40,
        )
    )
    fitted = [row["metrics"] for row in result["episodes"][0]["candidates"]]
    assert [row["parameters"] for row in fitted] == ([7] if mechanism == "offset" else [7, 6, 9, 6, 6])
    baseline = result["baseline_exploration"]["training_mse"]
    for row in [*fitted, result["confirmation"][0]["metrics"]]:
        assert row["initialization"] == "fitted_double_gaussian"
        assert row["training_mse"] <= baseline + 1e-6
    if mechanism == "shape":
        assert fitted[0]["training_mse"] < 1e-6
        assert fitted[0]["mse"] > 0.05


def test_combination_keeps_verified_parent_fit_and_ignores_false_mapping(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False, dtype=np.float32), (10, 1))
    responses = 1 + 0.4 * np.cos(angles)

    def objective(model, estimator, loss, x, y, **kwargs):
        start = estimator(x[0], y[0])
        return None, None, 1.0, np.tile(np.array(start, dtype=np.float32), (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)
    parent = {"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 0.4}}
    extension = {
        "equation": "u + v*cos(theta) + w*sin(theta)",
        "initial": {"u": 4, "v": 2, "w": 1},
        "nesting": {"a": {"u": "b", "v": "a", "w": 0}},
    }
    false_extension = {
        "equation": "u + v*cos(2*theta) + w*sin(theta)",
        "initial": extension["initial"],
        "nesting": extension["nesting"],
    }

    class Stub:
        async def submit(self, batch, prompt):
            if prompt.startswith("Combine"):
                return json.dumps([extension, false_extension])
            return json.dumps([parent, {"equation": "b + a*cos(3*theta)", "initial": parent["initial"]}])

    result = asyncio.run(
        execute(
            up,
            Stub(),
            (angles, responses),
            (angles, responses),
            tmp_path,
            ideas=(
                Idea("a", "peak", "harmonic", "One harmonic."),
                Idea("c", "peak", "add", "Add sine.", (("a",),)),
            ),
            batches=2,
            k=2,
            max_iter=0,
        )
    )
    verified, unverified = [row["metrics"] for row in result["episodes"][1]["candidates"]]
    assert verified["initialization"] == "fitted_parent"
    assert verified["training_mse"] < 1e-7
    assert unverified["initialization"] == "estimator"
    assert unverified["training_mse"] > 1


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


def test_invalid_and_exact_duplicate_proposals_share_one_score_blind_retry(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
    base = _baseline(up)
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False, dtype=np.float32), (10, 1))
    responses = 1 + 0.4 * np.cos(angles)

    def objective(model, estimator, loss, x, y, **kwargs):
        start = (0.7, 0.2, 2.5, 1.1, 0.6) if model is base.model else estimator(x[0], y[0])
        return None, None, float("inf") if start[0] == 7 else 1.0, np.tile(start, (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)
    invalid = {"equation": "a*log(theta)", "initial": {"a": 1}}
    reply = json.dumps(
        [
            {"equation": "b + a*cos(theta)", "initial": {"b": 1, "a": 1}},
            {"equation": BASE, "initial": {"m": 0, "s": 1, "a1": 1, "b": 1, "a2": 1}},
            invalid,
            {"equation": "a*cos(theta) + b", "initial": {"a": 1, "b": 1}},
            {"equation": "b + a*sin(theta)", "initial": {"b": 7, "a": 1}},
        ]
    )

    class Stub:
        async def submit(self, batch, prompt):
            return reply

        async def retry(self, batch, prompt, previous_reply, instruction):
            return json.dumps([invalid, {"equation": "b + a*cos(2*theta)", "initial": {"b": 1, "a": 1}}])

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
    instruction = event["retry"]["instruction"]
    assert BASE in instruction
    assert "unsupported_operator" in instruction
    assert "proposal 3" in instruction.lower()
    assert "exactly 2 replacements" in instruction
    assert all(word not in instruction.lower() for word in ("score", "mse", "nonfinite fit"))
    assert [row["status"] for row in event["candidates"]] == [
        "fitted",
        "duplicate",
        "invalid",
        "duplicate",
        "fit_failed",
        "invalid",
        "fitted",
    ]
    assert [row["reason_code"] for row in event["candidates"] if row["status"] == "invalid"] == [
        "unsupported_operator",
        "unsupported_operator",
    ]
    assert [row["replaces"] for row in event["candidates"] if row.get("source") == "retry"] == [2, 3]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("not json", "invalid_reply_json"),
        ("[]", "invalid_reply_count"),
        ('[{"initial":{"b":NaN}}]', "nonfinite_reply_number"),
    ],
)
def test_unreadable_reply_can_be_replaced_once(tmp_path, reply, reason):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = 2 + np.cos(angles)

    class Stub:
        async def submit(self, batch, prompt):
            return reply

        async def retry(self, batch, prompt, previous_reply, instruction):
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
    assert event["retry"]["reason_code"] == reason
    assert event["candidates"][0]["status"] == "fitted"
    assert event["candidates"][0]["source"] == "retry"


def test_explicit_offset_map_cannot_fit_negative_amplitudes_or_baseline(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = np.full_like(angles, -2)
    params = {"m": 0.5, "b": 1.0, "a1": 1.0, "a2": 1.0, "s": 0.5}

    def objective(model, estimator, loss, x, y, **kwargs):
        return None, None, 1.0, np.tile(tuple(params.values()), (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(
                [
                    {
                        "equation": BASE.replace("m-pi", "m-pi-k*sin(q)"),
                        "initial": {**params, "k": 1, "q": 0},
                        "nesting": {"double_gaussian": {**{name: name for name in params}, "k": 1, "q": 0}},
                    }
                ]
            )

    run = asyncio.run(
        execute(
            up,
            Stub(),
            (angles, responses),
            (angles, responses),
            tmp_path,
            ideas=(Idea("offset", "peak", "offset", "Free the offset."),),
            batches=1,
            k=1,
        )
    )
    metrics = run["episodes"][0]["candidates"][0]["metrics"]
    assert 4.0 <= metrics["training_mse"] < run["baseline_exploration"]["training_mse"]
    assert metrics["mse"] >= 4.0


def test_offline_wave_writes_audit_and_never_exposes_scores(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
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
            seen.append((len(batch.images), prompt))
            snapshots.append(json.loads((tmp_path / "wave.json").read_text()))
            return reply(1 if len(seen) == 3 else len(seen))

        async def retry(self, batch, prompt, previous_reply, instruction):
            retry_inputs.append((len(batch.images), prompt, previous_reply, instruction))
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
    assert retry_inputs == [(40, seen[2][1], reply(1), result["episodes"][2]["retry"]["instruction"])]
    assert "b + a*cos(1*theta)" in retry_inputs[0][3]
    assert "score" not in retry_inputs[0][3].lower()
    assert [row["status"] for row in result["episodes"][2]["candidates"]] == ["duplicate", "fitted"]
    assert run(resume=True) == result
    assert [n for n, _ in seen] == [20, 20, 40]
    assert any("Use one circular harmonic." in prompt for _, prompt in seen)
    assert all("score" not in prompt.lower() for _, prompt in seen)
    assert seen[2][1].startswith("Combine the ideas in this thread:")
    assert "Do not add an unrelated mechanism" in seen[2][1]
    assert result["version"] == 3
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
    distinct = _generated_idea(
        {"title": "Shared separation offset", "mechanism": "offset", "change": "Move both lobes."},
        choice,
        existing,
    )
    assert distinct.label == "Shared separation offset"
    with pytest.raises(InvalidReply, match="Existing ideas on this route: Directional peak shift"):
        _generated_idea(
            {"title": "directional peak shift", "mechanism": "width", "change": "Move both lobes."},
            choice,
            existing,
        )


def test_idea_description_allows_six_hundred_characters():
    choice = {"node": "idea-2", "mode": "single_idea", "parents": [], "lane": "root"}
    accepted = _generated_idea(
        {"title": "Long change", "mechanism": "mixture", "change": "a" * 600}, choice, {}
    )
    assert accepted.text == "a" * 600
    with pytest.raises(InvalidReply, match="at most 600"):
        _generated_idea({"title": "Too long", "mechanism": "mixture", "change": "a" * 601}, choice, {})


def test_revisit_prompt_lists_tried_equations_without_scores():
    node = Idea(
        "idea-1", "revisit", "width", "Split the widths.", (("double_gaussian",),), "frontier", "Widths"
    )
    prompt = _generated_prompt(
        {"node": "idea-1", "kind": "revisit", "mode": "revisit", "parents": ["double_gaussian"]},
        {"idea-1": node},
        {},
        4,
        ("b + a*cos(theta)",),
    )
    assert "do not repeat" in prompt
    assert "b + a*cos(theta)" in prompt
    assert "score" not in prompt.lower()


def test_six_parameter_standalone_equation_falls_back_when_offset_embedding_fails(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = 2 + np.cos(angles) + 0.3 * np.cos(2 * angles)
    proposal = {
        "equation": "b + a*cos(theta) + c*cos(2*theta) + d*cos(3*theta) + e*sin(theta) + f*sin(2*theta)",
        "initial": {"b": 2, "a": 1, "c": 0.3, "d": 0, "e": 0, "f": 0},
    }

    def objective(model, estimator, loss, x, y, **kwargs):
        return None, None, 1.0, np.tile(np.array(estimator(x[0], y[0])), (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(
                {
                    "idea": {"title": "Harmonic family", "mechanism": "offset", "change": "Use harmonics."},
                    "equations": [proposal],
                }
            )

        async def retry(self, batch, prompt, previous_reply, instruction):
            return json.dumps([proposal])

    result = asyncio.run(
        execute(up, Stub(), (angles, responses), (angles, responses), tmp_path, batches=1, max_iter=0)
    )
    metrics = result["episodes"][0]["candidates"][0]["metrics"]
    assert metrics["initialization"] == "estimator"
    assert metrics["training_mse"] == pytest.approx(0.0, abs=1e-6)


def test_unusable_probe_gets_one_retry_even_when_equations_only_fail_to_fit(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
    angles = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), (10, 1))
    responses = 2 + np.cos(angles)
    calls = []

    def objective(model, estimator, loss, x, y, **kwargs):
        start = estimator(x[0], y[0])
        if start[0] < 0:
            return None, None, float("inf"), np.tile(np.array(start), (len(x), 1))
        return None, None, 1.0, np.tile(np.array(start), (len(x), 1))

    up["hypothesis_engine"] = SimpleNamespace(objective=objective)

    class Stub:
        async def submit(self, batch, prompt):
            return json.dumps(
                {
                    "idea": {"title": "Harmonic", "mechanism": "harmonic", "change": "Add a cosine."},
                    "equations": [{"equation": "b + a*cos(theta)", "initial": {"b": -1, "a": 1}}],
                }
            )

        async def retry(self, batch, prompt, previous_reply, instruction):
            calls.append(instruction)
            return '[{"equation":"b+a*sin(theta+phi)","initial":{"b":2,"a":1,"phi":0}}]'

    result = asyncio.run(
        execute(up, Stub(), (angles, responses), (angles, responses), tmp_path, batches=1, max_iter=0)
    )
    event = result["episodes"][0]
    assert len(calls) == 1
    assert event["retry"]["reason_code"] == "no_usable_equation"
    assert [row["status"] for row in event["candidates"]] == ["fit_failed", "fitted"]
    assert "score" not in calls[0].lower()


def test_tried_independent_root_can_be_probed_again():
    root = Idea("root", "peaks", "harmonic", "Use circular harmonics.", lane="root")
    choice = draw((root,), {}, {"root": 1}, np.random.default_rng(8))
    assert choice["node"] == "root"


def test_generated_idea_is_visible_without_equations_and_can_be_revisited_after_resume(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
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

        async def retry(self, batch, prompt, previous_reply, instruction):
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


def test_generated_idea_is_visible_during_equation_retry(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[2])
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

        async def retry(self, batch, prompt, previous_reply, instruction):
            snapshots.append(json.loads((tmp_path / "wave.json").read_text()))
            return '[{"equation":"b+a*cos(theta)","initial":{"b":2,"a":1}}]'

    result = asyncio.run(execute(up, Stub(), (x, y), (x, y), tmp_path, batches=1, max_iter=0))
    assert len(snapshots) == 1
    assert 'data-node="idea-1" data-equations="0"' in render(snapshots[0])
    assert 'data-node="idea-1"' not in render(snapshots[0], 0)
    event = result["episodes"][0]
    assert [row["status"] for row in event["candidates"]] == ["invalid", "fitted"]
    assert event["retry"]["targets"][0]["reason_code"] == "unsupported_operator"
    assert result["graph"][1]["label"] == "Circular shoulder"

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.frontier_runner import (
    BASE,
    Idea,
    _baseline,
    _measure,
    _trials,
    compile_equation,
    draw,
    execute,
    validate_graph,
)
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


def test_frontier_warm_starts_nested_gaussian_families_without_selecting_on_held_out(tmp_path):
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
            ideas=(Idea("changes", "peak", "shape", "Modify the double Gaussian."),),
            batches=1,
            k=3,
            max_iter=40,
        )
    )
    fitted = [row["metrics"] for row in result["episodes"][0]["candidates"]]
    assert [row["parameters"] for row in fitted] == [7, 6, 9]
    baseline = result["baseline_exploration"]["training_mse"]
    for row in [*fitted, result["confirmation"][0]["metrics"]]:
        assert row["initialization"] == "fitted_double_gaussian"
        assert row["training_mse"] <= baseline + 1e-6
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
            return json.dumps([parent, {"equation": "b", "initial": {"b": 1}}])

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


def test_tried_independent_root_can_be_probed_again():
    root = Idea("root", "peaks", "harmonic", "Use circular harmonics.", lane="root")
    choice = draw((root,), {}, {"root": 1}, np.random.default_rng(8))
    assert choice["node"] == "root"

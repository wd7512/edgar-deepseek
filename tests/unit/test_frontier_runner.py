import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.frontier_runner import Idea, compile_equation, draw, execute, validate_graph
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

    class Stub:
        async def submit(self, batch, prompt):
            seen.append((len(batch.images), prompt))
            snapshots.append(json.loads((tmp_path / "wave.json").read_text()))
            return json.dumps([{"equation": f"b + a*cos({len(seen)}*theta)", "initial": {"b": 3, "a": 1}}])

    result = asyncio.run(
        execute(
            up,
            Stub(),
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
        )
    )
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

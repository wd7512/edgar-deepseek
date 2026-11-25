import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgar_study import fitting
from edgar_study.frontier.fits import double_gaussian_parent, measure, split_trials
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.runner import execute
from edgar_study.frontier.seed import BASE
from edgar_study.upstream import import_upstream


def test_free_offset_starts_at_the_fitted_baseline_and_improves_on_true_offsets():
    up = import_upstream(Path(__file__).resolve().parents[3])
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
    baseline = measure(up, double_gaussian_parent(up), (angles, responses), 2000)
    offset = measure(up, equation, (angles, responses), 2000, nested_offset=True)
    assert offset["initialization"] == "fitted_double_gaussian"
    assert offset["training_mse"] <= baseline["training_mse"] + 1e-7
    assert offset["mse"] < baseline["mse"]


def test_free_offset_accepts_roundoff_at_the_minimum_baseline_width():
    up = import_upstream(Path(__file__).resolve().parents[3])
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
    responses = np.asarray(double_gaussian_parent(up).model(angles, *params[0]))
    offset = measure(up, equation, (angles, responses), 2000, nested_offset=True)
    assert offset["training_mse"] < 1e-7


@pytest.mark.parametrize("mechanism", ["shape", "offset"])
def test_frontier_warm_starts_nested_gaussian_families_without_selecting_on_held_out(tmp_path, mechanism):
    up = import_upstream(Path(__file__).resolve().parents[3])
    angles = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False, dtype=np.float32), (10, 1))
    base = double_gaussian_parent(up)
    params = (0.7, 0.2, 2.5, 1.1, 0.6)
    responses = np.array(base.model(angles, *params))
    train, _ = split_trials(angles)
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
    up = import_upstream(Path(__file__).resolve().parents[3])
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


def test_explicit_offset_map_cannot_fit_negative_amplitudes_or_baseline(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
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


def test_six_parameter_standalone_equation_falls_back_when_offset_embedding_fails(tmp_path):
    up = import_upstream(Path(__file__).resolve().parents[3])
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

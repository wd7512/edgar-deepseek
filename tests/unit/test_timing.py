import asyncio
import json
from types import SimpleNamespace

import pytest

from edgar_study.timing import Timings, install, summarise


def test_summary_counts_overlapping_calls_once_in_wall_time():
    records = [
        {"stage": "proposal", "start": 10.0, "end": 14.0},
        {"stage": "proposal", "start": 11.0, "end": 15.0},
        {"stage": "fit", "start": 15.0, "end": 17.0},
        {"stage": "fit", "start": 18.0, "end": 19.0},
    ]

    summary = summarise(records, started=10.0, finished=20.0)

    assert summary["total_s"] == 10.0
    assert summary["stages"]["proposal"] == {"calls": 2, "busy_s": 8.0, "wall_s": 5.0}
    assert summary["stages"]["fit"] == {"calls": 2, "busy_s": 3.0, "wall_s": 3.0}
    assert summary["other_s"] == 2.0
    assert list(summary["stages"]) == ["proposal", "fit"]


def test_wrapped_functions_return_their_results_and_append_records(tmp_path):
    path = tmp_path / "timings.jsonl"
    timings = Timings(path)

    def add(a, b):
        return a + b

    async def double(a):
        await asyncio.sleep(0)
        return 2 * a

    assert timings.wrap(add, "fit")(1, 2) == 3
    assert asyncio.run(timings.wrap(double, "proposal")(4)) == 8

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["stage"] for line in lines] == ["fit", "proposal"]
    assert all(line["end"] >= line["start"] for line in lines)
    assert [record["stage"] for record in timings.records] == ["fit", "proposal"]


def test_a_failing_call_is_still_recorded():
    timings = Timings()

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        timings.wrap(fail, "fit")()

    assert [record["stage"] for record in timings.records] == ["fit"]


def test_install_times_upstream_stages_and_separates_the_final_refit():
    def objective(model, estimator, loss_func, x, y, max_iter=1_000):
        return max_iter

    async def generate(*args, **kwargs):
        return "code"

    def plot(*args, **kwargs):
        return None

    engine = SimpleNamespace(
        objective=objective,
        generate_new_neuron_model=generate,
        translate_to_jax=generate,
        generate_new_parameter_estimator=generate,
        compute_evaluation_matrix=plot,
        diagnostic=SimpleNamespace(
            plot_model_fits=plot, plot_single_model_fit=plot, plot_train_vs_test_loss=plot
        ),
        genetic_helpers=SimpleNamespace(
            perform_island_deduplication=plot,
            perform_population_pruning=plot,
            perform_probabilistic_migration=plot,
            remove_duplicates=plot,
        ),
    )
    timings = Timings()
    install(engine, timings)

    assert engine.objective(None, None, None, None, None) == 1_000
    assert engine.objective(None, None, None, None, None, max_iter=2_000) == 2_000
    asyncio.run(engine.generate_new_neuron_model())
    asyncio.run(engine.translate_to_jax())
    asyncio.run(engine.generate_new_parameter_estimator())
    engine.compute_evaluation_matrix()
    engine.diagnostic.plot_model_fits()
    engine.genetic_helpers.remove_duplicates()

    assert [record["stage"] for record in timings.records] == [
        "fit",
        "final_refit",
        "proposal",
        "translation",
        "estimator",
        "evaluation_matrix",
        "plots",
        "population",
    ]

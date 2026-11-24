"""Offline checks for the NEAT search loop and its strict holdout."""

from pathlib import Path

import numpy as np

from edgar_study import curves, fitting, neat_search
from edgar_study.upstream import import_upstream


def test_a_short_search_scores_every_genome_and_keeps_the_best():
    upstream = import_upstream(Path(".").resolve())
    engine = upstream["hypothesis_engine"]
    fitting.install(engine, "lm-bounded")
    x, y = curves.synthetic_cells(n_cells=6, n_trials=64)

    result = neat_search.search(
        engine.objective,
        upstream["loss_functions"].quadratic_loss,
        x,
        y,
        population_size=4,
        generations=2,
        seed=0,
    )

    assert result.attempts == 8
    assert 0 < result.evaluations <= result.attempts
    assert len(result.records) == result.attempts
    assert all(np.isfinite(record["score"]) for record in result.records)
    assert all(record["n_params"] >= 4 for record in result.records)
    best = min(record["score"] for record in result.records)
    assert result.best["score"] == best
    assert [row["generation"] for row in result.generations] == [0, 1]


def test_top_records_skips_failures_and_ranks_the_rest():
    records = [
        {"score": None, "code_key": "failed"},
        {"score": 2.0, "code_key": "second"},
        {"score": 1.0, "code_key": "first"},
        {"score": 1.0, "code_key": "first"},
    ]
    chosen = neat_search.top_records(records, top=5)
    assert [record["code_key"] for record in chosen] == ["first", "second"]


def test_search_is_reproducible_for_a_seed():
    upstream = import_upstream(Path(".").resolve())
    engine = upstream["hypothesis_engine"]
    fitting.install(engine, "lm-bounded")
    x, y = curves.synthetic_cells(n_cells=4, n_trials=48)
    kwargs = {
        "population_size": 2,
        "generations": 1,
        "seed": 7,
    }
    first = neat_search.search(engine.objective, upstream["loss_functions"].quadratic_loss, x, y, **kwargs)
    second = neat_search.search(engine.objective, upstream["loss_functions"].quadratic_loss, x, y, **kwargs)
    assert [record["code_key"] for record in first.records] == [
        record["code_key"] for record in second.records
    ]
    assert [record["score"] for record in first.records] == [record["score"] for record in second.records]


def test_holdout_selects_on_selection_cells_and_scores_the_winner_on_locked_cells():
    upstream = import_upstream(Path(".").resolve())
    engine = upstream["hypothesis_engine"]
    fitting.install(engine, "lm-bounded")
    x, y = curves.synthetic_cells(n_cells=6, n_trials=64)
    groups = {
        "discovery": np.arange(4),
        "selection": np.array([4]),
        "locked_test": np.array([5]),
    }

    result = neat_search.search(
        engine.objective,
        upstream["loss_functions"].quadratic_loss,
        x[groups["discovery"]],
        y[groups["discovery"]],
        population_size=4,
        generations=1,
        seed=0,
    )
    rescore = neat_search.select_and_score(upstream, result, x, y, groups, top=2)

    assert len(rescore["candidates"]) == 4  # two seeds and the top two genomes
    assert all(candidate["selection"] is not None for candidate in rescore["candidates"])
    winner = rescore["winner"]
    assert winner is not None
    learned = [candidate for candidate in rescore["candidates"] if candidate["generation"] >= 0]
    assert winner["selection"]["score"] == min(candidate["selection"]["score"] for candidate in learned)
    assert winner["locked_test"]["score"] > 0
    assert rescore["delta_locked_test_vs_double_gaussian"] is not None
    assert rescore["candidates"][0]["id"].startswith("seed_")

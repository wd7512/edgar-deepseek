"""Fit-speed benchmark: times every paper curve under every fitter through upstream's `objective`.

Run on its own with `uv run pytest tests/speed/curve_fitting.py -o log_cli=true --log-cli-level=INFO`.
Timings are logged, never asserted against a budget, because they depend on the host.
"""

import logging

import numpy as np
import pytest

from edgar_study import curves
from edgar_study.upstream import PENALTY

FITTERS = ("upstream", "adam", "lm")


@pytest.fixture(scope="module")
def rows():
    rows = curves.benchmark(fitters=FITTERS, n_cells=16, n_trials=144, max_iter=100)
    logging.getLogger(__name__).info("\n%s", curves.benchmark_table(rows))
    return rows


def test_every_curve_is_timed_under_every_fitter(rows):
    assert [(r["program"], r["fitter"]) for r in rows] == [
        (c.name, f) for f in FITTERS for c in curves.CURVES
    ]
    for row in rows:
        assert row["params"] == next(c for c in curves.CURVES if c.name == row["program"]).n_params
        assert row["first_seconds"] > 0 and row["repeat_seconds"] > 0
        assert np.isfinite(row["train_loss"]) and np.isfinite(row["test_loss"])
        assert row["score"] == pytest.approx(row["test_loss"] + PENALTY * row["params"], rel=1e-5)


def test_winner_beats_the_baseline_on_data_from_the_winner():
    rows = curves.benchmark(
        fitters=("lm",),
        n_cells=24,
        n_trials=200,
        curves=[c for c in curves.CURVES if c.name in (curves.WINNER, curves.BASELINE)],
    )
    loss = {r["program"]: r["test_loss"] for r in rows}
    assert loss[curves.WINNER] < loss[curves.BASELINE]


def test_table_lists_each_row_with_both_timings(rows):
    table = curves.benchmark_table(rows)
    lines = table.splitlines()
    assert len(lines) == 2 + len(rows)
    assert "First call (s)" in lines[0] and "Repeat (s)" in lines[0]
    assert any(line.startswith("| winner | H27 | 11 | lm |") for line in lines)

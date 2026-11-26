"""The prompt A/B test: contexts from a finished run, and the paired summary of the two arms."""

import json

import pytest

from edgar_study.frontier.ab import contexts, summarise


def _wave():
    proposal = {"equation": "b + a*cos(theta-m)", "initial": {"a": 1.0, "b": 0.0, "m": 0.0}}
    return {
        "idea_catalog": [
            {
                "node": "idea-1",
                "branch": "single_idea",
                "mechanism": "shape",
                "text": "Cosine.",
                "routes": [["double_gaussian"]],
                "lane": "frontier",
                "label": "Cos",
            }
        ],
        "episodes": [
            {
                "node": "idea-1",
                "candidates": [
                    {
                        "proposal": proposal,
                        "status": "fitted",
                        "equation": proposal["equation"],
                        "metrics": {"score": 28.0},
                    },
                    {"proposal": proposal, "status": "fit_failed", "equation": "b + a/(m - 1)"},
                ],
            }
        ],
    }


def test_contexts_cover_every_mode_on_the_baseline_and_the_strongest_idea():
    found = contexts(_wave())
    modes = {(c["choice"]["kind"], c["choice"]["mode"], tuple(c["choice"]["parents"])) for c in found}
    assert ("new", "single_idea", ("double_gaussian",)) in modes
    assert ("new", "single_idea", ("idea-1",)) in modes
    assert ("new", "independent_root", ()) in modes
    assert ("revisit", "revisit", ("double_gaussian",)) in modes
    revisit = next(c for c in found if c["choice"]["kind"] == "revisit")
    assert revisit["tried"] == [["b + a*cos(theta-m)", "fitted"], ["b + a/(m - 1)", "fit_failed"]]
    json.dumps(found)


def test_the_summary_counts_outcomes_per_arm_and_pairs_gains_by_context():
    rows = [
        {"arm": "v1", "context": 0, "status": "invalid", "gain": None},
        {"arm": "v1", "context": 0, "status": "fitted", "gain": -0.1},
        {"arm": "v2", "context": 0, "status": "fitted", "gain": 0.2},
        {"arm": "v2", "context": 0, "status": "fit_failed", "gain": None},
        {"arm": "v1", "context": 1, "status": "fitted", "gain": 0.0},
        {"arm": "v2", "context": 1, "status": "fitted", "gain": 0.1},
    ]
    summary = summarise(rows)
    assert summary["v1"]["equations"] == 3 and summary["v1"]["invalid"] == pytest.approx(1 / 3)
    assert summary["v2"]["fit_failed"] == pytest.approx(1 / 3)
    assert summary["v2"]["best_gain"] == pytest.approx(0.2)
    assert summary["paired"]["contexts"] == 2
    assert summary["paired"]["v2_better"] == 2

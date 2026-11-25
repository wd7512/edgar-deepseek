from edgar_study.frontier_viewer import render


def test_replay_shows_alternative_routes_without_node_status_labels():
    wave = {
        "status": "complete",
        "graph": [
            {"node": "a", "text": "A", "routes": []},
            {"node": "b", "text": "B", "routes": []},
            {"node": "c", "text": "C", "routes": [["a", "b"], ["a"]]},
        ],
        "episodes": [
            {"node": "a", "candidates": [{"status": "fitted"}]},
            {"node": "b", "candidates": [{"status": "fitted"}]},
        ],
        "shortlist": {"a": {"equation": "a < b"}},
        "claims": {"a": "supported_fit"},
    }
    earlier = render(wave, 1)
    latest = render(wave)
    assert latest.count('data-child="c"') == 2
    assert ">AND<" in latest
    for status in ("Baseline", "Ready", "Attempted", "Fitted", "Blocked", "Confirmed"):
        assert f">{status}<" not in latest
    assert "data-state=" not in latest
    assert "a &lt; b" in latest
    assert 'data-live="true"' in latest
    assert 'data-live="false"' in earlier
    assert 'data-idea-detail="a"' in latest


def test_missing_snapshot_is_a_live_waiting_page():
    view = render({})
    assert "Waiting for the first batch" in view
    assert view.count('data-node="') == 1
    assert 'data-node="double_gaussian" data-equations="1"' in view
    assert "Starting equation" in view
    assert "b + a1*exp" in view
    assert 'data-live="true"' in render({}, 0)


def test_inspector_shows_seed_and_candidate_scores_in_the_correct_batch():
    root = {
        "node": "double_gaussian",
        "text": "Double Gaussian",
        "routes": [],
        "equation": "exp(-theta**2)",
        "introduced_at": 0,
    }
    child = {
        "node": "child",
        "text": "Shift the second peak.",
        "routes": [["double_gaussian"]],
        "introduced_at": 1,
    }
    wave = {
        "version": 3,
        "graph": [root, child],
        "idea_catalog": [child],
        "baseline_exploration": {"score": 123.45},
        "episodes": [
            {
                "node": "child",
                "candidates": [{"equation": "cos(theta)", "status": "fitted", "metrics": {"score": 75.25}}],
            }
        ],
    }
    assert 'data-node="child"' not in render(wave, 0)
    view = render(wave)
    assert 'data-child="child"' in view
    assert 'data-parent="double_gaussian"' in view
    assert "0 probe(s)" in view
    assert "Fit score: 123.45" in view and "Fit score: 75.25" in view
    assert "Fit score: 75.25" not in render(wave, 0)


def test_node_metrics_come_from_one_best_scored_equation_and_follow_replay():
    def event(score, mse, parameters):
        return {
            "node": "a",
            "candidates": [
                {
                    "equation": f"{score}*theta",
                    "status": "fitted",
                    "metrics": {"score": score, "mse": mse, "parameters": parameters},
                }
            ],
        }

    wave = {
        "graph": [{"node": "a", "text": "A", "routes": []}],
        "episodes": [event(4, 0.3, 5), event(3, 0.4, 2)],
    }
    assert "Score 4 · MSE 0.3 · 5 params" in render(wave, 1)
    assert "Score 3 · MSE 0.4 · 2 params" in render(wave)


def test_probe_failure_is_visible_even_without_a_node():
    wave = {
        "version": 2,
        "graph": [],
        "idea_catalog": [{"node": "tip", "text": "Free the main-peak exponent.", "routes": []}],
        "episodes": [{"node": "tip", "error": "No proposal text returned."}],
        "replay_note": "Recorded proposals, not a new run.",
    }
    view = render(wave)
    assert "No proposal text returned." in view
    assert "Recorded proposals, not a new run." in view
    assert 'data-node="tip"' not in view


def test_idea_with_only_invalid_equations_shows_the_reason_and_retry():
    wave = {
        "graph": [
            {
                "node": "idea-1",
                "text": "Use peaks.",
                "label": "Windowed peaks",
                "routes": [],
                "introduced_at": 1,
            }
        ],
        "episodes": [
            {
                "node": "idea-1",
                "mode": "independent_root",
                "candidates": [
                    {
                        "status": "invalid",
                        "reason_code": "unsupported_operator",
                        "error": "Unknown operation.",
                    },
                    {
                        "status": "invalid",
                        "source": "retry",
                        "replaces": 1,
                        "reason_code": "invalid_syntax",
                        "error": "Not an expression.",
                        "proposal": {"equation": "<bad>"},
                    },
                ],
            }
        ],
    }
    view = render(wave)
    assert 'data-node="idea-1" data-equations="0"' in view
    assert "Windowed peaks" in view
    assert "unsupported_operator" in view
    assert "Replacement for proposal 1" in view
    assert "&lt;bad&gt;" in view


def test_growing_replay_shows_ideas_and_all_their_probes_only_when_available():
    wave = {
        "version": 2,
        "graph": [
            {
                "node": "root",
                "label": "Circular harmonics",
                "text": "Use circular harmonics.",
                "routes": [],
                "introduced_at": 0,
            },
            {
                "node": "child",
                "label": "Shift the second peak",
                "text": "Move the second peak off 180°.",
                "routes": [["root"]],
                "introduced_at": 2,
            },
        ],
        "episodes": [
            {"node": "root", "candidates": [{"equation": "cos(theta)", "status": "fitted"}]},
            {"node": "child", "candidates": [{"equation": "sin(theta)", "status": "fit_failed"}]},
            {"node": "child", "candidates": [{"equation": "cos(2*theta)", "status": "fitted"}]},
        ],
    }
    start = render(wave, 0)
    assert 'data-node="root" data-equations="0"' in start
    assert 'data-node="child"' not in render(wave, 1)
    middle = render(wave, 2)
    assert 'data-node="child" data-equations="1"' in middle
    assert "No fit score" in middle
    assert "Shift the second peak" in middle
    assert "sin(theta)" in middle
    assert "cos(2*theta)" not in middle
    latest = render(wave)
    assert 'data-node="child" data-equations="2"' in latest
    assert "Batch 2" in latest and "Batch 3" in latest
    assert "cos(2*theta)" in latest
    assert 'marker-end="url(#arrow)"' in latest

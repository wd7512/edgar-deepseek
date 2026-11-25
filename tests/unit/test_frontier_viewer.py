from edgar_study.frontier_viewer import render


def test_replay_distinguishes_an_unlocked_route_from_a_confirmed_claim():
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
    assert 'data-node="a" data-state="fitted"' in earlier
    assert 'data-node="b" data-state="ready"' in earlier
    assert 'data-node="c" data-state="ready"' in earlier
    assert earlier.count('data-route-state="ready"') == 1
    assert latest.count('data-route-state="ready"') == 2
    assert 'data-node="a" data-state="confirmed"' in latest
    assert "a &lt; b" in latest
    assert 'data-live="true"' in latest
    assert 'data-live="false"' in earlier
    assert 'data-idea-detail="a"' in latest


def test_missing_snapshot_is_a_live_waiting_page():
    assert "Waiting for the first batch" in render({})


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
    assert 'data-node="root" data-state="ready" data-equations="0"' in start
    assert 'data-node="child"' not in render(wave, 1)
    middle = render(wave, 2)
    assert 'data-node="child" data-state="attempted" data-equations="1"' in middle
    assert "Shift the second peak" in middle
    assert "sin(theta)" in middle
    assert "cos(2*theta)" not in middle
    latest = render(wave)
    assert 'data-node="child" data-state="fitted" data-equations="2"' in latest
    assert "Batch 2" in latest and "Batch 3" in latest
    assert "cos(2*theta)" in latest
    assert 'marker-end="url(#arrow)"' in latest

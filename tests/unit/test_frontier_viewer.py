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


def test_missing_snapshot_is_a_live_waiting_page():
    assert "Waiting for the first batch" in render({})

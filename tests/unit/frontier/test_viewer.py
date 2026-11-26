from edgar_study.frontier.viewer import render


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


def _homes(view):
    import re

    return {
        node: (float(x), float(y))
        for node, x, y in re.findall(r'data-node="([^"]+)"[^>]*data-home-x="([^"]+)" data-home-y="([^"]+)"', view)
    }


def _fitted(equation, mse, parameters):
    return {
        "equation": equation,
        "status": "fitted",
        "metrics": {"score": mse, "mse": mse, "parameters": parameters},
    }


def test_initial_layout_never_overlaps_however_many_ideas_there_are():
    names = [f"idea-{i}" for i in range(60)] + ["width", "drift", "double_gaussian"]
    wave = {"graph": [{"node": n, "text": n, "routes": [], "introduced_at": 0} for n in names]}
    homes = _homes(render(wave))
    assert len(homes) == len(names)
    points = list(homes.values())
    for i, (x, y) in enumerate(points):
        for ox, oy in points[i + 1 :]:
            assert abs(x - ox) >= 192 or abs(y - oy) >= 68
        assert x == x and y >= 34


def test_graph_can_be_zoomed_and_reset():
    view = render({"graph": [{"node": "a", "text": "A", "routes": []}]})
    assert '<g class="viewport"' in view
    for action in ("in", "out", "fit"):
        assert f'data-zoom="{action}"' in view
    assert 'data-node="a"' in view.split('<g class="viewport"')[1]


def test_lowest_mse_for_each_parameter_count_is_highlighted_gently():
    wave = {
        "graph": [{"node": n, "text": n, "routes": [], "introduced_at": 0} for n in "abcd"],
        "episodes": [
            {"node": "a", "candidates": [_fitted("x", 0.5, 3), _fitted("y", 0.1, 4)]},
            {"node": "b", "candidates": [_fitted("x", 0.2, 3)]},
            {"node": "c", "candidates": [_fitted("x", 0.9, 5)]},
            {"node": "d", "candidates": [_fitted("x", 0.9, 4), {"status": "fit_failed", "equation": "z"}]},
        ],
    }
    view = render(wave)

    def tag(node):
        return view.split(f'data-node="{node}"')[1].split(">")[0]

    assert 'data-lowest-mse="4"' in tag("a")
    assert 'data-lowest-mse="3"' in tag("b")
    assert 'data-lowest-mse="5"' in tag("c")
    assert "data-lowest-mse" not in tag("d")
    assert "Lowest MSE for 3 parameters" in view
    assert 'data-lowest-mse="3"' not in tag("a")


def test_highlight_follows_replay_and_ignores_ties_with_nothing():
    wave = {
        "graph": [{"node": n, "text": n, "routes": [], "introduced_at": 0} for n in "ab"],
        "episodes": [
            {"node": "a", "candidates": [_fitted("x", 0.5, 3)]},
            {"node": "b", "candidates": [_fitted("x", 0.2, 3)]},
        ],
    }
    first, both = render(wave, 1), render(wave)
    assert 'data-lowest-mse="3"' in first.split('data-node="a"')[1].split(">")[0]
    assert 'data-lowest-mse="3"' not in both.split('data-node="a"')[1].split(">")[0]
    assert 'data-lowest-mse="3"' in both.split('data-node="b"')[1].split(">")[0]


def test_layered_layout_puts_children_below_parents_and_orders_to_avoid_crossings():
    wave = {
        "graph": [
            {"node": "a1", "text": "a1", "routes": [], "introduced_at": 0},
            {"node": "a2", "text": "a2", "routes": [], "introduced_at": 0},
            {"node": "b1", "text": "b1", "routes": [["a2"]], "introduced_at": 0},
            {"node": "b2", "text": "b2", "routes": [["a1"]], "introduced_at": 0},
            {"node": "c", "text": "c", "routes": [["b1", "b2"]], "introduced_at": 0},
        ]
    }
    homes = _homes(render(wave))
    for parent, child in (("a1", "b2"), ("a2", "b1"), ("b1", "c"), ("b2", "c")):
        assert homes[parent][1] < homes[child][1]
    assert (homes["a1"][0] < homes["a2"][0]) == (homes["b2"][0] < homes["b1"][0])


def test_layout_of_a_ten_thousand_node_tree_is_fast_and_overlap_free():
    import time

    n = 10000
    graph = [
        {"node": f"n{i}", "text": "x", "routes": [[f"n{(i - 1) // 2}"]] if i else [], "introduced_at": 0}
        for i in range(n)
    ]
    from edgar_study.frontier.viewer import _layout

    start = time.perf_counter()
    positions, _ = _layout({g["node"]: g for g in graph})
    assert time.perf_counter() - start < 3
    cells = {(round(x / 192), round(y / 68)) for x, y in positions.values()}
    assert len(positions) == n and len(cells) == n


def _big_wave(n):
    graph = [
        {"node": f"n{i}", "text": f"idea {i}", "routes": [[f"n{(i - 1) // 2}"]] if i else [], "introduced_at": 0}
        for i in range(n)
    ]
    episodes = [{"node": f"n{i}", "candidates": [_fitted("x", 0.1 + i / 1000, 3 + i % 4)]} for i in range(n)]
    return {"graph": graph, "episodes": episodes}


def test_large_graphs_use_a_canvas_with_data_and_load_details_on_demand():
    import json
    import re

    wave = _big_wave(600)
    view = render(wave)
    assert "<canvas" in view and 'class="node' not in view
    assert "data-idea-detail" not in view
    blob = re.search(r'<script type="application/json" class="graph-data">(.*?)</script>', view, re.S).group(1)
    data = json.loads(blob)
    assert len(data["nodes"]) == 600
    assert len(data["links"]) == 599
    assert any(node["lowest"] for node in data["nodes"])
    detail = render(wave, inspect="n5")
    assert detail.startswith("<aside") and 'data-idea-detail="n5"' in detail
    assert render(wave, inspect="missing") == ""
    assert "<canvas" not in render(_big_wave(20))


def test_server_answers_cheap_snapshot_and_detail_requests(tmp_path):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from edgar_study.frontier.viewer import make_handler

    path = tmp_path / "wave.json"
    path.write_text(json.dumps(_big_wave(300)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(path, "layout"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/"
        snapshot = urlopen(base + "?snapshot=1").read().decode()
        assert snapshot and "<" not in snapshot
        assert f'data-snapshot="{snapshot}"' in urlopen(base).read().decode()
        assert 'data-idea-detail="n3"' in urlopen(base + "?inspect=n3").read().decode()
    finally:
        server.shutdown()
        server.server_close()

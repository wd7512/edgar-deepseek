import json

from edgar_study import run_index


def write_run(folder, run_id, hours, analysis=True):
    run = folder / "runs" / run_id
    run.mkdir(parents=True)
    data = {
        "run": {"id": run_id, "hours": hours, "model": "free-model"},
        "locked": {"winner": {"score": 28.5}, "double_gaussian": {"score": 28.8}},
        "pmin": {"median": 1.1},
    }
    (run / "data.json").write_text(json.dumps(data))
    if analysis:
        payload = {
            "data": {"eligible": {"log": {"alpha": 2.7}}},
            "models": {"EDGAR winner 1_0_0": {"pca": {"log": {"alpha": 3.8}}}},
            "wave": {"r2_gain_over_ai_model": {"median": 0.002}},
        }
        (run / "paper_analysis.json").write_text(json.dumps(payload))


def test_index_lists_runs_oldest_first_with_their_headline_numbers(tmp_path):
    write_run(tmp_path, "20261124-000000-bbbbbb", 2.0, analysis=False)
    write_run(tmp_path, "20261123-000000-aaaaaa", 1.5)

    index = run_index.build(tmp_path)

    assert [r["id"] for r in index["runs"]] == ["20261123-000000-aaaaaa", "20261124-000000-bbbbbb"]
    first, second = index["runs"]
    assert first["folder"] == "runs/20261123-000000-aaaaaa"
    assert first["summary"] == {
        "gain": 0.3000000000000007,
        "pmin": 1.1,
        "hours": 1.5,
        "alpha_data": 2.7,
        "alpha_winner": 3.8,
        "wave_gain": 0.002,
    }
    assert second["summary"]["alpha_data"] is None
    assert json.loads((tmp_path / "runs.json").read_text()) == index


def test_folders_without_run_data_are_skipped(tmp_path):
    (tmp_path / "runs" / "empty").mkdir(parents=True)

    assert run_index.build(tmp_path)["runs"] == []

import json

from edgar_study import run_index


def write_run(folder, run_id):
    run = folder / "runs" / run_id
    run.mkdir(parents=True)
    (run / "data.json").write_text(json.dumps({"run": {"id": run_id, "model": "free-model"}}))


def test_index_lists_runs_oldest_first_with_their_folder_and_label(tmp_path):
    write_run(tmp_path, "20261124-000000-bbbbbb")
    write_run(tmp_path, "20261123-000000-aaaaaa")

    index = run_index.build(tmp_path)

    assert index["runs"] == [
        {
            "id": "20261123-000000-aaaaaa",
            "folder": "runs/20261123-000000-aaaaaa",
            "label": "20261123-000000-aaaaaa · free-model",
        },
        {
            "id": "20261124-000000-bbbbbb",
            "folder": "runs/20261124-000000-bbbbbb",
            "label": "20261124-000000-bbbbbb · free-model",
        },
    ]
    assert json.loads((tmp_path / "runs.json").read_text()) == index


def test_folders_without_run_data_are_skipped(tmp_path):
    (tmp_path / "runs" / "empty").mkdir(parents=True)

    assert run_index.build(tmp_path)["runs"] == []

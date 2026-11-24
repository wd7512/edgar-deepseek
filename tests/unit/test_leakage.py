import pandas as pd
import pytest

from edgar_study.upstream import searched_programs

COLUMNS = [
    "iteration_number",
    "birth_island",
    "batch_index",
    "train_loss",
    "test_loss",
    "program_code_string",
]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def test_pool_comes_from_the_islands_not_the_test_loss_deduplicated_combined_database(tmp_path):
    final = tmp_path / "program_databases" / "11-23" / "17-08-37"
    write(final / "combined" / "programs_db.csv", [[1, 0, 0, 28.5, 28.0, "a"]])
    write(final / "island_0" / "programs_db.csv", [[1, 0, 0, 28.5, 28.0, "a"], [2, 0, 1, 28.4, 29.0, "b"]])
    write(
        final / "island_1" / "programs_db.csv", [[1, 0, 0, 28.5, 28.0, "a"], [-1, 1, 0, 30.0, 30.0, "seed"]]
    )

    pool = searched_programs(tmp_path)

    assert list(pool["program_code_string"]) == ["b", "a"]
    assert "test_loss" not in pool.columns


def test_pool_needs_a_finished_run(tmp_path):
    with pytest.raises(SystemExit, match="island"):
        searched_programs(tmp_path)


def test_image_reader_names_only_the_shape_features_upstream_prompts_already_list():
    from edgar_study.engine import IMAGE_READER_PROMPT

    for hint in ("cusp", "power", "exponent", "generalized", "kink"):
        assert hint not in IMAGE_READER_PROMPT.lower()

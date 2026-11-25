import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from edgar_study.upstream import UPSTREAM_DIR, UPSTREAM_MODULES, code_key, import_upstream, searched_programs

ROOT = Path(__file__).resolve().parents[2]
PROGRAM_COLUMNS = [
    "iteration_number",
    "birth_island",
    "batch_index",
    "train_loss",
    "test_loss",
    "program_code_string",
    "mean_loss",
    "params",
    "initial_params",
]


def test_code_key_is_stable_and_distinguishes_programs():
    assert code_key("a") == code_key("a")
    assert code_key("a") != code_key("b")


def test_upstream_modules_live_in_a_flat_folder_under_src():
    folder = ROOT / UPSTREAM_DIR
    assert folder.parent == ROOT / "src"
    assert sorted(p.stem for p in folder.glob("*.py")) == sorted(UPSTREAM_MODULES)
    assert not (folder / "__init__.py").exists()
    assert not any((ROOT / f"{name}.py").exists() for name in UPSTREAM_MODULES)


def test_import_upstream_loads_modules_from_the_upstream_folder():
    upstream = import_upstream(ROOT)
    for name in ("hypothesis_engine", "loss_functions", "seed_programs", "utils"):
        assert Path(upstream[name].__file__).resolve() == ROOT / UPSTREAM_DIR / f"{name}.py"


def test_import_upstream_rejects_a_modified_module(tmp_path):
    subprocess.run(["git", "clone", "-q", str(ROOT), str(tmp_path / "c")], check=True)
    checkout = tmp_path / "c"
    shutil.rmtree(checkout / UPSTREAM_DIR, ignore_errors=True)
    shutil.copytree(ROOT / UPSTREAM_DIR, checkout / UPSTREAM_DIR)
    target = checkout / UPSTREAM_DIR / "utils.py"
    target.write_text(target.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="utils.py"):
        import_upstream(checkout)


def _write_programs(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PROGRAM_COLUMNS).to_csv(path, index=False)


def test_pool_comes_from_the_islands_not_the_test_loss_deduplicated_combined_database(tmp_path):
    final = tmp_path / "program_databases" / "11-23" / "17-08-37"
    _write_programs(final / "combined" / "programs_db.csv", [[1, 0, 0, 28.5, 28.0, "a", 28.0, "[]", "[]"]])
    _write_programs(
        final / "island_0" / "programs_db.csv",
        [[1, 0, 0, 28.5, 28.0, "a", 28.0, "[]", "[]"], [2, 0, 1, 28.4, 29.0, "b", 29.0, "[]", "[]"]],
    )
    _write_programs(
        final / "island_1" / "programs_db.csv",
        [[1, 0, 0, 28.5, 28.0, "a", 28.0, "[]", "[]"], [-1, 1, 0, 30.0, 30.0, "seed", 30.0, "[]", "[]"]],
    )

    pool = searched_programs(tmp_path)

    assert list(pool["program_code_string"]) == ["b", "a"]
    for column in ("test_loss", "mean_loss", "params", "initial_params"):
        assert column not in pool.columns


def test_pool_needs_a_finished_run(tmp_path):
    with pytest.raises(SystemExit, match="island"):
        searched_programs(tmp_path)

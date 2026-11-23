import shutil
import subprocess
from pathlib import Path

import pytest

from edgar_study.upstream import UPSTREAM_DIR, UPSTREAM_MODULES, code_key, import_upstream

ROOT = Path(__file__).resolve().parents[2]


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

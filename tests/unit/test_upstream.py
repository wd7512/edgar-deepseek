import shutil
import subprocess
from pathlib import Path

import pytest

from edgar_study.upstream import UPSTREAM_MODULES, code_key, import_upstream

ROOT = Path(__file__).resolve().parents[2]


def test_code_key_is_stable_and_distinguishes_programs():
    assert code_key("a") == code_key("a")
    assert code_key("a") != code_key("b")


def test_import_upstream_rejects_a_modified_module(tmp_path):
    subprocess.run(["git", "clone", "-q", str(ROOT), str(tmp_path / "c")], check=True)
    checkout = tmp_path / "c"
    for name in UPSTREAM_MODULES:
        shutil.copy(ROOT / f"{name}.py", checkout / f"{name}.py")
    (checkout / "utils.py").write_text((checkout / "utils.py").read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="utils.py"):
        import_upstream(checkout)

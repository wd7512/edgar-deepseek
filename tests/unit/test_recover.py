import json

import numpy as np
import pytest

from edgar_study.recover import recover
from edgar_study.upstream import combined_file, searched_programs

DASHES = "-" * 50
MODEL = "import numpy as np\n\ndef neuron_model(theta, a=1.0):\n\n    return a * np.cos(theta)"
JAX = "import jax.numpy as jnp\n\ndef neuron_model(theta, a=1.0):\n    return a * jnp.cos(theta)"
ESTIMATOR = (
    "import numpy as np\n\ndef parameter_estimator(theta, spike_counts):\n\n    return np.array([1.0])"
)


def block(ident, model=MODEL, fitted=True):
    lines = [f"id={ident}", "lm-bounded ran 12 iterations; 3 of 3 cells stopped before the cap"]
    lines += ["Prompt: \nWrite a model.\nLoss: 30.00\n", "Loss: 28.75\n"]
    lines += [
        f"Neuron Model: \n{model}\n",
        f"Neuron Model (JAX): \n{JAX}\n",
        f"Parameter Estimator: \n{ESTIMATOR}\n",
    ]
    if fitted:
        lines.append("Optimized Parameters for 10 cells:\n   a\n0  1.0\n")
    return lines + [DASHES]


def census_row(g, i, b, loss):
    return [g, i, b, "m", loss, 1.0, None, None, np.zeros((1, 2)), 1]


def make_run(tmp_path):
    final = tmp_path / "program_databases" / "11-24" / "00-50-38"
    log = ["Generating 48 new programs..."]
    log += block("0,0,0")
    log += ["id=0,0,1", "Skipping island 0, batch 1 due to LLM generation failure.", DASHES]
    log += block("0,1,0", model=MODEL.replace("cos", "sin"), fitted=False)
    log += block("1,1,1", model=MODEL.replace("cos", "tan"))
    (final / "iteration_updates" / "iteration_1").mkdir(parents=True)
    (final / "hypothesis_engine.log").write_text("\n".join(log) + "\n")
    census = [census_row(-1, -1, 0, 30.0), census_row(0, 0, 0, 28.5), census_row(0, 1, 0, 28.2)]
    (final / "iteration_updates" / "iteration_0").mkdir()
    np.save(final / "iteration_updates" / "iteration_0" / "census.npy", np.array(census[:2], dtype=object))
    np.save(
        final / "iteration_updates" / "iteration_1" / "census.npy",
        np.array(census + [census_row(1, 1, 1, 28.9)], dtype=object),
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed: BudgetExhausted"}))
    return final


def test_recover_rebuilds_island_databases_from_the_log_and_latest_census(tmp_path):
    make_run(tmp_path)

    recover(tmp_path)

    pool = searched_programs(tmp_path)
    assert [(r.iteration_number, r.birth_island, r.batch_index) for r in pool.itertuples()] == [
        (0, 1, 0),
        (0, 0, 0),
        (1, 1, 1),
    ]
    assert list(pool["train_loss"]) == [28.2, 28.5, 28.9]
    assert pool["program_code_string"][1] == MODEL
    assert pool["parameter_estimator_code_string"][1] == ESTIMATOR
    assert combined_file(tmp_path, "census.npy") is not None
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["recovered"]["generations_completed"] == 2
    assert manifest["status"] == "failed: BudgetExhausted"


def test_recover_refuses_a_run_that_finished(tmp_path):
    final = make_run(tmp_path)
    (final / "combined").mkdir()
    (final / "combined" / "programs_db.csv").write_text("x\n")

    with pytest.raises(SystemExit, match="already"):
        recover(tmp_path)


def test_recover_keeps_the_logged_translation_for_rescore(tmp_path):
    make_run(tmp_path)

    recover(tmp_path)

    pool = searched_programs(tmp_path)
    assert pool["jax_code_string"][1] == JAX


def test_recover_can_be_rerun_after_an_interrupted_attempt(tmp_path):
    final = make_run(tmp_path)
    (final / "island_0").mkdir()

    recover(tmp_path)

    assert len(searched_programs(tmp_path)) == 3

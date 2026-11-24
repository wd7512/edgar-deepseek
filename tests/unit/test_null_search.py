import argparse
import asyncio
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgar_study import null_search, symbolic
from edgar_study.upstream import code_key, import_upstream, searched_programs


def recording(path, n_cells=40):
    rng = np.random.default_rng(5)
    angles = np.tile(np.linspace(0, 2 * np.pi, 36, endpoint=False), 4)
    centers = rng.uniform(0, 2 * np.pi, n_cells)[:, None]
    near = np.abs(np.angle(np.exp(1j * (angles - centers))))
    far = np.abs(np.angle(np.exp(1j * (angles - centers - np.pi))))
    response = 0.01 + np.exp(-0.5 * (near / 0.25) ** 1.2) + 0.5 * np.exp(-0.5 * (far / 0.25) ** 1.2)
    response *= 1 + rng.normal(0, 0.05, response.shape)
    np.save(path, {"sresp": response, "istim": angles}, allow_pickle=True)
    return path


def arguments(tmp_path, **overrides):
    values = {
        "dataset": recording(tmp_path / "recording.npy"),
        "upstream": Path(".").resolve(),
        "results": tmp_path / "results",
        "fitter": "lm-bounded",
        "seed": 0,
        "iterations": 2,
        "islands": 2,
        "batch_size": 2,
        "time_limit": 60.0,
        "cell_fraction": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_search_writes_a_run_folder_the_rescore_tools_read(tmp_path):
    run_dir = null_search.run(arguments(tmp_path))

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["models"]["proposer"] == "GP (no LLM)"
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 0
    assert manifest["generations_completed"] == 2
    assert not manifest["stopped_early"]

    programs = searched_programs(run_dir)
    learned = programs[programs["iteration_number"] >= 0]
    assert len(learned) > 0
    assert np.isfinite(learned["train_loss"]).all()

    translations = {}
    for line in (run_dir / "translations.jsonl").read_text().splitlines():
        record = json.loads(line)
        translations[record["numpy_sha256"]] = record["jax_code"]
    for code in learned["program_code_string"]:
        assert "import jax.numpy as jnp" in translations[code_key(code)]


def test_time_limit_stops_before_the_first_generation(tmp_path):
    run_dir = null_search.run(arguments(tmp_path, time_limit=0.0))

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["generations_completed"] == 0
    assert manifest["stopped_early"]


def island(upstream):
    seeds = upstream["seed_programs"]
    rows = []
    for which in (1, 2):
        rows.append(
            {
                "program_code_string": null_search.seed_code(seeds, which),
                "iteration_number": -1,
                "birth_island": -1,
                "batch_index": which - 1,
                "train_loss": 30.0 - which,
            }
        )
    return pd.DataFrame(rows)


def propose(seed):
    upstream = import_upstream(Path(".").resolve())
    proposer = null_search.Proposer(upstream["seed_programs"], seed)
    np.random.seed(seed)
    found = []
    for mode in ("explore", "exploit"):
        for _ in range(5):
            code, description, parents = asyncio.run(
                proposer.neuron_model(island(upstream), "GP", None, None, None, mode=mode)
            )
            found.append((code, description, parents))
    return proposer, found


def test_proposals_are_reproducible_for_a_seed_and_differ_between_seeds():
    _, first = propose(1)
    _, again = propose(1)
    _, other = propose(2)

    assert first == again
    assert [c for c, _, _ in first] != [c for c, _, _ in other]


def test_proposals_come_with_their_jax_program_and_a_generic_estimator():
    proposer, found = propose(3)
    for code, description, _ in found:
        tree = symbolic.parse(description)
        namespace = {}
        exec(code, namespace)
        signature = inspect.signature(namespace["neuron_model"])
        assert len(symbolic.parameters(tree)) == len(signature.parameters) - 1
        jax_code, model = asyncio.run(proposer.translate(code, None, "GP"))
        assert model is not None
        assert "jnp" in jax_code
        source, estimator = asyncio.run(proposer.estimator(None, code, "GP", None, None, None))
        assert estimator is not None
        assert "lstsq" not in source

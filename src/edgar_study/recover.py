"""Rebuild the program databases of a run that stopped early, such as one that hit the spend cap.

Upstream writes its island databases only when the search ends. Until then it keeps a census of
every scored program after each generation (ids and discovery losses, no code) and logs each
program's code under its id in `hypothesis_engine.log`. `recover` joins the two and writes one
`island_<k>/programs_db.csv` per birth island plus `combined/census.npy`, so `rescore` and the
page scripts run unchanged. Every scored program is kept, including ones a finished run would
have dropped from its islands.
"""

import json
import logging
import re
import shutil
from pathlib import Path

DASHES = "-" * 50
SECTIONS = {
    "program_code_string": ("Neuron Model: \n", "\n\nNeuron Model (JAX): \n"),
    "jax_code_string": ("Neuron Model (JAX): \n", "\n\nParameter Estimator: \n"),
}


def parse_log(text):
    """Map each logged program id `(generation, island, batch)` to its NumPy, JAX and estimator code."""
    programs = {}
    for match in re.finditer(r"^id=(-?\d+),(\d+),(\d+)\n(.*?)^" + DASHES + "$", text, re.M | re.S):
        body = match.group(4)
        start = body.rfind("\nNeuron Model: \n")
        if start < 0:
            continue
        body = body[start + 1 :]
        entry = {}
        for name, (head, tail) in SECTIONS.items():
            entry[name] = body[body.index(head) + len(head) : body.index(tail)]
        estimator = body[body.index("Parameter Estimator: \n") + len("Parameter Estimator: \n") :]
        end = estimator.find("\n\nOptimized Parameters for ")
        entry["parameter_estimator_code_string"] = estimator[:end] if end >= 0 else estimator.rstrip("\n")
        programs[tuple(int(part) for part in match.group(1, 2, 3))] = entry
    return programs


def recover(run_dir):
    """Write island databases and a combined census for an unfinished run; return the number of programs."""
    import numpy as np
    import pandas as pd

    run_dir = Path(run_dir)
    found = sorted(run_dir.glob("program_databases/*/*"))
    if not found:
        raise SystemExit(f"No program_databases under {run_dir}.")
    final = found[-1]
    if any(final.glob("island_*/programs_db.csv")):
        raise SystemExit(f"{final} already has island databases; nothing to recover.")
    censuses = sorted(
        final.glob("iteration_updates/iteration_*/census.npy"),
        key=lambda path: int(path.parent.name.split("_")[1]),
    )
    if not censuses:
        raise SystemExit(f"No per-generation census under {final}.")
    census = np.load(censuses[-1], allow_pickle=True)
    programs = parse_log((final / "hypothesis_engine.log").read_text())

    rows = []
    for row in census:
        key = (int(row[0]), int(row[1]), int(row[2]))
        if key[0] < 0 or not np.isfinite(float(row[4])):
            continue
        if key not in programs:
            logging.warning("Program %s is in the census but not in the log; skipped", key)
            continue
        code = programs[key]
        rows.append(
            {
                "program_code_string": code["program_code_string"],
                "parameter_estimator_code_string": code["parameter_estimator_code_string"],
                "iteration_number": key[0],
                "birth_island": key[1],
                "batch_index": key[2],
                "train_loss": float(row[4]),
                "test_loss": float("nan"),
                "llm_name": row[3],
            }
        )
    frame = pd.DataFrame(rows)
    for island, group in frame.groupby("birth_island"):
        (final / f"island_{island}").mkdir()
        group.to_csv(final / f"island_{island}" / "programs_db.csv", index=False)
    (final / "combined").mkdir(exist_ok=True)
    shutil.copy(censuses[-1], final / "combined" / "census.npy")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recovered"] = {
        "census": str(censuses[-1].relative_to(run_dir)),
        "generations_completed": int(max(int(row[0]) for row in census)) + 1,
        "programs": len(frame),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return len(frame)

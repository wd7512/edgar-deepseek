"""Build the `runs.json` index that the results page reads, one entry per run folder."""

import json
import sys
from pathlib import Path


def build(folder):
    """Index every `runs/<id>/data.json` under `folder`, oldest run id first, and write `runs.json`."""
    folder = Path(folder)
    runs = []
    for run in sorted((folder / "runs").iterdir()):
        path = run / "data.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        runs.append(
            {
                "id": run.name,
                "folder": f"runs/{run.name}",
                "label": f"{run.name} · {data['run'].get('model', 'unknown model')}",
            }
        )
    index = {"runs": runs}
    (folder / "runs.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


if __name__ == "__main__":
    print(f"Indexed {len(build(sys.argv[1])['runs'])} runs")

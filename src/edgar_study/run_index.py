"""Build the `runs.json` index that the results page reads, one entry per run folder."""

import json
import sys
from pathlib import Path


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def summarise(data, analysis):
    """Headline numbers for the page's across-runs strip; analyses a run lacks are None."""
    locked = data["locked"]
    winner_alpha = None
    if analysis:
        winner = next(v for k, v in analysis["models"].items() if k.startswith("EDGAR winner"))
        winner_alpha = winner["pca"]["log"]["alpha"]
    return {
        "gain": locked["double_gaussian"]["score"] - locked["winner"]["score"],
        "pmin": data["pmin"]["median"],
        "hours": data["run"]["hours"],
        "alpha_data": analysis["data"]["eligible"]["log"]["alpha"] if analysis else None,
        "alpha_winner": winner_alpha,
        "wave_gain": analysis["wave"]["r2_gain_over_ai_model"]["median"]
        if analysis and "wave" in analysis
        else None,
    }


def build(folder):
    """Index every `runs/<id>/data.json` under `folder`, oldest run id first, and write `runs.json`."""
    folder = Path(folder)
    runs = []
    for run in sorted((folder / "runs").iterdir()):
        data = _read(run / "data.json")
        if data is None:
            continue
        runs.append(
            {
                "id": run.name,
                "folder": f"runs/{run.name}",
                "label": f"{run.name} · {data['run'].get('model', 'unknown model')}",
                "summary": summarise(data, _read(run / "paper_analysis.json")),
            }
        )
    index = {"runs": runs}
    (folder / "runs.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


if __name__ == "__main__":
    print(f"Indexed {len(build(sys.argv[1])['runs'])} runs")

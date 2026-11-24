"""Build data.json for the run's results page: the search, the held-out scores, peakiness and example cells.

uv run python scripts/paper_analysis/page_data.py --run results/runs/<id> --dataset data/<file>.npy --out <dir>/data.json
"""

import argparse
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import load_dataset, prepare
from edgar_study.exponents import amplitude_pair, clip_bounds, exponent_pair, fitted_exponents
from edgar_study.upstream import combined_file, import_upstream, score_program, searched_programs

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
RUN, OUT = args.run, args.out
up = import_upstream(ROOT)
fitting.install(up["hypothesis_engine"], "lm-bounded")
utils, seeds = up["utils"], up["seed_programs"]
res = json.loads((RUN / "rescore.json").read_text())
rep = json.loads((RUN / "report.json").read_text())
man = json.loads((RUN / "manifest.json").read_text())

db = searched_programs(RUN)
census = np.load(combined_file(RUN, "census.npy"), allow_pickle=True)
programs = [
    {"g": int(r[0]), "i": int(r[1]), "b": int(r[2]), "loss": round(float(r[4]), 4)}
    for r in census
    if r[0] >= 0 and np.isfinite(float(r[4]))
]

resp, ang, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
x, y, groups, indices = prepare(resp, ang, "strict-holdout-v1")
held = indices["score_trial_columns"]
cells = groups["locked_test"]
w = res["winner"]
wm = utils.str_to_func(w["jax_code"], "neuron_model")
row = db[
    db.apply(lambda r: f"{r.iteration_number}_{r.birth_island}_{r.batch_index}" == w["id"], axis=1)
].iloc[0]
we = utils.str_to_func(row.parameter_estimator_code_string, "parameter_estimator")
ws, _, _, wp = score_program(up, wm, we, x[cells], y[cells], 2000)
ds, _, _, dp = score_program(
    up, seeds.neuron_model_2_jax, seeds.parameter_estimator_2, x[cells], y[cells], 2000
)
wp, dp = np.asarray(wp), np.asarray(dp)
names = [n for n in inspect.signature(wm).parameters if n != "theta"]
exponents = fitted_exponents(names, wp, w["jax_code"])
if exponents is None:
    raise SystemExit(
        f"Winner {w['id']} has no pair of peak exponents; the page's peakiness figure needs one."
    )
pc1, pc2 = exponents
low, high = clip_bounds(w["jax_code"], exponent_pair(names)[0])
low, high = (low, high) if np.isfinite(low) and np.isfinite(high) else (0.0, 4.0)
amps = amplitude_pair(names)
a1, a2 = (wp[:, names.index(a)] for a in amps) if amps else (None, None)
pmin = np.minimum(pc1, pc2)

grid = np.linspace(0, 2 * np.pi, 181)
pred_w = np.asarray(jax.vmap(lambda p: wm(jnp.asarray(grid), *p))(jnp.asarray(wp)))
pred_d = np.asarray(jax.vmap(lambda p: seeds.neuron_model_2_jax(jnp.asarray(grid), *p))(jnp.asarray(dp)))
mse_w = np.array(
    [
        np.mean((np.asarray(wm(jnp.asarray(x[c][held]), *wp[k])) - y[c][held]) ** 2)
        for k, c in enumerate(cells)
    ]
)
mse_d = np.array(
    [
        np.mean((np.asarray(seeds.neuron_model_2_jax(jnp.asarray(x[c][held]), *dp[k])) - y[c][held]) ** 2)
        for k, c in enumerate(cells)
    ]
)
gain = mse_d - mse_w
order = np.argsort(-gain)
pick = [int(order[0]), int(order[3]), int(order[len(order) // 4]), int(order[len(order) // 2])]
examples = []
for k in pick:
    c = cells[k]
    xa, ya = np.asarray(x[c]), np.asarray(y[c])
    bins = np.round(xa / (2 * np.pi) * 12).astype(int) % 12
    means = [[float(b * 30), float(ya[bins == b].mean())] for b in range(12) if np.any(bins == b)]
    examples.append(
        {
            "cell": int(c),
            "p1": float(pc1[k]),
            "p2": float(pc2[k]),
            "points": [
                [round(float(a) * 180 / np.pi, 1), round(float(v), 3)] for a, v in zip(xa, ya, strict=True)
            ],
            "means": means,
            "winner": [round(float(v), 3) for v in pred_w[k]],
            "double": [round(float(v), 3) for v in pred_d[k]],
            "mse_w": float(mse_w[k]),
            "mse_d": float(mse_d[k]),
        }
    )

bins = np.arange(low, high + 0.05, 0.1)
hist, edges = np.histogram(pmin, bins=bins)
gens = rep["per_generation"]
first_p = rep["first_generation_with_fitted_exponent"]
pdom = np.where(a1 >= a2, pc1, pc2) if amps else None
hist_dom = np.histogram(pdom, bins=bins)[0].tolist() if amps else None
data = {
    "run": {
        "id": man["run_id"],
        "hours": rep["wall_hours"],
        "proposals": rep["proposals_attempted"],
        "scored": rep["programs_scored"],
        "sha": man["repo_sha"][:7],
        "model": man["models"]["proposer"],
        "stopped_early": man.get("recovered"),
        "iterations": man["iterations"],
        "list_cost_usd": rep.get("list_cost_usd", 0.0),
        "timings": man["timings"],
    },
    "seeds": rep["seed_losses"],
    "generations": gens,
    "programs": programs,
    "exp_programs": rep["scored_programs_with_fitted_exponent"],
    "first_exponent_gen": first_p,
    "locked": {
        k: {
            "score": res[k]["locked_test"]["score"],
            "mse": res[k]["locked_test"]["mse"],
            "params": res[k]["locked_test"]["parameters"],
            "sel": res[k]["selection"]["score"],
        }
        for k in ("winner", "double_gaussian", "single_gaussian")
    },
    "candidates": [
        {
            "id": c["id"],
            "g": c["iteration"],
            "disc": c.get("discovery_loss"),
            "sel": c["selection"]["score"] if c["selection"] else None,
            "params": c["selection"]["parameters"] if c["selection"] else None,
        }
        for c in res["candidates"]
    ],
    "winner_code": w["code"].split('"""')[0] + "".join(w["code"].split('"""')[2:])
    if w["code"].count('"""') >= 2
    else w["code"],
    "pmin": {
        "median": float(np.median(pmin)),
        "q1": float(np.percentile(pmin, 25)),
        "q3": float(np.percentile(pmin, 75)),
        "below2": float(np.mean(pmin < 2)),
        "at_floor": float(np.mean(pmin <= low + 0.001)),
        "floor": low,
        "ceiling": high,
        "hist": hist.tolist(),
        "edges": [round(e, 2) for e in edges.tolist()],
        "second_lobe_share": float(np.mean(a2 > 0.1)) if amps else None,
        "n": int(len(cells)),
        "dom_hist": hist_dom,
        "dom_median": float(np.median(pdom)) if amps else None,
        "dom_below2": float(np.mean(pdom < 2)) if amps else None,
    },
    "per_cell_gain": {
        "trials": "held-out half",
        "median": float(np.median(gain)),
        "share_better": float(np.mean(gain > 0)),
        "mse_w": float(mse_w.mean()),
        "mse_d": float(mse_d.mean()),
    },
    "examples": examples,
    "winner_id": w["id"],
    "cell_counts": res["cell_counts"],
    "grid_deg": [round(float(g) * 180 / np.pi, 1) for g in grid],
    "refit_scores": {"winner": float(ws), "double": float(ds)},
}
OUT.write_text(json.dumps(data))
print(json.dumps({k: data[k] for k in ("per_cell_gain", "exp_programs", "first_exponent_gen")}, indent=1))
print(len(programs), data["pmin"]["dom_median"], data["pmin"]["dom_below2"], [e["cell"] for e in examples])

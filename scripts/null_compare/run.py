"""Is the LLM needed? Compare LLM-run winners with no-LLM winners, neuron by neuron, on the locked-test cells.

Each run's winner (picked by `edgar rescore` on the selection cells) and the double Gaussian are refitted
on the locked cells exactly as rescore scores them. For every LLM winner against every null winner, a
paired bootstrap over cells gives the difference in locked-test score and its 95% interval. The null
hypothesis (the LLM adds nothing) is rejected for a pairing when the interval lies below zero.

    uv run python scripts/null_compare/run.py --dataset data/....npy \
        --llm results/runs/<a> results/runs/<b> --null results/runs/<c> --out results/null_test.json
"""

import argparse
import json
import logging
from pathlib import Path

from edgar_study import fitting
from edgar_study.datasets import load_dataset, prepare
from edgar_study.null_test import cell_losses, paired_bootstrap
from edgar_study.upstream import PENALTY, import_upstream, score_program, searched_programs

ROOT = Path(__file__).resolve().parents[2]
MAX_ITER = 2_000
log = logging.getLogger(__name__)


def winner(run_dir, utils):
    best = json.loads((run_dir / "rescore.json").read_text())["winner"]
    frame = searched_programs(run_dir)
    ids = (
        frame.iteration_number.astype(str)
        + "_"
        + frame.birth_island.astype(str)
        + "_"
        + frame.batch_index.astype(str)
    )
    estimator = frame.loc[ids == best["id"], "parameter_estimator_code_string"].iloc[0]
    return (
        best["id"],
        utils.str_to_func(best["jax_code"], "neuron_model"),
        utils.str_to_func(estimator, "parameter_estimator"),
    )


def summary(run_dir, arm):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    generations = report.get("per_generation", [])
    return {
        "run": manifest["run_id"],
        "arm": arm,
        "proposer": manifest["models"]["proposer"],
        "seed": manifest.get("seed"),
        "wall_hours": report.get("wall_hours"),
        "proposals": report.get("proposals_attempted"),
        "scored": report.get("programs_scored"),
        "generation_0_best": generations[0]["best"] if generations else None,
        "best_so_far": [g["best_so_far"] for g in generations],
        "best_discovery_loss": report.get("best_discovery_loss"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--llm", type=Path, nargs="+", required=True)
    parser.add_argument("--null", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fitter", default="lm-bounded")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    upstream = import_upstream(ROOT)
    fitting.install(upstream["hypothesis_engine"], args.fitter)
    utils, seeds = upstream["utils"], upstream["seed_programs"]
    response, angles, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    rows = groups["locked_test"]
    xl, yl = x[rows], y[rows]

    def locked(model, estimator):
        score, _, count, params = score_program(upstream, model, estimator, xl, yl, MAX_ITER)
        return score, count, cell_losses(model, params, xl, yl, indices["score_trial_columns"])

    dg_score, dg_count, dg_cells = locked(seeds.neuron_model_2_jax, seeds.parameter_estimator_2)
    runs, cells = [], {}
    for arm, dirs in (("llm", args.llm), ("null", args.null)):
        for run_dir in dirs:
            run_dir = run_dir.resolve()
            program_id, model, estimator = winner(run_dir, utils)
            score, count, losses = locked(model, estimator)
            entry = summary(run_dir, arm) | {
                "winner": program_id,
                "parameters": count,
                "locked_score": score,
                "gain_over_double_gaussian": dg_score - score,
            }
            runs.append(entry)
            cells[entry["run"]] = (losses, count)
            log.info("%s %s locked %.4f (%d params)", arm, entry["run"], score, count)

    pairs = []
    for llm in (r for r in runs if r["arm"] == "llm"):
        for null in (r for r in runs if r["arm"] == "null"):
            (a, ca), (b, cb) = cells[llm["run"]], cells[null["run"]]
            test = paired_bootstrap(a, b, penalty_delta=PENALTY * (ca - cb))
            pairs.append(
                {
                    "llm": llm["run"],
                    "null": null["run"],
                    **test,
                    "null_share_of_llm_gain": null["gain_over_double_gaussian"]
                    / llm["gain_over_double_gaussian"]
                    if llm["gain_over_double_gaussian"] > 0
                    else None,
                    "reject_null": test["high"] < 0,
                }
            )
    vs_dg = {
        r["run"]: paired_bootstrap(
            cells[r["run"]][0], dg_cells, penalty_delta=PENALTY * (cells[r["run"]][1] - dg_count)
        )
        for r in runs
    }
    worst_llm = max((r for r in runs if r["arm"] == "llm"), key=lambda r: r["locked_score"])
    best_null = min((r for r in runs if r["arm"] == "null"), key=lambda r: r["locked_score"])
    headline = next(p for p in pairs if p["llm"] == worst_llm["run"] and p["null"] == best_null["run"])
    result = {
        "locked_cells": int(len(rows)),
        "double_gaussian": {"locked_score": dg_score, "parameters": dg_count},
        "runs": runs,
        "vs_double_gaussian": vs_dg,
        "pairs": pairs,
        "headline": headline,
        "rejected_in_every_pairing": all(p["reject_null"] for p in pairs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {"headline": headline, "rejected_in_every_pairing": result["rejected_in_every_pairing"]}, indent=1
        )
    )


if __name__ == "__main__":
    main()

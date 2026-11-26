"""A/B test of proposal prompts on fixed contexts, scored against each context's parent.

Each arm sees the same contexts (mode, parents and the same ten neurons) and is sampled the same number of
times. Only the first reply is used, with no follow-up turns, so the prompt alone is measured. Every equation
is fitted on a fixed subset of exploration neurons, and its gain is its parent's score minus its own on
those same neurons (positive means better than the parent).
"""

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from edgar_study import fitting
from edgar_study.datasets import discovery_partition, load_dataset, prepare
from edgar_study.frontier.fits import double_gaussian_parent
from edgar_study.frontier.grammar import InvalidEquation, compile_equation
from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.images import (
    IMAGES_PER_NEURON,
    NEURONS,
    FrontierImages,
    ParentFit,
    population_panel,
    population_residuals,
    prepare_images,
)
from edgar_study.frontier.prompts import PROMPTS
from edgar_study.frontier.protocol import fit_equation, solve, split_trials
from edgar_study.frontier.replies import InvalidReply, decode_json
from edgar_study.frontier.seed import BASE_NAMES, BASE_PROBES, BASELINE_ID
from edgar_study.frontier.warm import Parent, declared_probes, warm_start
from edgar_study.go_client import GoClient, Ledger, fetch_models, is_free, load_key
from edgar_study.upstream import import_upstream

MODEL = "hy4-flash-free"
OUTCOMES = ("fitted", "invalid", "duplicate", "fit_failed", "unreadable")
BLOW_UP = 2.0
log = logging.getLogger(__name__)


def _ideas(wave):
    return {
        row["node"]: Idea(
            row["node"],
            row["branch"],
            row["mechanism"],
            row["text"],
            tuple(map(tuple, row["routes"])),
            row["lane"],
            row["label"],
        )
        for row in wave["idea_catalog"]
    }


def contexts(wave):
    """Fixed contexts from a finished run: every new-idea mode on the baseline, and the run's strongest
    ideas as parents, revisits and a combination. Scores pick the contexts; no prompt shows them."""
    best, tried = {}, {}
    for event in wave["episodes"]:
        for record in event["candidates"]:
            if record.get("status") in {"fitted", "fit_failed"} and record.get("equation"):
                tried.setdefault(event["node"], []).append([record["equation"], record["status"]])
            if record.get("status") == "fitted":
                score = record["metrics"]["score"]
                if event["node"] not in best or score < best[event["node"]][0]:
                    best[event["node"]] = (score, record["proposal"])
    ideas = _ideas(wave)
    ranked = [node for node, _ in sorted(best.items(), key=lambda item: item[1][0]) if node in ideas]
    references = {node: best[node][1] for node in ranked}

    def new(mode, parents):
        return {"node": "ab-new", "kind": "new", "mode": mode, "parents": list(parents)}

    found = [
        {"choice": new("single_idea", [BASELINE_ID])},
        {"choice": new("anomaly_detour", [BASELINE_ID])},
        {"choice": new("independent_root", [])},
    ]
    if ranked:
        top = ranked[0]
        found.append({"choice": new("single_idea", [top])})
        route = list(ideas[top].routes[0]) if ideas[top].routes else []
        revisit = {"node": top, "kind": "revisit", "mode": "revisit", "parents": route}
        found.append({"choice": revisit, "tried": tried.get(top, [])[-8:]})
    if len(ranked) > 1:
        found.append({"choice": new("combination", sorted(ranked[:2]))})
    for context in found:
        context.setdefault("tried", [])
        context["references"] = references
    return found


def summarise(rows):
    """Outcome rates and gains per arm, and per context which arm's best equation gained more."""
    arms = sorted({row["arm"] for row in rows})
    summary = {}
    for arm in arms:
        mine = [row for row in rows if row["arm"] == arm]
        gains = [row["gain"] for row in mine if row["status"] == "fitted" and row["gain"] is not None]
        summary[arm] = {
            "equations": len(mine),
            **{
                outcome: sum(row["status"] == outcome for row in mine) / max(1, len(mine))
                for outcome in OUTCOMES
            },
            "blow_up": sum(bool(row.get("blow_up")) for row in mine) / max(1, len(mine)),
            "beats_parent": sum(gain > 0 for gain in gains) / max(1, len(mine)),
            "median_gain": statistics.median(gains) if gains else None,
            "best_gain": max(gains) if gains else None,
        }
    paired = {"contexts": 0, "ties": 0, **{f"{arm}_better": 0 for arm in arms}}
    for context in sorted({row["context"] for row in rows}):
        top = {}
        for arm in arms:
            gains = [
                row["gain"]
                for row in rows
                if row["arm"] == arm
                and row["context"] == context
                and row["status"] == "fitted"
                and row["gain"] is not None
            ]
            top[arm] = max(gains) if gains else -np.inf
        if all(value == -np.inf for value in top.values()):
            continue
        paired["contexts"] += 1
        leader = max(top.values())
        winners = [arm for arm, value in top.items() if value == leader]
        if len(winners) > 1:
            paired["ties"] += 1
        else:
            paired[f"{winners[0]}_better"] += 1
    summary["paired"] = paired
    return summary


def _equations(reply, kind, k):
    value = decode_json(reply)
    if kind == "new":
        if not isinstance(value, dict):
            raise InvalidReply("invalid_idea_reply", "Expected an object with idea and equations.")
        value = value.get("equations", [])
    if not isinstance(value, list):
        raise InvalidReply("invalid_reply", "Expected a list of equations.")
    return value[:k]


def _report(summary):
    arms = [arm for arm in summary if arm != "paired"]
    keys = ("equations", *OUTCOMES, "blow_up", "beats_parent", "median_gain", "best_gain")
    lines = ["| Measure | " + " | ".join(arms) + " |", "| --- |" + " ---: |" * len(arms)]
    for key in keys:
        cells = []
        for arm in arms:
            value = summary[arm][key]
            if value is None:
                cells.append("–")
            elif key == "equations":
                cells.append(str(value))
            elif key.endswith("gain"):
                cells.append(f"{value:+.3f}")
            else:
                cells.append(f"{value:.0%}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    paired = summary["paired"]
    lines.append("")
    lines.append(
        f"Paired by context ({paired['contexts']}): "
        + ", ".join(f"{arm} better in {paired[f'{arm}_better']}" for arm in arms)
        + f", ties {paired['ties']}."
    )
    return "\n".join(lines)


def run(args):
    """Sample every arm on every context, fit the equations on a subset of neurons and report."""
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = sorted(set(arms) - set(PROMPTS))
    if unknown or len(arms) < 2:
        raise SystemExit(f"Choose at least two arms from {sorted(PROMPTS)}.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    entry = fetch_models(load_key()).get(MODEL)
    if not entry or not is_free(entry):
        raise SystemExit(f"{MODEL} is not confirmed free.")
    wave = json.loads(Path(args.wave).read_text())
    found = contexts(wave)
    up = import_upstream(args.upstream.resolve())
    fitting.install(up["hypothesis_engine"], "lm-bounded")
    response, angles, _ = load_dataset(args.dataset, up["utils"].extract_stimulus_related_response)
    x, y, groups, _ = prepare(response, angles, "strict-holdout-v1")
    explore, _ = discovery_partition(groups)
    x, y = np.asarray(x)[explore], np.asarray(y)[explore]
    subset = (x[: args.cells], y[: args.cells])
    scale = {
        "baseline": float(np.median(np.percentile(y, 5, axis=1))),
        "peak": float(np.median(np.percentile(y, 95, axis=1))),
    }
    baseline = double_gaussian_parent(up)
    _, held = split_trials(subset[0])
    held = np.asarray(held)
    nodes = _ideas(wave)
    fits: dict[str, _Comparator] = {}

    def parent(name, references):
        if name not in fits:
            if name == BASELINE_ID:
                score, mse, _, params = baseline.solve(subset, args.max_iter)
                link = Parent(name, BASE_NAMES, baseline.model, np.array(BASE_PROBES), lambda c, m: params)
                fit = ParentFit(name, baseline.model, baseline.estimator, baseline.solve)
            else:
                equation = compile_equation(references[name])
                result = fit_equation(equation, subset, args.max_iter)
                score, mse, params = result.score, result.mse, result.params
                link = Parent(
                    name,
                    tuple(equation.initial),
                    equation.model,
                    declared_probes(equation),
                    lambda c, m: params,
                )
                fit = ParentFit(name, equation.model, equation.estimator, partial(solve, equation))
            panel = population_panel(
                name, *population_residuals(link.model, params, subset[0][:, held], subset[1][:, held])
            )
            fits[name] = _Comparator(float(score), float(mse), link, fit, panel)
        return fits[name]

    plans: list[_Plan] = []
    for index, context in enumerate(found):
        choice, references = context["choice"], context["references"]
        names = choice["parents"] or [BASELINE_ID]
        comparators = [parent(name, references) for name in names]
        shown = tuple(parent(name, references).fit for name in choice["parents"])
        batch = prepare_images(up, x, y, seed=args.seed + index, parents=shown, max_iter=args.max_iter)
        block = NEURONS * IMAGES_PER_NEURON
        blocks = [batch.panels[i * block : (i + 1) * block] for i in range(max(1, len(shown)))]
        populations = [parent(name, references).panel for name in names]
        with_population = tuple(
            panel for pop, panels in zip(populations, blocks, strict=True) for panel in (pop, *panels)
        )
        compiled = {name: _Reference(compile_equation(p)) for name, p in references.items()}
        tried = [tuple(row) for row in context["tried"]]
        for arm in arms:
            prompt = PROMPTS[arm](choice, nodes, compiled, args.k, tried, scale)
            images = batch if arm == "v1" else FrontierImages(batch.cells, batch.parent_ids, with_population)
            for sample in range(args.samples):
                plans.append(
                    _Plan(
                        index,
                        arm,
                        sample,
                        prompt,
                        images,
                        min(comparators, key=lambda c: c.score),
                        [c.link for c in comparators if choice["parents"]],
                    )
                )

    async def ask():
        go = GoClient(output, Ledger(output / "spend.jsonl"), output.name, [MODEL], max_tokens=8192)
        try:
            return await asyncio.gather(
                *(plan.images.submit(go, plan.prompt, effort="none", temperature=0.5) for plan in plans)
            )
        finally:
            await go.close()

    replies = asyncio.run(ask())
    rows, seen = [], {}
    for plan, reply in zip(plans, replies, strict=True):
        base = {"arm": plan.arm, "context": plan.context, "sample": plan.sample}
        kind = found[plan.context]["choice"]["kind"]
        try:
            proposals = _equations(reply, kind, args.k)
        except InvalidReply as exc:
            rows.append({**base, "status": "unreadable", "gain": None, "error": str(exc)})
            continue
        keys = seen.setdefault((plan.arm, plan.context), set())
        for proposal in proposals:
            row = {**base, "proposal": proposal, "gain": None}
            try:
                equation = compile_equation(proposal)
            except InvalidEquation as exc:
                rows.append({**row, "status": "invalid", "error": f"{exc.code}: {exc}"})
                continue
            if equation.key in keys:
                rows.append({**row, "status": "duplicate"})
                continue
            keys.add(equation.key)
            comparator = plan.comparator
            try:
                warm = warm_start(equation, plan.links, subset, args.max_iter)
                result = fit_equation(equation, subset, args.max_iter, warm=warm)
            except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
                rows.append({**row, "status": "fit_failed", "error": str(exc)})
                continue
            rows.append(
                {
                    **row,
                    "status": "fitted",
                    "score": result.score,
                    "gain": comparator.score - result.score,
                    "blow_up": result.mse > BLOW_UP * comparator.mse,
                }
            )
            log.info("context %s %s: gain %+.3f", plan.context, plan.arm, rows[-1]["gain"])
    summary = summarise(rows)
    (output / "contexts.json").write_text(json.dumps(found, indent=2) + "\n")
    (output / "replies.json").write_text(
        json.dumps(
            [
                {
                    "context": plan.context,
                    "arm": plan.arm,
                    "sample": plan.sample,
                    "prompt": plan.prompt,
                    "reply": reply,
                }
                for plan, reply in zip(plans, replies, strict=True)
            ],
            indent=2,
        )
        + "\n"
    )
    (output / "rows.jsonl").write_text("".join(json.dumps(row, default=float) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(_report(summary))


@dataclass(frozen=True)
class _Comparator:
    score: float
    mse: float
    link: Parent
    fit: ParentFit
    panel: bytes


@dataclass(frozen=True)
class _Plan:
    context: int
    arm: str
    sample: int
    prompt: str
    images: FrontierImages
    comparator: _Comparator
    links: list


class _Reference:
    """What the prompts read from a reference: its equation text."""

    def __init__(self, equation):
        self.text = equation.text

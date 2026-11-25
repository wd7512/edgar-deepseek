"""Run a score-blind HY4 frontier wave and confirm a frozen shortlist."""

import asyncio
import fcntl
import json
import logging
from collections import Counter
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np

from edgar_study import fitting
from edgar_study.datasets import load_dataset, prepare
from edgar_study.frontier.fits import (
    BASE_NAMES,
    BASE_PROBES,
    baseline_templates,
    double_gaussian_parent,
    fit_nested,
    fit_offset,
    measure,
    offset_bounds,
    offset_embedding,
    verified_mapping,
)
from edgar_study.frontier.grammar import InvalidEquation, compile_equation
from edgar_study.frontier.ideas import Idea, draw, draw_generated, generated_idea, route_mode, validate_graph
from edgar_study.frontier.images import ParentFit, prepare_images
from edgar_study.frontier.prompts import generated_prompt, idea_prompt
from edgar_study.frontier.replies import InvalidReply, decode, decode_json, equation_list
from edgar_study.frontier.seed import BASE, BASELINE_ID, baseline_node
from edgar_study.go_client import GoClient, Ledger, fetch_models, is_free, load_key
from edgar_study.upstream import import_upstream

MODEL = "hy4-flash-free"
log = logging.getLogger(__name__)


def _write(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


async def execute(
    up,
    proposer,
    explore,
    confirm,
    output,
    *,
    ideas=None,
    batches=6,
    k=4,
    seed=0,
    max_iter=2000,
    final=None,
    waves=1,
    resume=False,
    new_probability=0.8,
):
    """Explore rolling waves, then confirm a frozen shortlist; optionally select and test once."""
    discovery = ideas is None
    ideas = () if discovery else ideas
    validate_graph(ideas)
    if not 0 <= new_probability <= 1:
        raise ValueError("New-idea probability must be between zero and one.")
    if min(batches, k, waves) < 1:
        raise ValueError("Wave count, batch count and equations per batch must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "wave.json"
    if path.exists() and not resume:
        raise FileExistsError(path)
    nodes = {i.node: i for i in ideas}
    references, metrics, attempts, depth, seen, exact = {}, {}, Counter(), Counter(), set(), {BASE}
    lineage = {}
    history: dict[str, list[bool]] = {}
    tried: dict[str, list[str]] = {}
    rng = np.random.default_rng(seed)
    born = set()
    wave: dict[str, Any] = {
        "version": 3,
        "seed": seed,
        "fit_protocol": "nested_parent_v3",
        "idea_source": "generated" if discovery else "catalog",
        "idea_catalog": [asdict(i) for i in ideas],
        "graph": [baseline_node()],
        "planned_batches": batches * waves,
        "episodes": [],
        "confirmation": [],
        "status": "exploring",
    }
    config = {"batches": batches, "waves": waves, "k": k, "max_iter": max_iter, "finalize": final is not None}
    if discovery:
        config["new_probability"] = new_probability

    def next_choice(step):
        if discovery:
            return draw_generated(nodes, references, history, rng, new_probability, step)
        return draw(ideas, references, attempts, rng, depth)

    def track_proposals(event):
        if not discovery:
            return
        if event["node"] in nodes:
            history.setdefault(event["node"], []).extend(
                row["status"] in {"duplicate", "invalid"} for row in event["candidates"]
            )
        equations = tried.setdefault(event["node"], [])
        for record in event["candidates"]:
            equation = record.get("equation")
            if equation and equation not in equations:
                equations.append(equation)
        del equations[:-8]

    if resume:
        saved = json.loads(path.read_text())
        fields = ("version", "seed", "idea_source", "planned_batches", "fit_protocol")
        if any(saved.get(key) != json.loads(json.dumps(wave[key])) for key in fields):
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if saved.get("config", config) != config:
            raise ValueError("Run checkpoint does not match this frontier configuration.")
        if not discovery and saved["idea_catalog"] != json.loads(json.dumps(wave["idea_catalog"])):
            raise ValueError("Run checkpoint does not match the idea catalog.")
        if saved["status"] in {"complete", "no_fitted_equations"}:
            return saved
        if "selection" in saved:
            raise ValueError("Cannot resume after selection has started; locked tests are one-shot.")
        for step, event in enumerate(saved["episodes"]):
            choice = next_choice(step)
            node = event["node"]
            if not choice or any(event.get(key) != choice[key] for key in choice):
                raise ValueError("Checkpoint does not match the scheduler replay.")
            if discovery and "idea" in event:
                value = event["idea"]
                nodes[node] = Idea(
                    node,
                    value["branch"],
                    value["mechanism"],
                    value["text"],
                    tuple(map(tuple, value["routes"])),
                    value["lane"],
                    value["label"],
                )
                born.add(node)
            attempts[node] += 1
            for record in event["candidates"]:
                if record["status"] not in {"fitted", "fit_failed"}:
                    continue
                candidate = compile_equation(record["proposal"])
                seen.add(candidate.key)
                exact.add(candidate.text.strip())
                born.add(node)
                if record["status"] == "fitted":
                    depth[nodes[node].branch] += 1
                    if node not in references:
                        references[node], metrics[node] = candidate, record["metrics"]
                        lineage[node] = event["parents"]
            track_proposals(event)
        if discovery:
            replayed = {event["node"] for event in saved["episodes"]}
            saved["graph"] = [
                row for row in saved["graph"] if row["node"] == BASELINE_ID or row["node"] in replayed
            ]
            saved["idea_catalog"] = [row for row in saved["idea_catalog"] if row["node"] in replayed]
        wave = saved
        wave.pop("active_probe", None)
        wave["confirmation"] = []
        wave["status"] = "exploring"
    wave["config"] = config
    _write(path, wave)
    if "baseline_exploration" not in wave:
        wave["baseline_exploration"] = measure(up, double_gaussian_parent(up), explore, max_iter)

    baseline_fit = double_gaussian_parent(up)

    @cache
    def reference_fit(node):
        if node == BASELINE_ID:
            return baseline_fit
        candidate = references[node]
        solve, _ = fit_strategy(node, candidate, tuple(lineage[node]))
        return ParentFit(node, candidate.model, candidate.estimator, solve)

    def fit_strategy(node, equation, parents):
        mappings = []
        bounds = None
        standalone_offset = tuple(parents) in {(), (BASELINE_ID,)}
        supplied = equation.nesting if isinstance(equation.nesting, dict) else {}
        template = baseline_templates().get(equation.key) if len(equation.initial) > len(BASE_NAMES) else None
        if template is not None or BASELINE_ID in supplied:
            bindings = (
                dict(zip(equation.canonical_names, template, strict=True))
                if template is not None
                else supplied[BASELINE_ID]
            )
            embed = verified_mapping(equation, baseline_fit, BASE_NAMES, BASE_PROBES, bindings)
            if embed is not None:
                mappings.append((baseline_fit, embed))
                if standalone_offset:
                    bounds = offset_bounds(embed)
        for name in parents:
            if name == BASELINE_ID:
                continue
            parent = references[name]
            if name not in supplied or len(equation.initial) <= len(parent.initial):
                continue
            initial = np.array(tuple(parent.initial.values()))
            perturb = np.random.default_rng(0).uniform(-0.3, 0.3, size=(4, len(initial)))
            probes = np.concatenate((initial[None, :], initial + perturb * (0.2 + np.abs(initial))))
            fit = reference_fit(name)
            embed = verified_mapping(equation, fit, tuple(parent.initial), probes, supplied[name])
            if embed is not None:
                mappings.append((fit, embed))
        if not mappings:
            if standalone_offset and len(equation.initial) == 6:
                try:
                    start = offset_embedding(equation, baseline_fit.model)
                except ValueError:
                    return None, "estimator"
                return partial(fit_offset, up, equation, embedding=start), "fitted_double_gaussian"
            return None, "estimator"
        source = (
            "fitted_parent"
            if any(parent.node_id != BASELINE_ID for parent, _ in mappings)
            else "fitted_double_gaussian"
        )
        return partial(fit_nested, up, equation, mappings=tuple(mappings), bounds=bounds), source

    def measure_node(node, equation, cohort, *, parents=None):
        solve, source = fit_strategy(node, equation, tuple(lineage[node] if parents is None else parents))
        return measure(up, equation, cohort, max_iter, solve=solve, initialization=source)

    def evaluate(proposal, event, step, *, replaces=None, replacement=False):
        node, parents = event["node"], event["parents"]
        idea = nodes[node]
        record = {"proposal": proposal}
        if replacement:
            record["source"] = "retry"
            if replaces is not None:
                record["replaces"] = replaces
        event["candidates"].append(record)
        try:
            candidate = compile_equation(proposal)
        except InvalidEquation as exc:
            record.update(status="invalid", reason_code=exc.code, error=str(exc))
            return exc.code, str(exc)
        record["equation"] = candidate.text
        text = candidate.text.strip()
        if text in exact or candidate.key in seen:
            record["status"] = "duplicate"
            return ("exact_repeat", json.dumps(text)) if text in exact else None
        seen.add(candidate.key)
        exact.add(text)
        if node not in born:
            born.add(node)
            routes = idea.routes or (() if idea.lane == "root" else ((BASELINE_ID,),))
            wave["graph"].append({**asdict(idea), "routes": routes, "introduced_at": step + 1})
        try:
            result = measure_node(node, candidate, explore, parents=parents)
            record.update(status="fitted", metrics=result)
            depth[idea.branch] += 1
            if node not in references:
                references[node], metrics[node] = candidate, result
                lineage[node] = parents
        except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as exc:
            record.update(status="fit_failed", error=str(exc))
        return None

    def decode_probe(reply, event, step):
        if not discovery:
            return decode(reply, k)
        value = decode_json(reply)
        if event["kind"] == "revisit":
            return equation_list(value, k, exact=False)
        if not isinstance(value, dict):
            raise InvalidReply("invalid_idea_reply", "Return an object containing idea and equations.")
        equations = equation_list(value.get("equations", []), k, exact=False)
        accepted = generated_idea(value.get("idea"), event, nodes)
        nodes[accepted.node] = accepted
        born.add(accepted.node)
        record = asdict(accepted)
        event["idea"] = record
        wave["active_probe"].update(idea=record, label=accepted.label)
        _write(path, wave)
        wave["idea_catalog"].append(record)
        wave["graph"].append({**record, "introduced_at": step + 1})
        return equations

    for step in range(len(wave["episodes"]), batches * waves):
        choice = next_choice(step)
        if choice is None:
            break
        node, parents = choice["node"], choice["parents"]
        idea = nodes.get(node)
        attempts[node] += 1
        fitted = tuple(reference_fit(p) for p in parents)
        prompt = (
            generated_prompt(choice, nodes, references, k, tried.get(node, ()))
            if discovery
            else idea_prompt(idea, parents, nodes, references, k)
        )
        wave_number = step // batches
        event = {
            **choice,
            "mode": choice["mode"] if discovery else route_mode(idea, parents),
            "wave": wave_number + 1,
            "image_seed": seed + wave_number,
            "prompt": prompt,
            "candidates": [],
        }
        wave["active_probe"] = {
            "batch": step + 1,
            "node": node,
            "label": (idea.label or idea.text) if idea else "New idea",
            "lane": choice["lane"],
            "mode": event["mode"],
        }
        _write(path, wave)
        batch = None
        unreadable = None

        try:
            batch = prepare_images(up, *explore, seed=event["image_seed"], parents=fitted, max_iter=max_iter)
            event.update(cells=list(batch.cells), images=len(batch.images))
            event["reply"] = await proposer.submit(batch, prompt)
            proposals = decode_probe(event["reply"], event, step)
        except InvalidReply as exc:
            unreadable = exc
            event["reply_error"] = {"reason_code": exc.code, "error": str(exc)}
            proposals = []
        except ValueError as exc:
            event["error"] = str(exc)
            proposals = []
        targets = []
        for position, proposal in enumerate(proposals, start=1):
            rejected = evaluate(proposal, event, step)
            if rejected is not None:
                reason, detail = rejected
                targets.append({"position": position, "reason_code": reason, "detail": detail})
        retried = unreadable is not None
        if unreadable is not None and batch is not None:
            instruction = (
                f"{unreadable.code}: {unreadable} Return the entire reply again in the requested format."
            )
            retry = event["retry"] = {"instruction": instruction, "reason_code": unreadable.code}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"] or "", instruction)
                replacements = decode_probe(retry["reply"], event, step)
            except ValueError as exc:
                retry["error"] = str(exc)
                event["error"] = str(unreadable)
            else:
                for replacement in replacements:
                    evaluate(replacement, event, step, replacement=True)
        elif targets and batch is not None:
            retried = True
            reasons = "\n".join(
                f"- Proposal {target['position']}: {target['reason_code']} — {target['detail']}"
                for target in targets
            )
            instruction = (
                f"Replace these proposals from your original reply:\n{reasons}\n"
                "Suggest one different equation for each listed proposal, keeping the same idea. "
                f"Return only a JSON array of exactly {len(targets)} replacements in the listed order "
                "and original format."
            )
            retry = event["retry"] = {"instruction": instruction, "targets": targets}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"], instruction)
                replacements = decode(retry["reply"], len(targets))
            except ValueError as exc:
                retry["error"] = str(exc)
            else:
                for replacement, target in zip(replacements, targets, strict=True):
                    evaluate(replacement, event, step, replaces=target["position"], replacement=True)
        fitted_here = any(
            record.get("status") == "fitted" for record in event["candidates"] if "status" in record
        )
        if batch is not None and not retried and not fitted_here:
            instruction = (
                "No equation from your last reply could be used. Return a fresh reply in the requested"
                " format with materially different equations. Do not repeat earlier equations."
            )
            retry = event["retry"] = {"instruction": instruction, "reason_code": "no_usable_equation"}
            try:
                retry["reply"] = await proposer.retry(batch, prompt, event["reply"] or "", instruction)
                if discovery and (event.get("idea") or event["kind"] == "revisit"):
                    replacements = equation_list(decode_json(retry["reply"]), k, exact=False)
                else:
                    replacements = decode(retry["reply"], k)
            except ValueError as exc:
                retry["error"] = str(exc)
            else:
                for replacement in replacements:
                    evaluate(replacement, event, step, replacement=True)
        log.info("Frontier episode %s: %s (%s images)", step + 1, node, event.get("images"))
        del wave["active_probe"]
        track_proposals(event)
        wave["episodes"].append(event)
        _write(path, wave)
    wave["shortlist"] = {
        node: {
            "equation": eq.text,
            "initial": eq.initial,
            "exploration": metrics[node],
            "parents": lineage[node],
        }
        for node, eq in references.items()
    }
    wave["confirmation_rule"] = (
        "Lower whole-curve MSE versus the double Gaussian and every used parent; "
        "no 30-degree bin may regress over 5% of the comparator's MSE."
    )
    wave["status"] = "confirming"
    _write(path, wave)
    if not references:
        wave["status"] = "no_fitted_equations"
        _write(path, wave)
        return json.loads(path.read_text())
    baseline = measure(up, double_gaussian_parent(up), confirm, max_iter)
    wave["confirmation_baseline"] = baseline
    measured = {}
    for node, eq in references.items():
        row = {"node": node}
        try:
            result = measure_node(node, eq, confirm)
            measured[node] = result
            comparators = {
                BASELINE_ID: baseline,
                **{p: measured[p] for p in lineage[node] if p != BASELINE_ID},
            }
            improved = all(
                result["mse"] < ref["mse"]
                and all(
                    a - b <= 0.05 * ref["mse"]
                    for a, b in zip(result["angle_mse"], ref["angle_mse"], strict=True)
                    if a is not None and b is not None
                )
                for ref in comparators.values()
            )
            row.update(metrics=result, outcome="confirmed_fit" if improved else "unresolved")
        except (KeyError, ValueError) as exc:
            row.update(outcome="inconclusive", error=str(exc))
        wave["confirmation"].append(row)
    supported = {row["node"] for row in wave["confirmation"] if row["outcome"] == "confirmed_fit"}

    def supported_claim(node):
        return node == BASELINE_ID or (
            node in supported and all(supported_claim(parent) for parent in lineage[node])
        )

    wave["claims"] = {
        row["node"]: "supported_fit" if supported_claim(row["node"]) else "unresolved"
        for row in wave["graph"][1:]
    }
    if final is not None and supported:
        wave["selection"] = {"status": "started"}
        _write(path, wave)
        selection, locked = final
        scores = {"double_gaussian": measure(up, double_gaussian_parent(up), selection, max_iter)}
        for node in sorted(n for n in supported if supported_claim(n)):
            try:
                scores[node] = measure_node(node, references[node], selection)
            except ValueError:
                continue
        winner = min(scores, key=lambda name: (scores[name]["score"], name))
        wave["selection"] = {"metrics": scores, "winner": winner}
        _write(path, wave)
        locked_baseline = measure(up, double_gaussian_parent(up), locked, max_iter)
        wave["locked"] = {"double_gaussian": locked_baseline}
        if winner != "double_gaussian":
            wave["locked"][winner] = measure_node(winner, references[winner], locked)
    wave["status"] = "complete"
    _write(path, wave)
    return json.loads(path.read_text())


class _Live:
    def __init__(self, client):
        self.client = client

    async def submit(self, batch, prompt):
        return await batch.submit(self.client, prompt, effort="none", temperature=0.5)

    async def retry(self, batch, prompt, previous_reply, instruction):
        return await batch.submit(
            self.client, prompt, effort="none", temperature=0.5, followup=(previous_reply, instruction)
        )


def run(args):
    """Load the pinned recording and run one explicitly bounded, free-model wave."""
    output = Path(args.output)
    if args.resume:
        if not (output / "manifest.json").is_file() or not (output / "wave.json").is_file():
            raise SystemExit("Resume requires an existing run manifest and checkpoint.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("This run is already active.") from exc
        _run(args, output)


def _run(args, output):
    if min(args.waves, args.batches, args.k) < 1 or args.seed < 0:
        raise SystemExit("Use positive waves, batches and K and a nonnegative seed.")
    ideas = None
    if args.graph is not None:
        ideas = tuple(
            Idea(
                node=row["node"],
                branch=row["branch"],
                mechanism=row["mechanism"],
                text=row["text"],
                routes=tuple(tuple(route) for route in row.get("routes", ())),
                lane=row.get("lane", "frontier"),
                label=row.get("label", ""),
            )
            for row in json.loads(args.graph.read_text())
        )
    if ideas is not None:
        validate_graph(ideas)
    if not 0 <= args.new_probability <= 1:
        raise SystemExit("--new-probability must be between zero and one.")
    entry = fetch_models(load_key()).get(MODEL)
    if not entry or not is_free(entry) or "image/png" not in entry["modalities"]["input"]:
        raise SystemExit(f"{MODEL} is not confirmed free with PNG support.")
    up = import_upstream(args.upstream.resolve())
    fitting.install(up["hypothesis_engine"], "lm-bounded")
    response, angles, sha = load_dataset(args.dataset, up["utils"].extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    discovery = np.random.default_rng(881).permutation(groups["discovery"])
    cut = len(discovery) * 3 // 4
    explore, confirm = discovery[:cut], discovery[cut:]
    manifest = {
        "dataset_sha256": sha,
        "model": MODEL,
        "seed": args.seed,
        "discovery_partition_seed": 881,
        "exploration_cells": indices["eligible_cell_ids"][explore].tolist(),
        "confirmation_cells": indices["eligible_cell_ids"][confirm].tolist(),
        "batches": args.batches,
        "k": args.k,
        "waves": args.waves,
        "finalize": args.finalize,
        "idea_source": "generated" if ideas is None else "catalog",
        "new_probability": args.new_probability if ideas is None else None,
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("Resume requires the same dataset, split, and run options.")
    else:
        _write(manifest_path, manifest)

    def cohort(rows):
        return x[rows], y[rows]

    async def wave():
        go = GoClient(output, Ledger(output / "spend.jsonl"), output.name, [MODEL], max_tokens=8192)
        try:
            final = (cohort(groups["selection"]), cohort(groups["locked_test"])) if args.finalize else None
            return await execute(
                up,
                _Live(go),
                cohort(explore),
                cohort(confirm),
                output,
                batches=args.batches,
                k=args.k,
                seed=args.seed,
                final=final,
                waves=args.waves,
                ideas=ideas,
                new_probability=args.new_probability,
                resume=args.resume,
            )
        finally:
            await go.close()

    result = asyncio.run(wave())
    print(f"{result['status']}: {output / 'wave.json'}")
    if result["status"] != "complete":
        raise SystemExit(1)

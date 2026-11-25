"""Serve a local, read-only replay of a frontier wave."""

import json
import os
from collections import Counter, defaultdict
from contextlib import suppress
from hashlib import sha256
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from textwrap import wrap
from urllib.parse import parse_qs, urlsplit

from edgar_study.frontier_seed import BASELINE_ID, baseline_node

MODES = {
    "single_idea": "Single-idea frontier",
    "combination": "AND combination",
    "anomaly_detour": "Anomaly detour",
    "independent_root": "Independent root",
}

LAYOUT = {
    "width": (145, 145),
    "drift": (390, 80),
    "width-drift": (165, 240),
    "tip": (590, 145),
    "offset": (845, 90),
    "peak-pair": (750, 260),
    "harmonic": (190, 350),
    "window": (560, 405),
    "blend": (870, 385),
    BASELINE_ID: (500, 305),
}

STYLE = """
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101721;color:#edf4fc;
font:13px system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:12px 20px;position:relative}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:0}p,small{color:#acbfd0}
.bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:7px 0}
a,.pill{color:#b7ecff;padding:2px 4px;text-decoration:none}a:hover{text-decoration:underline}
a.active{color:#80e8cd;font-weight:700}.graph-shell{position:relative;border-block:1px solid #344759}
.graph{display:block;width:100%;height:min(var(--graph-h),max(125px,calc(100dvh - 155px)))}
.node{cursor:grab;touch-action:none;outline:none}.node:active{cursor:grabbing}
.node rect{fill:#222f40;stroke:#899aae;stroke-width:1.5}
.node.current rect,.node:focus-visible rect{stroke:#b7ecff;stroke-width:2.5}
.node text{fill:#edf4fc;font-size:12px;font-weight:650;text-anchor:middle;pointer-events:none;user-select:none}
.node .sub{fill:#bfd0e1;font-size:10px;font-weight:400}.route path{fill:none;stroke:#657686;stroke-width:1.5}
.route circle{fill:#263647;stroke:#657686}.route text{fill:#e0e8f2;font-size:9px;text-anchor:middle}
button{font:inherit;color:#b7ecff;background:#182331;border:1px solid #41546a;border-radius:4px;cursor:pointer}
.inspector{position:absolute;right:0;top:0;width:min(360px,100%);max-height:100%;overflow:auto;z-index:1;
background:#182331;border:1px solid #41546a;border-radius:4px;padding:12px;overflow-wrap:anywhere}
.inspector>button{float:right;margin-left:8px}.probe{border-top:1px solid #41546a;margin-top:8px;padding-top:8px}
summary{cursor:pointer;padding:3px 0}code{color:#c3e9de;white-space:pre-wrap}input{accent-color:#59d8c6;max-width:150px}
"""

SCRIPT = """
const $ = (q, el = document) => el.querySelector(q);
const $$ = (q, el = document) => Array.from(el.querySelectorAll(q));
const storageKey = 'frontier-layout:' + $('main').dataset.layout;
let positions = new Map(), dragging, selected;
try { positions = new Map(JSON.parse(localStorage.getItem(storageKey) || '[]')); } catch {}
function save() {
  try { localStorage.setItem(storageKey, JSON.stringify([...positions])); } catch {}
}
function setPosition(node, x, y) {
  node.setAttribute('transform', `translate(${x},${y})`);
  node.dataset.x = x;
  node.dataset.y = y;
}
function edge([x, y], [tx, ty], w, h, pad = 0) {
  const dx = tx - x, dy = ty - y;
  const scale = Math.min(1, 1 / Math.max(Math.abs(dx) / (w + pad), Math.abs(dy) / (h + pad)));
  return [x + dx * scale, y + dy * scale];
}
function drawArrows() {
  const nodes = new Map($$('.node').map(n => [n.dataset.node, [+n.dataset.x, +n.dataset.y]]));
  for (const route of $$('.route')) {
    const child = nodes.get(route.dataset.child);
    const paths = $$('path[data-parent]', route);
    const parents = paths.map(path => nodes.get(path.dataset.parent));
    if (parents.length === 1) {
      const [parent] = parents;
      paths[0].setAttribute('d', `M${edge(parent, child, 96, 34)} L${edge(child, parent, 96, 34, 3)}`);
      continue;
    }
    const center = parents.reduce(([x, y], [px, py]) => [x + px / parents.length, y + py / parents.length], [0, 0]);
    let dx = center[0] - child[0], dy = center[1] - child[1];
    if (!dx && !dy) dy = -1;
    const len = Math.hypot(dx, dy), offset = +route.dataset.offset;
    const anchor = [child[0] + dx / len * 160 - dy / len * offset, child[1] + dy / len * 125 + dx / len * offset];
    const joint = edge(child, anchor, 96, 34, 33);
    paths.forEach((path, i) => path.setAttribute('d',
      `M${edge(parents[i], joint, 96, 34)} L${edge(joint, parents[i], 12, 12)}`));
    $('.joint', route).setAttribute('transform', `translate(${joint})`);
    $('[data-out]', route).setAttribute('d', `M${edge(joint, child, 12, 12)} L${edge(child, joint, 96, 34, 3)}`);
  }
}
function applyLayout() {
  for (const node of $$('.node')) {
    setPosition(node, ...(positions.get(node.dataset.node) || [+node.dataset.homeX, +node.dataset.homeY]));
  }
  drawArrows();
}
function inspect(id, expanded = new Set()) {
  const detail = $$('.inspector').find(el => el.dataset.ideaDetail === id);
  for (const el of $$('.inspector')) el.hidden = el !== detail;
  selected = detail?.dataset.ideaDetail;
  if (detail) for (const item of $$('details', detail)) item.open ||= expanded.has(item.dataset.equationId);
}
function point(event, svg) {
  return new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
}
document.addEventListener('pointerdown', event => {
  const node = event.target.closest('.node');
  if (!node || !event.isPrimary || event.button !== 0) return;
  const svg = node.ownerSVGElement;
  const start = point(event, svg);
  dragging = {node, svg, start, x: +node.dataset.x, y: +node.dataset.y, id: event.pointerId, moved: false};
  node.setPointerCapture(event.pointerId);
});
document.addEventListener('pointermove', event => {
  if (dragging?.id !== event.pointerId) return;
  const {node, svg, start, x, y} = dragging;
  const p = point(event, svg), dx = p.x - start.x, dy = p.y - start.y;
  if (!dragging.moved && Math.hypot(dx, dy) < 4) return;
  dragging.moved = true;
  const rect = svg.viewBox.baseVal;
  setPosition(node, Math.max(96, Math.min(rect.width - 96, x + dx)),
    Math.max(34, Math.min(rect.height - 34, y + dy)));
  drawArrows();
});
function finishDrag(event) {
  if (dragging?.id !== event.pointerId) return;
  const {node, moved} = dragging;
  dragging = undefined;
  if (moved) {
    positions.set(node.dataset.node, [+node.dataset.x, +node.dataset.y]);
    save();
  } else if (event.type === 'pointerup') {
    inspect(node.dataset.node === selected ? undefined : node.dataset.node);
  }
}
document.addEventListener('pointerup', finishDrag);
document.addEventListener('pointercancel', finishDrag);
document.addEventListener('keydown', event => {
  const node = event.target.closest('.node');
  if (node && ['Enter', ' '].includes(event.key)) {
    event.preventDefault();
    inspect(node.dataset.node);
    $(`.inspector:not([hidden]) button`)?.focus();
  }
  if (event.key === 'Escape' && selected) closeInspector();
});
function closeInspector() {
  const node = $$('.node').find(el => el.dataset.node === selected);
  inspect();
  node?.focus();
}
document.addEventListener('click', event => {
  if (event.target.closest('[data-reset]')) {
    positions.clear();
    save();
    applyLayout();
  }
  if (event.target.closest('[data-close]')) closeInspector();
});
document.addEventListener('change', event => {
  if (event.target.id === 'step') location.search = '?step=' + event.target.value;
});
async function refresh() {
  const current = $('main');
  if (current.dataset.live !== 'true') return;
  try {
    const response = await fetch('/?fragment=1', {cache: 'no-store'});
    if (response.ok) {
      const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
      const fresh = $('main', doc);
      if (fresh && !dragging && fresh.dataset.snapshot !== current.dataset.snapshot) {
        const panel = $('.inspector:not([hidden])', current);
        const expanded = new Set($$('details[open]', panel || current).map(el => el.dataset.equationId));
        const top = panel?.scrollTop || 0;
        current.replaceWith(fresh);
        applyLayout();
        inspect(selected, expanded);
        const detail = $('.inspector:not([hidden])');
        if (detail) detail.scrollTop = top;
      }
    }
  } catch {}
  setTimeout(refresh, 2000);
}
applyLayout();
setTimeout(refresh, 2000);
"""


def _layout(catalog):
    height = max(460, 100 * ((len(catalog) + 3) // 4))
    positions = {node: LAYOUT[node] for node in catalog if node in LAYOUT}
    for node in catalog:
        if node in positions:
            continue
        candidates = (
            (110 + (i * 233) % 781, 42 + (i * 137) % (height - 84)) for i in range(len(catalog) * 24 + 48)
        )
        positions[node] = max(
            candidates,
            key=lambda pos: min(
                (max(abs(pos[0] - x) / 205, abs(pos[1] - y) / 78) for x, y in positions.values()),
                default=0,
            ),
        )
    return positions, height


def _label(idea):
    return idea.get("label") or idea["text"]


def _graph(nodes, catalog, counts, best, current):
    positions, height = _layout(catalog)
    edges, cards = [], []
    for node, idea in nodes.items():
        x, y = positions[node]
        routes = idea["routes"]
        for i, route in enumerate(routes):
            if not all(p in nodes for p in route):
                continue
            label = escape(" + ".join(_label(nodes[p]) for p in route) + " → " + _label(idea))
            edges.append(
                f'<g class="route" data-child="{escape(node)}" data-offset="{40 * (i - (len(routes) - 1) / 2)}">'
                f"<title>{label}</title>"
            )
            marker = ' marker-end="url(#arrow)"'
            for parent in route:
                edges.append(f'<path data-parent="{escape(parent)}"{marker}/>')
            if len(route) > 1:
                edges.append(
                    f'<path data-out=""{marker}/><g class="joint"><circle r="12"/><text y="3">AND</text></g>'
                )
            edges.append("</g>")
        title = _label(idea)
        lines = wrap(title, width=29, max_lines=3, placeholder="…")
        labels = "".join(
            f'<text y="{13 * (i - (len(lines) - 1) / 2) - 6}">{escape(line)}</text>'
            for i, line in enumerate(lines)
        )
        count = counts[node]
        summary = f"{count} equation{'s' if count != 1 else ''}"
        metrics = _metrics_line(best.get(node))
        cards.append(
            f'<g class="node{" current" if node == current else ""}" data-node="{escape(node)}" '
            f'data-equations="{count}" data-home-x="{x}" data-home-y="{y}" '
            f'transform="translate({x},{y})" tabindex="0" role="button" '
            f'aria-label="{escape(title)}: {summary}; {metrics}">'
            f"<title>{escape(idea['text'])} {summary}. MSE is unpenalized and measured on held-out trials.</title>"
            '<rect width="192" height="68" x="-96" y="-34" rx="5"/>'
            f'{labels}<text class="sub" y="25">{metrics}</text></g>'
        )
    return (
        f'<svg class="graph" style="--graph-h:{height}px" viewBox="0 0 1000 {height}" '
        'role="group" aria-label="Draggable prerequisite graph">'
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        'orient="auto"><path d="M0 0L8 4L0 8Z" style="fill:context-stroke"/></marker></defs>'
        + "".join(edges + cards)
        + "</svg>"
    )


def _equation(candidate):
    if candidate.get("equation"):
        return candidate["equation"]
    proposal = candidate.get("proposal")
    if isinstance(proposal, dict) and candidate.get("status") in {"fitted", "duplicate", "fit_failed"}:
        return proposal.get("equation")
    return None


def _mode(event):
    legacy = {"root": "independent_root", "detour": "anomaly_detour"}
    fallback = "combination" if event.get("parents") else legacy.get(event.get("lane"), "single_idea")
    return MODES.get(event.get("mode", fallback), MODES[fallback])


def _score_label(metrics, reason=""):
    score = (metrics or {}).get("score")
    if isinstance(score, (float, int)):
        return f'<span title="Penalized exploration fit score; lower is better">Fit score: {score:.6g}</span>'
    return f"No fit score ({escape(reason)})" if reason else "No fit score"


def _metrics_line(metrics):
    if not metrics or not all(key in metrics for key in ("score", "mse", "parameters")):
        return "Score — · MSE — · params —"
    return f"Score {metrics['score']:.5g} · MSE {metrics['mse']:.5g} · {metrics['parameters']} params"


def _metrics_detail(metrics):
    if metrics and "mse" in metrics and "parameters" in metrics:
        return f"<p>Held-out MSE: {metrics['mse']:.6g} · Free parameters: {metrics['parameters']}</p>"
    return ""


def _inspector(node, idea, probes, reference, baseline_metrics):
    seed = idea.get("equation") if node == BASELINE_ID else None
    cards = (
        [
            '<details class="probe" data-equation-id="seed">'
            f"<summary>Starting equation · {_score_label(baseline_metrics)}</summary>"
            f"<code>{escape(seed)}</code>{_metrics_detail(baseline_metrics)}</details>"
        ]
        if seed
        else []
    )
    for batch, event in probes:
        equations = []
        invalid = 0
        for i, candidate in enumerate(event.get("candidates", []), 1):
            equation = _equation(candidate)
            if not equation:
                invalid += candidate.get("status") == "invalid"
                continue
            reason = {"duplicate": "duplicate", "fit_failed": "fit failed"}.get(candidate.get("status"), "")
            score = _score_label(candidate.get("metrics"), reason)
            error = candidate.get("error")
            equations.append(
                f'<details data-equation-id="{batch}-{i}"><summary>Equation {i} · {score}</summary>'
                f"<code>{escape(equation)}</code>{_metrics_detail(candidate.get('metrics'))}"
                + (f"<p>{escape(error)}</p>" if error else "")
                + "</details>"
            )
        if invalid:
            equations.append(f"<small>{invalid} invalid proposal(s)</small>")
        if event.get("error"):
            equations.append(f"<p>{escape(event['error'])}</p>")
        elif not equations:
            equations.append("<p>No equation recorded.</p>")
        cards.append(
            f'<div class="probe"><strong>Batch {batch} · {_mode(event)}</strong>{"".join(equations)}</div>'
        )
    if reference and not any(
        reference["equation"] == _equation(c) for _, e in probes for c in e.get("candidates", [])
    ):
        cards.append(
            f"<p>Reference equation · {_score_label(reference.get('exploration'))}<br>"
            f"<code>{escape(reference['equation'])}</code></p>"
        )
    history = "".join(cards) or "<p>No probes yet.</p>"
    return (
        f'<aside class="inspector" data-idea-detail="{escape(node)}" aria-label="Idea details" hidden>'
        '<button type="button" data-close aria-label="Close idea details">Close</button>'
        f"<h2>{escape(_label(idea))}</h2><p>{escape(idea['text'])}</p>"
        f"<small>{len(probes)} probe(s)</small>{history}</aside>"
    )


def render(wave, step=None, *, fragment=False, snapshot="", layout=""):
    """Render the latest state or replay a completed batch prefix."""
    total = len(wave.get("episodes", []))
    index = total if step is None else min(max(0, step), total)
    episodes = wave.get("episodes", [])[:index]
    live = step is None or not wave
    status = wave.get("status", "waiting")
    final = live and status == "complete"
    graph = {idea["node"]: idea for idea in wave.get("graph", [])} if wave else {BASELINE_ID: baseline_node()}
    nodes = {node: idea for node, idea in graph.items() if idea.get("introduced_at", 0) <= index}
    catalog = {idea["node"]: idea for idea in wave.get("idea_catalog", [])} | graph
    probes = defaultdict(list)
    best = {}
    for batch, event in enumerate(episodes, 1):
        node = event["node"]
        probes[node].append((batch, event))
        for candidate in event.get("candidates", []):
            metrics = candidate.get("metrics")
            if (
                metrics
                and candidate.get("status") == "fitted"
                and (node not in best or metrics["score"] < best[node]["score"])
            ):
                best[node] = metrics
    counts = Counter(
        event["node"]
        for event in episodes
        for candidate in event.get("candidates", [])
        if candidate.get("status") in {"fitted", "fit_failed"} and _equation(candidate)
    )
    if nodes.get(BASELINE_ID, {}).get("equation"):
        counts[BASELINE_ID] = 1
        if wave.get("baseline_exploration"):
            best[BASELINE_ID] = wave["baseline_exploration"]
    active = wave.get("active_probe") if live else None
    selected = active or (episodes[-1] if episodes else {})
    current = selected.get("node", "")
    planned = wave.get("planned_batches", total) if live else total
    progress = f"Batch {index} / {planned}" if planned else f"Batch {index}"
    bar = f'<div class="bar"><strong>{progress}</strong>'
    bar += f'<span class="pill">{escape(status.replace("_", " ")) if live else "replay"}</span>'
    if wave and catalog:
        for target, label in ((0, "Start"), (index - 1, "Previous"), (index + 1, "Next")):
            if 0 <= target <= total and target != index:
                bar += f'<a href="?step={target}">{label}</a>'
        bar += (
            f'<input id="step" aria-label="Replay batch" type="range" min="0" max="{total}" value="{index}">'
            f'<a class="{"active" if live else ""}" href="/">Live / latest</a>'
        )
    if nodes:
        bar += '<button type="button" data-reset>Reset layout</button>'
    if current in catalog:
        label = escape(_label(catalog[current]))
        lead = f"Running batch {active['batch']}" if active else "Latest probe"
        focus = f"{lead}: <strong>{label}</strong> · {_mode(selected)}"
        if selected.get("error"):
            focus += f" · {escape(selected['error'])}"
    else:
        focus = "Waiting for the first batch."
    content = f"<p>{focus}</p>"
    if nodes:
        shortlist = wave.get("shortlist", {}) if final else {}
        inspectors = "".join(
            _inspector(node, idea, probes[node], shortlist.get(node), wave.get("baseline_exploration"))
            for node, idea in nodes.items()
        )
        graph = _graph(nodes, catalog, counts, best, current)
        content += (
            f'<section class="graph-shell">{graph}{inspectors}</section>'
            "<small>Click a node for probes and equations · Drag to move · AND requires every parent fit</small>"
        )
    replay_note = wave.get("replay_note")
    note = f'<p role="note">{escape(replay_note)}</p>' if replay_note else ""
    main = (
        f'<main data-live="{str(live).lower()}" data-snapshot="{escape(snapshot)}" data-layout="{escape(layout)}">'
        "<h1>Knowledge frontier</h1>"
        f"{note}{bar}</div>{content}</main>"
    )
    if fragment:
        return main
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Knowledge frontier</title><style>{STYLE}</style></head><body>{main}"
        f"<script>{SCRIPT}</script></body></html>"
    )


def serve(args):
    """Bind to loopback and serve only the chosen wave, even before it exists."""
    path = args.run if args.run.suffix == ".json" else args.run / "wave.json"
    layout = sha256(os.fsencode(path.resolve())).hexdigest()[:16]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path != "/":
                self.send_error(404)
                return
            query = parse_qs(url.query)
            try:
                step = int(query["step"][0]) if "step" in query else None
            except ValueError:
                self.send_error(400, "Invalid batch number")
                return
            try:
                with path.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                    snapshot = f"{stat.st_ino}-{stat.st_mtime_ns}"
                    wave = json.load(stream)
            except FileNotFoundError:
                wave, snapshot = {}, "waiting"
            except (OSError, ValueError):
                self.send_error(503, "Wave snapshot unavailable")
                return
            content = render(
                wave, step, fragment="fragment" in query, snapshot=snapshot, layout=layout
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()

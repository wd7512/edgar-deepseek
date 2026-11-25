"""Serve a local, read-only replay of a frontier wave."""

import json
import os
from collections import Counter, defaultdict
from contextlib import suppress
from functools import lru_cache
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

COLORS = {
    "blocked": "#657686",
    "ready": "#9ebaf7",
    "attempted": "#ffc977",
    "fitted": "#59d8c6",
    "confirmed": "#ade978",
}

STYLE = """
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101721;color:#edf4fc;
font:13px system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:12px 20px}
h1{font-size:19px;margin:0}h3{font-size:13px}p,small{color:#acbfd0}
.bar,.legend{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:7px 0}
a,.pill{color:#b7ecff;padding:2px 4px;text-decoration:none}a:hover{text-decoration:underline}
a.active{color:#80e8cd;font-weight:700}.panel{border-top:1px solid #344759;padding:7px 0}
.panel p{margin:0 0 5px}.scroll{overflow:auto}
.graph{display:block;width:100%;height:min(var(--graph-h),max(125px,calc(100dvh - 185px)))}
summary{width:fit-content;cursor:pointer;color:#acbfd0}
.node rect{fill:#222f40;stroke-width:2}.node.current rect{stroke-width:4}.node text{fill:#edf4fc;font-weight:650}
.node .sub{fill:#bfd0e1;font-size:11px;font-weight:400}.route path{fill:none;stroke:#526477;stroke-width:1.5}
.route circle{fill:#263647;stroke:#657686}.route text{fill:#e0e8f2;font-size:9px;text-anchor:middle}
.route[data-route-state=ready] path,.route[data-route-state=ready] circle{stroke:#6bcabc}
.key{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--key);margin-right:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:8px;margin-top:8px}
.card{border:1px solid #344759;border-radius:4px;padding:8px;overflow-wrap:anywhere}
.card h3,.card p{margin:0 0 6px}code{color:#c3e9de}input{accent-color:#59d8c6;max-width:150px}
"""

SCRIPT = """
document.addEventListener('change', event => {
  if (event.target.id === 'step') location.search = '?step=' + event.target.value;
});
async function refresh() {
  const current = document.querySelector('main');
  if (current.dataset.live !== 'true') return;
  try {
    const response = await fetch('/?fragment=1', {cache: 'no-store'});
    if (response.ok) {
      const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
      const fresh = doc.querySelector('main');
      if (fresh && fresh.dataset.snapshot !== current.dataset.snapshot) {
        const left = current.querySelector('.scroll')?.scrollLeft || 0;
        const expanded = current.querySelector('details')?.open;
        current.replaceWith(fresh);
        const scroll = fresh.querySelector('.scroll');
        if (scroll) scroll.scrollLeft = left;
        const details = fresh.querySelector('details');
        if (details && expanded) details.open = true;
      }
    }
  } catch {}
  setTimeout(refresh, 2000);
}
setTimeout(refresh, 2000);
"""


def _graph(nodes, fitted, attempted, confirmed, current):
    @lru_cache
    def rank(node):
        return max((rank(p) + 1 for route in nodes[node]["routes"] for p in route), default=0)

    columns = defaultdict(list)
    for node in nodes:
        columns[rank(node)].append(node)
    height = max(map(len, columns.values())) * 48 + 12
    width = 180 + max(columns) * 222
    positions = {
        node: (9 + level * 222, (height - (len(col) - 1) * 48) / 2 + i * 48)
        for level, col in columns.items()
        for i, node in enumerate(col)
    }
    edges, cards = [], []
    for node, (x, y) in positions.items():
        routes = nodes[node]["routes"]
        ready = not routes
        for i, route in enumerate(routes):
            available = all(p in fitted for p in route)
            ready |= available
            rx, ry = x - 32, y + 22 * (i - (len(routes) - 1) / 2)
            label = escape(" + ".join(route) + " → " + node)
            edges.append(
                f'<g class="route" data-route-state="{"ready" if available else "blocked"}"><title>{label}</title>'
            )
            for parent in route:
                px, py = positions[parent]
                mid = (px + 162 + rx) / 2
                edges.append(f'<path d="M{px + 162},{py} C{mid},{py} {mid},{ry} {rx},{ry}"/>')
            edges.append(
                f'<path d="M{rx},{ry} L{x},{y}"/><circle cx="{rx}" cy="{ry}" r="10"/>'
                f'<text x="{rx}" y="{ry + 3}">{"AND" if len(route) > 1 else "1"}</text></g>'
            )
        state = next(
            name
            for name, enabled in (
                ("confirmed", node in confirmed),
                ("fitted", node in fitted),
                ("attempted", node in attempted),
                ("ready", ready),
                ("blocked", True),
            )
            if enabled
        )
        title = escape(node[:22] + ("…" if len(node) > 22 else ""))
        cards.append(
            f'<g class="node{" current" if node == current else ""}" data-node="{escape(node)}" '
            f'data-state="{state}" transform="translate({x},{y})">'
            f"<title>{escape(node)}: {escape(nodes[node]['text'])}</title>"
            f'<rect width="162" height="34" y="-17" rx="4" stroke="{COLORS[state]}"/>'
            f'<text x="8" y="-1" font-size="12">{title}</text><text class="sub" x="8" y="12">{state}</text></g>'
        )
    return (
        f'<svg class="graph" style="--graph-h:{height}px" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMinYMid meet" role="img" aria-label="Prerequisite graph">'
        + "".join(edges + cards)
        + "</svg>"
    )


def render(wave, step=None, *, fragment=False, snapshot=""):
    """Render the latest state or replay a completed batch prefix."""
    total = len(wave.get("episodes", []))
    index = total if step is None else min(max(0, step), total)
    episodes = wave.get("episodes", [])[:index]
    live = step is None
    status = wave.get("status", "waiting")
    final = live and status == "complete"
    nodes = {idea["node"]: idea for idea in wave.get("graph", [])}
    attempted = Counter(event["node"] for event in episodes)
    fitted = Counter(
        event["node"]
        for event in episodes
        for candidate in event.get("candidates", [])
        if candidate.get("status") == "fitted"
    )
    confirmed = {key for key, state in wave.get("claims", {}).items() if final and state == "supported_fit"}
    selected = episodes[-1] if episodes else {}
    current = selected.get("node", "")
    bar = f'<div class="bar"><strong>Batch {index} / {total}</strong>'
    bar += f'<span class="pill">{escape(status.replace("_", " ")) if live else "replay"}</span>'
    if nodes:
        for target, label in ((0, "Start"), (index - 1, "Previous"), (index + 1, "Next")):
            if 0 <= target <= total and target != index:
                bar += f'<a href="?step={target}">{label}</a>'
        bar += (
            f'<input id="step" aria-label="Replay batch" type="range" min="0" max="{total}" value="{index}">'
            f'<a class="{"active" if live else ""}" href="/">Live / latest</a>'
        )
    legend = "".join(
        f'<span class="pill"><i class="key" style="--key:{color}"></i>{state.title()}</span>'
        for state, color in COLORS.items()
    )
    cards = []
    for node, idea in nodes.items():
        equation = wave.get("shortlist", {}).get(node, {}).get("equation") if final else None
        cards.append(
            f'<article class="card"><h3>{escape(node)}</h3><p>{escape(idea["text"])}</p>'
            f"<small>{attempted[node]} batches · {fitted[node]} fitted proposals</small>"
            + (f"<p><code>{escape(equation)}</code></p>" if equation else "")
            + "</article>"
        )
    content = "<p>Waiting for the first batch. This page updates automatically.</p>"
    if nodes:
        focus = (
            f"Latest in this view: <strong>{escape(current)}</strong> · {escape(selected.get('lane', ''))}"
            f" · {int(selected.get('images', 0))} attachments"
            if current
            else "Starting graph, before any batches."
        )
        content = (
            f'<section class="panel"><p>{focus}</p><div class="legend">{legend}</div>'
            f'<div class="scroll">{_graph(nodes, fitted, attempted, confirmed, current)}</div>'
            "<p><small>AND needs every incoming parent fit. Multiple routes are alternatives. "
            "Ready is not confirmed.</small></p></section>"
            '<details class="panel"><summary>Idea details</summary><div class="cards">'
            + "".join(cards)
            + "</div></details>"
        )
    main = (
        f'<main data-live="{str(live).lower()}" data-snapshot="{escape(snapshot)}">'
        "<h1>Knowledge frontier</h1>"
        f"{bar}</div>{content}</main>"
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
            content = render(wave, step, fragment="fragment" in query, snapshot=snapshot).encode()
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

"""Serve a local, read-only replay of a frontier wave."""

import json
import os
from collections import Counter, defaultdict, deque
from contextlib import suppress
from hashlib import sha256
from html import escape
from math import sqrt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from textwrap import wrap
from urllib.parse import parse_qs, urlsplit

from edgar_study.frontier.seed import BASELINE_ID, baseline_node

MODES = {
    "single_idea": "Single-idea frontier",
    "combination": "AND combination",
    "revisit": "Revisit",
    "anomaly_detour": "Anomaly detour",
    "independent_root": "Independent root",
}

NODE_W, NODE_H, GAP, WIDTH = 192, 68, 10, 1000

CANVAS_AT = 250

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
.node.lowest rect{fill:#1f3a3c;stroke:#6fcfb8;stroke-opacity:.75;filter:drop-shadow(0 0 5px #59d8c666)}
.node.current rect,.node:focus-visible rect{stroke:#b7ecff;stroke-width:2.5}
.node text{fill:#edf4fc;font-size:12px;font-weight:650;text-anchor:middle;pointer-events:none;user-select:none}
.node .sub{fill:#bfd0e1;font-size:10px;font-weight:400}.route path{fill:none;stroke:#657686;stroke-width:1.5}
.route circle{fill:#263647;stroke:#657686}.route text{fill:#e0e8f2;font-size:9px;text-anchor:middle}
.graph{touch-action:none;cursor:grab}.graph.panning{cursor:grabbing}.zoom{display:inline-flex;gap:4px}
.zoom button{min-width:28px}.legend::before{content:"";display:inline-block;width:10px;height:10px;margin-right:5px;
border:1px solid #6fcfb8;background:#1f3a3c;border-radius:2px;vertical-align:-1px}
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
const NODE_W = 192, NODE_H = 68, PAD = 18, MIN_K = 0.01, MAX_K = 4, REST = 270, DROP = 150, SMALL = 300;
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
let positions = new Map(), bodies = new Map(), list = [], links = [], center = [500, 300];
let dragging, panning, selected, hover, canvasMode = false, userView = false;
let view = {k: 1, x: 0, y: 0}, alpha = 0, target = 0, frame = 0, drawQueued = false, tickMs = 1, ticks = 0;
try { positions = new Map(JSON.parse(localStorage.getItem(storageKey) || '[]')); } catch {}
function save() {
  positions = new Map(list.map(b => [b.id, [b.x, b.y]]));
  try { localStorage.setItem(storageKey, JSON.stringify([...positions])); } catch {}
}
function extent() {
  const el = $('.graph');
  return canvasMode ? {width: el.clientWidth, height: el.clientHeight} : el.viewBox.baseVal;
}
function applyView() {
  if (canvasMode) { requestDraw(); return; }
  for (const g of $$('.viewport')) g.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
}
let goal, gliding = false, lastGlide = 0;
function glide(next) {
  userView = next.keep ?? true;
  goal = next;
  if (calm) { view = {k: goal.k, x: goal.x ?? goal.ax - goal.wx * goal.k, y: goal.y ?? goal.ay - goal.wy * goal.k}; goal = undefined; applyView(); return; }
  if (gliding) return;
  gliding = true;
  lastGlide = performance.now();
  requestAnimationFrame(glideStep);
}
function glideStep(now) {
  if (!goal) { gliding = false; return; }
  const f = 1 - Math.exp(-Math.min(50, now - lastGlide) / 85);
  lastGlide = now;
  const done = Math.abs(Math.log(goal.k / view.k)) < 0.002;
  const k = done ? goal.k : Math.exp(Math.log(view.k) + (Math.log(goal.k) - Math.log(view.k)) * f);
  if (goal.ax !== undefined) view = {k, x: goal.ax - goal.wx * k, y: goal.ay - goal.wy * k};
  else view = {k, x: done ? goal.x : view.x + (goal.x - view.x) * f, y: done ? goal.y : view.y + (goal.y - view.y) * f};
  applyView();
  if (done) { goal = undefined; gliding = false; } else requestAnimationFrame(glideStep);
}
function zoomAt(factor, cx, cy) {
  const k = Math.max(MIN_K, Math.min(MAX_K, (goal?.k ?? view.k) * factor));
  glide({k, ax: cx, ay: cy, wx: (cx - view.x) / view.k, wy: (cy - view.y) / view.k});
}
function fit(instant = false) {
  if (!list.length || !$('.graph')) return;
  const box = extent();
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const b of list) { x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x); y0 = Math.min(y0, b.y); y1 = Math.max(y1, b.y); }
  x0 -= NODE_W / 2; x1 += NODE_W / 2; y0 -= NODE_H / 2; y1 += NODE_H / 2;
  const k = Math.max(MIN_K, Math.min(1.5, 0.94 * Math.min(box.width / (x1 - x0), box.height / (y1 - y0))));
  const next = {k, x: box.width / 2 - k * (x0 + x1) / 2, y: box.height / 2 - k * (y0 + y1) / 2};
  if (instant) { view = next; applyView(); } else glide({...next, keep: false});
}
function viewPoint(event, el) {
  if (!canvasMode) return new DOMPoint(event.clientX, event.clientY).matrixTransform(el.getScreenCTM().inverse());
  const r = el.getBoundingClientRect();
  return {x: event.clientX - r.left, y: event.clientY - r.top};
}
function worldPoint(event, el) {
  const p = viewPoint(event, el);
  return {x: (p.x - view.x) / view.k, y: (p.y - view.y) / view.k};
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
    const mid = parents.reduce(([x, y], [px, py]) => [x + px / parents.length, y + py / parents.length], [0, 0]);
    let dx = mid[0] - child[0], dy = mid[1] - child[1];
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

const COLORS = {fill: '#222f40', stroke: '#899aae', lowFill: '#1f3a3c', lowStroke: '#6fcfb8', link: '#657686', ink: '#edf4fc', soft: '#bfd0e1', ring: '#b7ecff'};
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}
function card(ctx, x, y, w, h, r) {
  if (ctx.roundRect) ctx.roundRect(x - w / 2, y - h / 2, w, h, r);
  else ctx.rect(x - w / 2, y - h / 2, w, h);
}
let fullTimer = 0;
function draw(forceFull = false) {
  const cv = $('canvas.graph');
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d'), k = view.k;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * view.x, dpr * view.y);
  const m = 120, vx0 = -view.x / k - m, vx1 = (w - view.x) / k + m, vy0 = -view.y / k - m, vy1 = (h - view.y) / k + m;
  const mode = k >= 0.5 ? 'card' : k >= 0.14 ? 'box' : 'dot';
  const cheap = !forceFull && list.length > 3000 && (gliding || panning || dragging || frame);
  clearTimeout(fullTimer);
  if (cheap) fullTimer = setTimeout(() => draw(true), 140);
  const dim = hover ? 0.22 : 1;
  const visible = [], buckets = [[], [], []];
  for (const b of list) {
    if (b.x <= vx0 || b.x >= vx1 || b.y <= vy0 || b.y >= vy1) continue;
    visible.push(b);
    buckets[b.current ? 2 : b.lowest.length ? 1 : 0].push(b);
  }
  let shown = 0;
  for (const [p, c] of links) {
    if (Math.max(p.x, c.x) >= vx0 && Math.min(p.x, c.x) <= vx1 && Math.max(p.y, c.y) >= vy0 && Math.min(p.y, c.y) <= vy1) shown++;
  }
  const stride = cheap ? Math.max(1, Math.ceil(shown / 4000)) : 1;
  const lineAlpha = Math.max(0.035, Math.min(0.55, 0.55 * Math.sqrt(600 / Math.max(1, shown)))) * dim;
  ctx.lineWidth = Math.max(0.8, 1.3 * Math.min(1, k * 2)) / k;
  ctx.strokeStyle = COLORS.link;
  ctx.fillStyle = COLORS.link;
  const strokeLinks = (pick, alphaValue) => {
    ctx.globalAlpha = alphaValue;
    ctx.beginPath();
    const heads = new Path2D();
    let seen = 0;
    for (const [p, c] of links) {
      if (!pick(p, c)) continue;
      if (Math.max(p.x, c.x) < vx0 || Math.min(p.x, c.x) > vx1 || Math.max(p.y, c.y) < vy0 || Math.min(p.y, c.y) > vy1) continue;
      if (stride > 1 && seen++ % stride) continue;
      if (mode === 'card') {
        const a = edge([p.x, p.y], [c.x, c.y], 96, 34), z = edge([c.x, c.y], [p.x, p.y], 96, 34, 3);
        ctx.moveTo(...a); ctx.lineTo(...z);
        const ang = Math.atan2(z[1] - a[1], z[0] - a[0]);
        heads.moveTo(z[0], z[1]);
        heads.lineTo(z[0] - 9 * Math.cos(ang - 0.4), z[1] - 9 * Math.sin(ang - 0.4));
        heads.lineTo(z[0] - 9 * Math.cos(ang + 0.4), z[1] - 9 * Math.sin(ang + 0.4));
        heads.closePath();
      } else { ctx.moveTo(p.x, p.y); ctx.lineTo(c.x, c.y); }
    }
    ctx.stroke();
    if (mode === 'card') ctx.fill(heads);
  };
  strokeLinks(() => true, lineAlpha);
  const near = hover ? new Set([hover, ...hover.nb]) : null;
  if (hover) {
    ctx.strokeStyle = COLORS.ring; ctx.fillStyle = COLORS.ring;
    strokeLinks((p, c) => p === hover || c === hover, 0.9);
  }
  const groups = [[0, COLORS.fill, COLORS.stroke], [1, COLORS.lowFill, COLORS.lowStroke], [2, COLORS.fill, COLORS.ring]];
  const r = Math.max(2, Math.min(8, 24 * k)) / k;
  const square = mode === 'dot' && (cheap || r * k < 2.8);
  for (const [index, fill, stroke] of groups) {
    const set = near ? buckets[index].filter(b => !near.has(b)) : buckets[index];
    ctx.globalAlpha = dim;
    ctx.fillStyle = fill; ctx.strokeStyle = stroke;
    ctx.lineWidth = (mode === 'dot' ? 1 : 1.5) / k;
    if (square) {
      for (const b of set) ctx.fillRect(b.x - r, b.y - r, 2 * r, 2 * r);
      continue;
    }
    ctx.beginPath();
    for (const b of set) {
      if (mode === 'dot') { ctx.moveTo(b.x + r, b.y); ctx.arc(b.x, b.y, r, 0, 6.2832); }
      else card(ctx, b.x, b.y, NODE_W, NODE_H, 5);
    }
    ctx.fill(); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  if (mode === 'dot' && !cheap) {
    ctx.fillStyle = 'rgba(111,207,184,0.18)';
    ctx.beginPath();
    for (const b of visible) if (b.lowest.length && !(near && near.has(b))) { ctx.moveTo(b.x + r * 2.4, b.y); ctx.arc(b.x, b.y, r * 2.4, 0, 6.2832); }
    ctx.globalAlpha = dim; ctx.fill(); ctx.globalAlpha = 1;
  }
  ctx.textAlign = 'center';
  if (mode === 'card') {
    ctx.globalAlpha = dim;
    for (const b of visible) if (!(near && near.has(b))) label(ctx, b);
    ctx.globalAlpha = 1;
  }
  if (near) {
    for (const b of near) {
      if (b.x < vx0 || b.x > vx1 || b.y < vy0 || b.y > vy1) continue;
      ctx.fillStyle = b.lowest.length ? COLORS.lowFill : COLORS.fill;
      ctx.strokeStyle = b === hover ? COLORS.ring : b.lowest.length ? COLORS.lowStroke : COLORS.stroke;
      ctx.lineWidth = (b === hover ? 2.5 : 1.5) / k;
      ctx.beginPath();
      if (mode === 'dot') ctx.arc(b.x, b.y, r * 1.3, 0, 6.2832); else card(ctx, b.x, b.y, NODE_W, NODE_H, 5);
      ctx.fill(); ctx.stroke();
      if (mode === 'card') label(ctx, b);
      else tag(ctx, b, mode === 'dot' ? r * 1.3 : NODE_H / 2);
    }
  }
}
function label(ctx, b) {
  ctx.fillStyle = COLORS.ink;
  ctx.font = '650 12px system-ui, sans-serif';
  b.lines.forEach((line, i) => ctx.fillText(line, b.x, b.y + 13 * (i - (b.lines.length - 1) / 2) - 2));
  ctx.fillStyle = COLORS.soft;
  ctx.font = '400 10px system-ui, sans-serif';
  ctx.fillText(b.sub, b.x, b.y + 28);
}
function tag(ctx, b, rise) {
  const k = view.k, size = 12 / k, text = b.lines.join(' ');
  ctx.font = `650 ${size}px system-ui, sans-serif`;
  const width = ctx.measureText(text).width + 10 / k, y = b.y - rise - 10 / k;
  ctx.fillStyle = 'rgba(16,23,33,0.9)';
  ctx.fillRect(b.x - width / 2, y - size, width, size * 1.5);
  ctx.fillStyle = COLORS.ink;
  ctx.fillText(text, b.x, y);
}

function build() {
  const previous = bodies;
  bodies = new Map();
  const seeds = positions, made = [];
  const add = (id, x, y, extra) => {
    const old = previous.get(id), seed = seeds.get(id);
    const [px, py] = old ? [old.x, old.y] : seed || [x, y];
    const body = Object.assign({id, x: px + (made.length % 3) * 0.01, y: py + (made.length % 5) * 0.01, vx: 0, vy: 0, fixed: false, nb: [],
      fresh: !old && !seed, lowest: [], lines: [], sub: '', current: false}, extra);
    bodies.set(id, body);
    made.push(body);
  };
  if (canvasMode) {
    const data = JSON.parse($('.graph-data').textContent);
    for (const n of data.nodes) add(n.id, n.x, n.y, {lowest: n.lowest, lines: n.lines, sub: n.sub, current: n.current, tip: n.tip});
    links = data.links.map(([p, c]) => [made[p], made[c]]);
  } else {
    for (const el of $$('.node')) add(el.dataset.node, +el.dataset.homeX, +el.dataset.homeY, {el});
    links = [];
    for (const route of $$('.route')) for (const path of $$('path[data-parent]', route)) {
      const pair = [path.dataset.parent, route.dataset.child];
      if (pair.every(id => bodies.has(id))) links.push(pair.map(id => bodies.get(id)));
    }
  }
  list = made;
  for (const [p, c] of links) { p.nb.push(c); c.nb.push(p); }
  hover = undefined;
}

let T = {cap: 0};
function allocTree(cap) {
  T = {cap, n: 0, x: new Float64Array(cap), y: new Float64Array(cap), h: new Float64Array(cap), cx: new Float64Array(cap),
    cy: new Float64Array(cap), m: new Float64Array(cap), b: new Int32Array(cap), k: new Int32Array(cap * 4), stack: new Int32Array(512)};
}
function treeNode(x, y, h) {
  const id = T.n++;
  T.x[id] = x; T.y[id] = y; T.h[id] = h; T.cx[id] = 0; T.cy[id] = 0; T.m[id] = 0; T.b[id] = -1;
  T.k.fill(-1, id * 4, id * 4 + 4);
  return id;
}
function buildTree() {
  const n = list.length;
  if (T.cap < 12 * n + 64) allocTree(12 * n + 64);
  T.n = 0;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const b of list) { x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x); y0 = Math.min(y0, b.y); y1 = Math.max(y1, b.y); }
  treeNode((x0 + x1) / 2, (y0 + y1) / 2, Math.max(x1 - x0, y1 - y0) / 2 + 1);
  const quad = (node, px, py) => (px >= T.x[node] ? 1 : 0) | (py >= T.y[node] ? 2 : 0);
  const child = (node, q) => {
    let c = T.k[node * 4 + q];
    if (c < 0) {
      const hh = T.h[node] / 2;
      c = treeNode(T.x[node] + (q & 1 ? hh : -hh), T.y[node] + (q & 2 ? hh : -hh), hh);
      T.k[node * 4 + q] = c;
    }
    return c;
  };
  for (let i = 0; i < n; i++) {
    const px = list[i].x, py = list[i].y;
    let node = 0, depth = 0;
    for (;;) {
      const m = T.m[node];
      T.cx[node] = (T.cx[node] * m + px) / (m + 1);
      T.cy[node] = (T.cy[node] * m + py) / (m + 1);
      T.m[node] = m + 1;
      if (m === 0) { T.b[node] = i; break; }
      if (depth >= 30 || T.n > T.cap - 8) { T.b[node] = -1; break; }
      if (T.b[node] >= 0) {
        const j = T.b[node], low = list[j], c = child(node, quad(node, low.x, low.y));
        T.b[node] = -1;
        T.b[c] = j; T.m[c] = 1; T.cx[c] = low.x; T.cy[c] = low.y;
      }
      node = child(node, quad(node, px, py));
      depth++;
    }
  }
}
function repel(a) {
  buildTree();
  const stack = T.stack, strength = 45000 * a;
  for (let i = 0; i < list.length; i++) {
    const body = list[i];
    if (body.fixed) continue;
    let fx = 0, fy = 0, sp = 1;
    stack[0] = 0;
    while (sp) {
      const node = stack[--sp], base = node * 4;
      let dx = T.cx[node] - body.x, dy = T.cy[node] - body.y, d2 = dx * dx + dy * dy;
      const leaf = T.k[base] < 0 && T.k[base + 1] < 0 && T.k[base + 2] < 0 && T.k[base + 3] < 0;
      if (leaf) { if (T.b[node] === i) continue; }
      else if (4 * T.h[node] * T.h[node] > 0.81 * d2) {
        for (let q = 0; q < 4; q++) if (T.k[base + q] >= 0) stack[sp++] = T.k[base + q];
        continue;
      }
      if (d2 < 1) { dx = 1; dy = 0.5; d2 = 1.25; }
      const d = Math.sqrt(d2), f = strength * T.m[node] / d2;
      fx -= dx / d * f; fy -= dy / d * f;
    }
    const mag = Math.hypot(fx, fy), cap = 40;
    if (mag > cap) { fx *= cap / mag; fy *= cap / mag; }
    body.vx += fx; body.vy += fy;
  }
}
function tick(a) {
  const t0 = performance.now(), push = (b, fx, fy) => { if (!b.fixed) { b.vx += fx; b.vy += fy; } };
  repel(a);
  for (const [p, c] of links) {
    const dx = c.x - p.x, dy = c.y - p.y, d = Math.hypot(dx, dy) || 1, f = Math.max(-10, Math.min(10, (d - REST) * 0.02)) * a;
    push(p, dx / d * f, dy / d * f);
    push(c, -dx / d * f, -dy / d * f);
    const short = DROP - dy;
    if (short > 0) { push(c, 0, short * 0.04 * a); push(p, 0, -short * 0.04 * a); }
  }
  if (list.length <= SMALL) {
    for (const [p, c] of links) {
      const sx = c.x - p.x, sy = c.y - p.y, len2 = sx * sx + sy * sy || 1;
      for (const m of list) {
        if (m === p || m === c) continue;
        const t = ((m.x - p.x) * sx + (m.y - p.y) * sy) / len2;
        if (t < 0.02 || t > 0.98) continue;
        const ox = m.x - (p.x + sx * t), oy = m.y - (p.y + sy * t);
        const hit = Math.abs(ox) / (NODE_W / 2 + 22), vit = Math.abs(oy) / (NODE_H / 2 + 22);
        if (Math.max(hit, vit) >= 1) continue;
        const d = Math.hypot(ox, oy) || 1, f = (1 - Math.max(hit, vit)) * 14 * a;
        push(m, ox / d * f, oy / d * f);
        push(p, -ox / d * f * 0.5, -oy / d * f * 0.5);
        push(c, -ox / d * f * 0.5, -oy / d * f * 0.5);
      }
    }
  }
  for (const b of list) {
    const gx = center[0] - b.x, gy = center[1] - b.y, gd = Math.hypot(gx, gy) || 1, pull = Math.min(5, gd * 0.006) * a;
    push(b, gx / gd * pull, gy / gd * pull);
    if (b.fixed) { b.vx = b.vy = 0; continue; }
    b.vx *= 0.62; b.vy *= 0.62;
    b.x += b.vx; b.y += b.vy;
  }
  separate();
  tickMs = 0.7 * tickMs + 0.3 * (performance.now() - t0);
}
function separate(passes = list.length > SMALL ? 2 : 4) {
  const cw = NODE_W + PAD, ch = NODE_H + PAD;
  const shove = (p, q) => {
    const dx = q.x - p.x, dy = q.y - p.y, ox = cw - Math.abs(dx), oy = ch - Math.abs(dy);
    if (ox <= 0 || oy <= 0) return false;
    const share = p.fixed ? [0, 1] : q.fixed ? [1, 0] : [0.5, 0.5];
    if (ox < oy) { const s = dx < 0 ? -1 : 1; p.x -= s * ox * share[0]; q.x += s * ox * share[1]; }
    else { const s = dy < 0 ? -1 : 1; p.y -= s * oy * share[0]; q.y += s * oy * share[1]; }
    return true;
  };
  for (let pass = 0; pass < passes; pass++) {
    const grid = new Map();
    for (const b of list) {
      b.gx = Math.floor(b.x / cw); b.gy = Math.floor(b.y / ch);
      const key = b.gx * 200003 + b.gy, cell = grid.get(key);
      if (cell) cell.push(b); else grid.set(key, [b]);
    }
    let moved = false;
    for (const cell of grid.values()) {
      const {gx, gy} = cell[0];
      for (let i = 0; i < cell.length; i++) for (let j = i + 1; j < cell.length; j++) moved = shove(cell[i], cell[j]) || moved;
      for (const [ox, oy] of [[1, 0], [-1, 1], [0, 1], [1, 1]]) {
        const other = grid.get((gx + ox) * 200003 + gy + oy);
        if (other) for (const p of cell) for (const q of other) moved = shove(p, q) || moved;
      }
    }
    if (!moved) break;
  }
}
function settle(start = 1, count = 300) {
  let a = start;
  for (let i = 0; i < count; i++) { tick(a); a = Math.max(0.02, a * 0.985); }
  for (const b of list) b.vx = b.vy = 0;
  separate(40);
  render();
}
function render() {
  if (canvasMode) { draw(); return; }
  for (const b of list) setPosition(b.el, Math.round(b.x * 10) / 10, Math.round(b.y * 10) / 10);
  drawArrows();
}
function wake(level = 0.3) {
  alpha = Math.max(alpha, level);
  if (calm || frame) return;
  const step = () => {
    const per = list.length <= SMALL ? 1 : Math.max(1, Math.min(3, Math.floor(10 / tickMs)));
    for (let i = 0; i < per; i++) { alpha = Math.max(target, alpha * 0.975); tick(alpha); }
    render();
    if (!userView && ++ticks % 8 === 0) fit();
    frame = alpha > 0.004 || target ? requestAnimationFrame(step) : 0;
    if (!frame) { for (const b of list) b.vx = b.vy = 0; separate(40); render(); if (!userView) fit(); save(); }
  };
  frame = requestAnimationFrame(step);
}
function applyLayout(reset = false) {
  canvasMode = !!$('canvas.graph');
  const had = (positions.size > 0 || bodies.size > 0) && !reset;
  if (reset) { positions.clear(); bodies = new Map(); userView = false; }
  build();
  if (!had && list.length) center = [list.reduce((s, b) => s + b.x, 0) / list.length, list.reduce((s, b) => s + b.y, 0) / list.length];
  const added = list.some(b => b.fresh);
  const large = list.length > SMALL;
  for (const b of list) b.fresh = false;
  if (!had || added) {
    if (large && !calm) { render(); wake(had ? 0.4 : 0.6); }
    else if (large) settle(0.4, 40);
    else if (!had) settle(1, 300);
    else settle(0.5, 160);
  } else render();
  if (!had || !userView) fit(!had);
  applyView();
  if (!large) save();
}
async function inspect(id, expanded = new Set()) {
  if (canvasMode && id && !$$('.inspector').some(el => el.dataset.ideaDetail === id)) {
    try {
      const response = await fetch(`/?inspect=${encodeURIComponent(id)}&${location.search.slice(1)}`, {cache: 'no-store'});
      if (response.ok) {
        for (const old of $$('.inspector')) old.remove();
        $('.graph-shell').insertAdjacentHTML('beforeend', await response.text());
      }
    } catch {}
  }
  const detail = $$('.inspector').find(el => el.dataset.ideaDetail === id);
  for (const el of $$('.inspector')) el.hidden = el !== detail;
  selected = detail?.dataset.ideaDetail;
  if (detail) for (const item of $$('details', detail)) item.open ||= expanded.has(item.dataset.equationId);
}
function hit(world, el) {
  if (canvasMode) {
    const k = view.k, mode = k >= 0.14 ? 'box' : 'dot', radius = Math.max(2, Math.min(8, 24 * k)) / k + 3 / k;
    for (let i = list.length - 1; i >= 0; i--) {
      const b = list[i];
      if (mode === 'dot' ? Math.hypot(b.x - world.x, b.y - world.y) <= radius
        : Math.abs(b.x - world.x) <= NODE_W / 2 && Math.abs(b.y - world.y) <= NODE_H / 2) return b;
    }
    return undefined;
  }
  return bodies.get(el.closest('.node')?.dataset.node);
}
document.addEventListener('wheel', event => {
  const graph = event.target.closest('.graph');
  if (!graph) return;
  event.preventDefault();
  const p = viewPoint(event, graph);
  zoomAt(Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.0015)), p.x, p.y);
}, {passive: false});
document.addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0) return;
  const graph = event.target.closest('.graph');
  if (!graph) return;
  const body = hit(worldPoint(event, graph), event.target);
  if (!body) {
    panning = {graph, start: viewPoint(event, graph), x: view.x, y: view.y, id: event.pointerId};
    graph.setPointerCapture(event.pointerId);
    graph.classList.add('panning');
    return;
  }
  dragging = {body, graph, start: worldPoint(event, graph), x: body.x, y: body.y, id: event.pointerId, moved: false};
  graph.setPointerCapture(event.pointerId);
});
document.addEventListener('pointermove', event => {
  if (panning?.id === event.pointerId) {
    const p = viewPoint(event, panning.graph);
    view.x = panning.x + p.x - panning.start.x;
    view.y = panning.y + p.y - panning.start.y;
    userView = true;
    goal = undefined;
    applyView();
    return;
  }
  if (dragging?.id === event.pointerId) {
    const {body, graph, start, x, y} = dragging, p = worldPoint(event, graph);
    if (!dragging.moved && Math.hypot((p.x - start.x) * view.k, (p.y - start.y) * view.k) < 4) return;
    if (!dragging.moved) { dragging.moved = true; body.fixed = true; target = 0.25; }
    body.x = x + p.x - start.x;
    body.y = y + p.y - start.y;
    if (calm) { separate(); render(); } else wake(0.3);
    return;
  }
  const graph = event.target.closest?.('canvas.graph');
  if (canvasMode && graph) {
    const found = hit(worldPoint(event, graph), event.target);
    if (found !== hover) {
      hover = found;
      graph.title = found ? `${found.tip}\n${found.sub}` : '';
      graph.style.cursor = found ? 'pointer' : '';
      requestDraw();
    }
  }
});
function finishDrag(event) {
  if (panning?.id === event.pointerId) {
    panning.graph.classList.remove('panning');
    panning = undefined;
    return;
  }
  if (dragging?.id !== event.pointerId) return;
  const {body, moved} = dragging;
  dragging = undefined;
  target = 0;
  if (moved) {
    body.fixed = false;
    wake(0.2);
    save();
  } else if (event.type === 'pointerup') {
    inspect(body.id === selected ? undefined : body.id);
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
  const zoom = event.target.closest('[data-zoom]');
  if (zoom) {
    const box = extent();
    if (zoom.dataset.zoom === 'fit') { userView = false; fit(); }
    else zoomAt(zoom.dataset.zoom === 'in' ? 1.25 : 0.8, box.width / 2, box.height / 2);
  }
  if (event.target.closest('[data-reset]')) applyLayout(true);
  if (event.target.closest('[data-close]')) closeInspector();
});
document.addEventListener('change', event => {
  if (event.target.id === 'step') location.search = '?step=' + event.target.value;
});
window.addEventListener('resize', () => canvasMode && requestDraw());
async function refresh() {
  const current = $('main');
  if (current.dataset.live !== 'true') return;
  try {
    const probe = await fetch('/?snapshot=1', {cache: 'no-store'});
    if (probe.ok && !dragging && (await probe.text()) !== current.dataset.snapshot) {
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
          await inspect(selected, expanded);
          const detail = $('.inspector:not([hidden])');
          if (detail) detail.scrollTop = top;
        }
      }
    }
  } catch {}
  setTimeout(refresh, 2000);
}
applyLayout();
setTimeout(refresh, 2000);
"""


def _layout(catalog):
    """Seed positions: one row per depth, ordered by parent position to limit crossings."""
    ids = list(catalog)
    known = set(ids)
    parents = {node: [] for node in ids}
    children = {node: [] for node in ids}
    for node, idea in catalog.items():
        for route in idea.get("routes", []):
            if not all(p in known for p in route):
                continue
            for parent in route:
                if parent != node and parent not in parents[node]:
                    parents[node].append(parent)
                    children[parent].append(node)
    depth = {}
    waiting = {node: len(parents[node]) for node in ids}
    queue = deque(node for node in ids if not waiting[node])
    depth.update(dict.fromkeys(queue, 0))
    while queue:
        node = queue.popleft()
        for child in children[node]:
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            waiting[child] -= 1
            if not waiting[child]:
                queue.append(child)
    loose = max(depth.values(), default=-1) + 1
    layers = defaultdict(list)
    for node in ids:
        layers[depth.get(node, loose)].append(node)
    rank = {}

    def reindex(layer):
        for i, node in enumerate(layer):
            rank[node] = (i + 0.5) / len(layer)

    def order(layer, around):
        layer.sort(key=lambda n: sum(rank[o] for o in around[n]) / len(around[n]) if around[n] else rank[n])
        reindex(layer)

    for layer in layers.values():
        reindex(layer)
    last = max(layers, default=0)
    for _ in range(4):
        for d in range(1, last + 1):
            order(layers[d], parents)
        for d in range(last - 1, -1, -1):
            order(layers[d], children)
    columns = max(5, round(sqrt(len(ids)) * 1.3))
    step, row_height = NODE_W + GAP + 8, NODE_H + 70
    positions, row = {}, 0
    for d in range(last + 1):
        layer = layers.get(d, [])
        for start in range(0, len(layer), columns):
            chunk = layer[start : start + columns]
            for j, node in enumerate(chunk):
                positions[node] = (
                    round(WIDTH / 2 + (j - (len(chunk) - 1) / 2) * step),
                    NODE_H // 2 + 8 + row * row_height,
                )
            row += 1
    return positions, max(460, NODE_H // 2 + 28 + row * row_height)


def _label(idea):
    return idea.get("label") or idea["text"]


def _canvas(nodes, counts, best, current, lowest, positions):
    index = {node: i for i, node in enumerate(nodes)}
    records, links, seen = [], [], set()
    for node, idea in nodes.items():
        x, y = positions[node]
        count = counts[node]
        records.append(
            {
                "id": node,
                "lines": wrap(_label(idea), width=29, max_lines=3, placeholder="…"),
                "sub": _metrics_line(best.get(node)),
                "tip": f"{_label(idea)} · {count} equation{'s' if count != 1 else ''}",
                "lowest": sorted(lowest.get(node, ())),
                "current": node == current,
                "x": x,
                "y": y,
            }
        )
        for route in idea["routes"]:
            if all(p in nodes for p in route):
                for parent in route:
                    if (pair := (index[parent], index[node])) not in seen:
                        seen.add(pair)
                        links.append(list(pair))
    blob = json.dumps({"nodes": records, "links": links}, separators=(",", ":")).replace("</", "<\\/")
    return (
        '<canvas class="graph" style="--graph-h:2000px" role="img" '
        'aria-label="Prerequisite graph; scroll to zoom, drag to pan, click a node for details"></canvas>'
        f'<script type="application/json" class="graph-data">{blob}</script>'
    )


def _graph(nodes, catalog, counts, best, current, lowest):
    positions, height = _layout(catalog)
    if len(nodes) > CANVAS_AT:
        return _canvas(nodes, counts, best, current, lowest, positions)
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
        wins = sorted(lowest.get(node, ()))
        note = " Lowest MSE for " + ", ".join(f"{n} parameters" for n in wins) + "." if wins else ""
        mark = f' data-lowest-mse="{",".join(map(str, wins))}"' if wins else ""
        cards.append(
            f'<g class="node{" current" if node == current else ""}{" lowest" if wins else ""}" '
            f'data-node="{escape(node)}"{mark} '
            f'data-equations="{count}" data-home-x="{x}" data-home-y="{y}" '
            f'transform="translate({x},{y})" tabindex="0" role="button" '
            f'aria-label="{escape(title)}: {summary}; {metrics}.{note}">'
            f"<title>{escape(idea['text'])} {summary}. MSE is unpenalized and measured on held-out trials.{note}</title>"
            '<rect width="192" height="68" x="-96" y="-34" rx="5"/>'
            f'{labels}<text class="sub" y="25">{metrics}</text></g>'
        )
    return (
        f'<svg class="graph" style="--graph-h:{height}px" viewBox="0 0 1000 {height}" '
        'role="group" aria-label="Prerequisite graph; scroll to zoom, drag the background to pan">'
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        'orient="auto"><path d="M0 0L8 4L0 8Z" style="fill:context-stroke"/></marker></defs>'
        + '<g class="viewport">'
        + "".join(edges + cards)
        + "</g></svg>"
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
        train = metrics.get("training_mse")
        train_text = f" · Training MSE: {train:.6g}" if train is not None else ""
        return f"<p>Held-out MSE: {metrics['mse']:.6g}{train_text} · Free parameters: {metrics['parameters']}</p>"
    return ""


def _lowest_mse(episodes, nodes, baseline):
    """Map each node to the parameter counts for which it holds the lowest held-out MSE."""
    entries = []
    if BASELINE_ID in nodes and baseline and "mse" in baseline and "parameters" in baseline:
        entries.append((BASELINE_ID, baseline["parameters"], baseline["mse"]))
    for event in episodes:
        for candidate in event.get("candidates", []):
            metrics = candidate.get("metrics")
            if candidate.get("status") == "fitted" and metrics and "mse" in metrics and "parameters" in metrics:
                entries.append((event["node"], metrics["parameters"], metrics["mse"]))
    floor = {}
    for _, count, mse in entries:
        floor[count] = min(mse, floor.get(count, mse))
    owners = defaultdict(set)
    for node, count, mse in entries:
        if mse == floor[count]:
            owners[node].add(count)
    return owners


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
        for i, candidate in enumerate(event.get("candidates", []), 1):
            equation = _equation(candidate)
            label = f"Equation {i}"
            if candidate.get("source") == "retry":
                position = candidate.get("replaces")
                label = (
                    f"Replacement for proposal {position}" if position is not None else f"Retry equation {i}"
                )
            if candidate.get("status") == "invalid":
                proposal = candidate.get("proposal")
                rejected = proposal.get("equation") if isinstance(proposal, dict) else None
                source = f"<code>{escape(rejected)}</code>" if isinstance(rejected, str) else ""
                reason = escape(candidate.get("reason_code") or "invalid")
                error = escape(candidate.get("error") or "Invalid proposal.")
                equations.append(
                    f'<details data-equation-id="{batch}-{i}"><summary>{label} · Invalid</summary>'
                    f"{source}<p>{reason}: {error}</p></details>"
                )
                continue
            if not equation:
                continue
            reason = {"duplicate": "duplicate", "fit_failed": "fit failed"}.get(candidate.get("status"), "")
            score = _score_label(candidate.get("metrics"), reason)
            error = candidate.get("error")
            equations.append(
                f'<details data-equation-id="{batch}-{i}"><summary>{label} · {score}</summary>'
                f"<code>{escape(equation)}</code>{_metrics_detail(candidate.get('metrics'))}"
                + (f"<p>{escape(error)}</p>" if error else "")
                + "</details>"
            )
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


def render(wave, step=None, *, fragment=False, snapshot="", layout="", inspect=None):
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
    active = wave.get("active_probe") if live else None
    if active and (idea := active.get("idea")):
        nodes[idea["node"]] = catalog[idea["node"]] = idea
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
    lowest = _lowest_mse(episodes, nodes, wave.get("baseline_exploration"))
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
        bar += (
            '<span class="zoom"><button type="button" data-zoom="out" aria-label="Zoom out">−</button>'
            '<button type="button" data-zoom="fit" aria-label="Reset zoom">Fit</button>'
            '<button type="button" data-zoom="in" aria-label="Zoom in">+</button></span>'
            '<button type="button" data-reset>Reset layout</button>'
        )
    if current in catalog or current:
        label = escape(_label(catalog[current]) if current in catalog else selected.get("label", "New idea"))
        lead = f"Running batch {active['batch']}" if active else "Latest probe"
        focus = f"{lead}: <strong>{label}</strong> · {_mode(selected)}"
        if selected.get("error"):
            focus += f" · {escape(selected['error'])}"
    else:
        focus = "Waiting for the first batch."
    content = f"<p>{focus}</p>"
    shortlist = wave.get("shortlist", {}) if final else {}
    if inspect is not None:
        if inspect not in nodes:
            return ""
        return _inspector(
            inspect, nodes[inspect], probes[inspect], shortlist.get(inspect), wave.get("baseline_exploration")
        )
    if nodes:
        inspectors = "".join(
            _inspector(node, idea, probes[node], shortlist.get(node), wave.get("baseline_exploration"))
            for node, idea in nodes.items()
            if len(nodes) <= CANVAS_AT
        )
        graph = _graph(nodes, catalog, counts, best, current, lowest)
        content += (
            f'<section class="graph-shell">{graph}{inspectors}</section>'
            "<small>Click a node for probes and equations · Drag to move · Scroll to zoom, drag the background to pan · "
            '<span class="legend">lowest MSE for its parameter count</span> · AND requires every parent fit</small>'
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


def make_handler(path, layout):
    """Build the request handler that serves one wave file."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, kind="text/html; charset=utf-8"):
            content = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

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
                    wave = {} if "snapshot" in query else json.load(stream)
            except FileNotFoundError:
                wave, snapshot = {}, "waiting"
            except (OSError, ValueError):
                self.send_error(503, "Wave snapshot unavailable")
                return
            if "snapshot" in query:
                self._send(snapshot, "text/plain; charset=utf-8")
                return
            if "inspect" in query:
                detail = render(wave, step, inspect=query["inspect"][0])
                if detail:
                    self._send(detail)
                else:
                    self.send_error(404)
                return
            self._send(render(wave, step, fragment="fragment" in query, snapshot=snapshot, layout=layout))

        def log_message(self, format, *args):
            pass

    return Handler


def serve(args):
    """Bind to loopback and serve only the chosen wave, even before it exists."""
    path = args.run if args.run.suffix == ".json" else args.run / "wave.json"
    layout = sha256(os.fsencode(path.resolve())).hexdigest()[:16]
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(path, layout)) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()

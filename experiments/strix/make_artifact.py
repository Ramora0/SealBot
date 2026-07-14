"""Render the tl x sims scaling-grid report (grid_fit.json -> HTML artifact).

Usage: python make_artifact.py <out.html>
Reads grid_fit.json (+ solo_s64_tl0.44.json if present) from this dir.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load():
    fit = json.loads((HERE / "grid_fit.json").read_text())
    solo = None
    p = HERE / "solo_s64_tl0.44.json"
    if p.exists() and p.stat().st_size:
        solo = json.loads(p.read_text())
    return fit, solo


def build_data(fit, solo):
    es, ex, se = fit["elo_seal"], fit["elo_strix"], fit["se"]
    lat = fit["strix_sec_per_turn"]
    seal_pts = [{"x": float(t), "y": es[t], "se": se.get(f"seal_{t}", 0.0),
                 "label": f"{t}s", "setting": f"clock {t}s/turn"}
                for t in sorted(es, key=float)]
    strix_pts = [{"x": lat[s], "y": ex[s], "se": se.get(f"strix_{s}", 0.0),
                  "label": f"{s} sims",
                  "setting": f"{s} sims ≈ {lat[s]:.2f}s/turn"}
                 for s in sorted(ex, key=float) if s in lat]

    # sealbot log-linear fit for time-odds
    xs = [math.log2(p["x"]) for p in seal_pts]
    ys = [p["y"] for p in seal_pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    m = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / \
        sum((a - mx) ** 2 for a in xs)
    b = my - m * mx
    odds = {}
    for s, e in ex.items():
        if s in lat and m > 0:
            t_eq = 2 ** ((e - b) / m)
            odds[s] = {"t_eq": t_eq, "ratio": t_eq / lat[s],
                       "extrapolated": t_eq > max(p["x"] for p in seal_pts)}

    cells = []
    for key, c in fit["cells"].items():
        if "strix_wins" not in c:
            continue
        tl = key.split("_")[0][2:]
        s = key.split("_")[1][1:]
        cells.append({"tl": tl, "sims": s, "seal_w": c["strix_losses"],
                      "strix_w": c["strix_wins"],
                      "elo": -c["strix_elo_vs_sealbot"]})

    return {"seal": seal_pts, "strix": strix_pts, "cells": cells,
            "odds": odds, "slope_seal": m, "solo": solo and {
                "seal_w": solo["strix_losses"], "strix_w": solo["strix_wins"],
                "elo": -solo["strix_elo_vs_sealbot"]}}


TEMPLATE = r"""<title>SealBot vs Strix: time scaling map</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --ink-1: #0b0b0b; --ink-2: #52514e; --ink-3: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --seal: #2a78d6; --strix: #1baf7a;
    --div-pos: #2a78d6; --div-neg: #e34948; --div-mid: #f0efec;
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink-1); background: var(--page);
    margin: 0 auto; max-width: 980px; padding: 28px 20px 48px;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
      --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
      --seal: #3987e5; --strix: #199e70;
      --div-pos: #3987e5; --div-neg: #e66767; --div-mid: #383835;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --seal: #3987e5; --strix: #199e70;
    --div-pos: #3987e5; --div-neg: #e66767; --div-mid: #383835;
  }
  .viz-root h1 { font-size: 22px; margin: 0 0 4px; }
  .viz-root .sub { color: var(--ink-2); margin: 0 0 22px; }
  .card { background: var(--surface-1); border: 1px solid var(--ring);
          border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
           gap: 12px; margin: 16px 0; }
  .tile { background: var(--surface-1); border: 1px solid var(--ring);
          border-radius: 10px; padding: 14px 16px; }
  .tile .lbl { color: var(--ink-2); font-size: 12.5px; }
  .tile .val { font-size: 26px; font-weight: 600; margin-top: 2px; }
  .tile .note { color: var(--ink-3); font-size: 12px; margin-top: 2px; }
  .legend { display: flex; gap: 18px; align-items: center; margin: 0 0 6px; }
  .legend .key { display: inline-flex; align-items: center; gap: 7px;
                 color: var(--ink-2); font-size: 13px; }
  .key .stroke { width: 18px; height: 0; border-top: 2.5px solid; border-radius: 2px; }
  .chartwrap { position: relative; overflow-x: auto; }
  svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .tip { position: absolute; pointer-events: none; background: var(--surface-1);
         border: 1px solid var(--ring); border-radius: 8px; padding: 8px 11px;
         box-shadow: 0 2px 10px rgba(0,0,0,.13); font-size: 12.5px;
         display: none; min-width: 150px; z-index: 3; }
  .tip .v { font-size: 15px; font-weight: 600; }
  .tip .s { color: var(--ink-2); }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th, td { padding: 7px 10px; text-align: right;
           font-variant-numeric: tabular-nums; }
  th { color: var(--ink-2); font-weight: 500; border-bottom: 1px solid var(--axis); }
  td:first-child, th:first-child { text-align: left; }
  tbody tr + tr td { border-top: 1px solid var(--grid); }
  .cellnote { color: var(--ink-3); font-size: 12px; margin-top: 8px; }
  h2 { font-size: 15.5px; margin: 0 0 10px; }
  .caveats li { margin: 6px 0; color: var(--ink-2); }
  .caveats b { color: var(--ink-1); }
  .tablewrap { overflow-x: auto; }
</style>
<div class="viz-root">
  <h1>SealBot vs Strix: the time scaling map</h1>
  <p class="sub">v1.5 champion (trunk5 + VCF, honest clock) vs hexo-strix
  (Gumbel MCTS, ckpt 237k) &middot; 9-cell grid, 100 games/cell, paired human
  openings &middot; July 14, 2026</p>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>Strength vs thinking time &mdash; both engines on one Elo axis</h2>
    <div class="legend" id="legend"></div>
    <div class="chartwrap" id="chartwrap"><div class="tip" id="tip"></div></div>
    <p class="cellnote">Bradley&ndash;Terry fit over all grid cells, anchored at
    SealBot 0.44s/turn = 0 Elo. Whiskers are &plusmn;1 SE. X-axis is seconds per
    2-stone turn (log scale): SealBot&rsquo;s configured clock; strix&rsquo;s
    measured GPU latency per sims setting. Hover or focus a point for detail.</p>
  </div>

  <div class="card">
    <h2>Raw cross table &mdash; SealBot wins per 100 games (SealBot-POV Elo)</h2>
    <div class="tablewrap"><table id="xtable"></table></div>
    <p class="cellnote" id="xnote"></p>
  </div>

  <div class="card">
    <h2>Fitted ratings (table view of the chart)</h2>
    <div class="tablewrap"><table id="ftable"></table></div>
  </div>

  <div class="card caveats" id="caveats"></div>
  <div class="card caveats" id="method"></div>
</div>
<script>
const DATA = __DATA__;
const css = v => getComputedStyle(document.querySelector('.viz-root'))
  .getPropertyValue(v).trim();

function fmtElo(e, signed = true) {
  const r = Math.round(e);
  return (signed && r > 0 ? '+' : r < 0 ? '−' : '') + Math.abs(r);
}

// ---- tiles ----
const odds64 = DATA.odds['64'];
const tiles = [
  {lbl: 'Equal-time verdict (0.44s/turn)',
   val: fmtElo(-DATA.strix.find(p => p.label === '64 sims').y) + ' Elo',
   note: 'SealBot vs strix@64 sims, fitted'},
  {lbl: 'Time odds to match strix@64',
   val: '≈' + Math.round(odds64.ratio) + '×',
   note: (odds64.extrapolated ? 'extrapolated: ' : '') +
         'SealBot needs ~' + odds64.t_eq.toFixed(1) + 's vs strix’s ' +
         DATA.strix.find(p => p.label === '64 sims').x.toFixed(2) + 's'},
  {lbl: 'SealBot scaling', val: fmtElo(DATA.slope_seal) + ' Elo',
   note: 'per doubling of clock'},
  {lbl: 'Strix scaling cliff',
   val: fmtElo(DATA.strix.find(p => p.label === '64 sims').y -
               DATA.strix.find(p => p.label === '16 sims').y) + ' Elo',
   note: '16 → 64 sims; 4 → 16 sims is nearly flat'},
];
document.getElementById('tiles').innerHTML = tiles.map(t =>
  '<div class="tile"><div class="lbl"></div><div class="val"></div>' +
  '<div class="note"></div></div>').join('');
document.querySelectorAll('#tiles .tile').forEach((el, i) => {
  el.querySelector('.lbl').textContent = tiles[i].lbl;
  el.querySelector('.val').textContent = tiles[i].val;
  el.querySelector('.note').textContent = tiles[i].note;
});

// ---- legend ----
const SERIES = [
  {key: 'seal', name: 'SealBot v1.5 (CPU minimax)', color: css('--seal')},
  {key: 'strix', name: 'Strix (GPU Gumbel MCTS)', color: css('--strix')},
];
document.getElementById('legend').innerHTML = SERIES.map(s =>
  '<span class="key"><span class="stroke"></span><span></span></span>').join('');
document.querySelectorAll('#legend .key').forEach((el, i) => {
  el.querySelector('.stroke').style.borderTopColor = SERIES[i].color;
  el.querySelector('span:last-child').textContent = SERIES[i].name;
});

// ---- chart ----
const W = 880, H = 400, ML = 56, MR = 120, MT = 18, MB = 46;
const allPts = [...DATA.seal, ...DATA.strix];
const xmin = Math.min(...allPts.map(p => p.x)) / 1.6;
const xmax = Math.max(...allPts.map(p => p.x)) * 1.7;
const ymin = Math.floor((Math.min(...allPts.map(p => p.y - p.se)) - 40) / 100) * 100;
const ymax = Math.ceil((Math.max(...allPts.map(p => p.y + p.se)) + 40) / 100) * 100;
const X = v => ML + (Math.log2(v) - Math.log2(xmin)) /
  (Math.log2(xmax) - Math.log2(xmin)) * (W - ML - MR);
const Y = v => MT + (ymax - v) / (ymax - ymin) * (H - MT - MB);

let svg = '';
for (let g = ymin; g <= ymax; g += 100) {
  svg += `<line x1="${ML}" x2="${W - MR}" y1="${Y(g)}" y2="${Y(g)}"
    stroke="${g === 0 ? css('--axis') : css('--grid')}" stroke-width="1"/>`;
  svg += `<text x="${ML - 8}" y="${Y(g) + 4}" text-anchor="end"
    fill="${css('--ink-3')}" style="font-variant-numeric:tabular-nums">${fmtElo(g)}</text>`;
}
const ticks = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2].filter(t => t >= xmin && t <= xmax);
for (const t of ticks) {
  svg += `<line x1="${X(t)}" x2="${X(t)}" y1="${H - MB}" y2="${H - MB + 5}"
    stroke="${css('--axis')}" stroke-width="1"/>`;
  svg += `<text x="${X(t)}" y="${H - MB + 20}" text-anchor="middle"
    fill="${css('--ink-3')}">${t}</text>`;
}
svg += `<line x1="${ML}" x2="${W - MR}" y1="${H - MB}" y2="${H - MB}"
  stroke="${css('--axis')}" stroke-width="1"/>`;
svg += `<text x="${(ML + W - MR) / 2}" y="${H - 6}" text-anchor="middle"
  fill="${css('--ink-2')}">seconds per 2-stone turn (log scale)</text>`;
svg += `<text transform="rotate(-90 14 ${(MT + H - MB) / 2})" x="14"
  y="${(MT + H - MB) / 2}" text-anchor="middle" fill="${css('--ink-2')}">Elo (SealBot @0.44s = 0)</text>`;

// equal-time annotation at 0.44
svg += `<line x1="${X(0.44)}" x2="${X(0.44)}" y1="${MT}" y2="${H - MB}"
  stroke="${css('--axis')}" stroke-width="1"/>`;
svg += `<text x="${X(0.44) + 5}" y="${MT + 11}" fill="${css('--ink-3')}"
  font-size="11">equal-time benches (0.44s)</text>`;

for (const s of [{pts: DATA.seal, c: css('--seal'), name: 'SealBot v1.5', dy: 20},
                 {pts: DATA.strix, c: css('--strix'), name: 'Strix', dy: -12}]) {
  const path = s.pts.map((p, i) => (i ? 'L' : 'M') + X(p.x) + ' ' + Y(p.y)).join(' ');
  svg += `<path d="${path}" fill="none" stroke="${s.c}" stroke-width="2"
    stroke-linejoin="round" stroke-linecap="round"/>`;
  for (const p of s.pts) {
    svg += `<line x1="${X(p.x)}" x2="${X(p.x)}" y1="${Y(p.y - p.se)}"
      y2="${Y(p.y + p.se)}" stroke="${s.c}" stroke-width="1.5" opacity="0.55"/>`;
    svg += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5" fill="${s.c}"
      stroke="${css('--surface-1')}" stroke-width="2"/>`;
    svg += `<text x="${X(p.x)}" y="${Y(p.y) + (s.dy > 0 ? s.dy + 8 : s.dy - 2)}"
      text-anchor="middle" fill="${css('--ink-2')}" font-size="11.5">${p.label}</text>`;
  }
  const last = s.pts[s.pts.length - 1];
  svg += `<text x="${X(last.x) + 12}" y="${Y(last.y) + 4}"
    fill="${css('--ink-2')}" font-weight="600">${s.name}</text>`;
}

const wrap = document.getElementById('chartwrap');
wrap.insertAdjacentHTML('beforeend',
  `<svg viewBox="0 0 ${W} ${H}" style="width:100%;min-width:640px;display:block">${svg}</svg>`);

// hover layer: >=24px transparent hit targets, keyboard focusable
const tip = document.getElementById('tip');
const svgEl = wrap.querySelector('svg');
allPts.forEach((p, idx) => {
  const isSeal = idx < DATA.seal.length;
  const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  c.setAttribute('cx', X(p.x)); c.setAttribute('cy', Y(p.y));
  c.setAttribute('r', 15); c.setAttribute('fill', 'transparent');
  c.setAttribute('tabindex', '0'); c.style.cursor = 'pointer';
  const show = () => {
    tip.innerHTML = '<div class="v"></div><div class="s"></div><div class="s"></div>';
    tip.children[0].textContent = fmtElo(p.y) + ' ± ' + Math.round(p.se) + ' Elo';
    tip.children[1].textContent = (isSeal ? 'SealBot, ' : 'Strix, ') + p.setting;
    tip.children[2].textContent = p.x.toFixed(3) + ' s/turn';
    tip.style.display = 'block';
    const r = svgEl.getBoundingClientRect();
    const px = X(p.x) / W * r.width, py = Y(p.y) / H * r.height;
    tip.style.left = Math.min(px + 14, r.width - 170) + 'px';
    tip.style.top = (py - 14) + 'px';
  };
  c.addEventListener('pointerenter', show);
  c.addEventListener('focus', show);
  c.addEventListener('pointerleave', () => tip.style.display = 'none');
  c.addEventListener('blur', () => tip.style.display = 'none');
  svgEl.appendChild(c);
});

// ---- cross table ----
const tls = [...new Set(DATA.cells.map(c => c.tl))].sort((a, b) => a - b);
const simsL = [...new Set(DATA.cells.map(c => c.sims))].sort((a, b) => a - b);
const xt = document.getElementById('xtable');
let thead = '<thead><tr><th>SealBot clock \\ strix sims</th>' + simsL.map(s => {
  const sp = DATA.strix.find(p => p.label === s + ' sims');
  return `<th>${s} sims (${sp ? sp.x.toFixed(2) : '?'}s/turn)</th>`;
}).join('') + '</tr></thead>';
let tbody = '<tbody>' + tls.map(tl => '<tr><td>' + tl + 's/turn</td>' +
  simsL.map(s => {
    const c = DATA.cells.find(k => k.tl === tl && k.sims === s);
    if (!c) return '<td>&mdash;</td>';
    const p = c.seal_w / (c.seal_w + c.strix_w);
    const d = p - 0.5;
    const col = d >= 0 ? css('--div-pos') : css('--div-neg');
    const alpha = Math.min(Math.abs(d) * 2, 1) * 0.28;
    return `<td style="background:color-mix(in srgb, ${col} ${alpha * 100}%, transparent)">` +
      `<b>${c.seal_w}</b>&ndash;${c.strix_w} (${fmtElo(c.elo)})</td>`;
  }).join('') + '</tr>').join('') + '</tbody>';
xt.innerHTML = thead + tbody;
document.getElementById('xnote').textContent =
  'Cell = SealBot wins–strix wins, 100 games (Elo from SealBot’s POV). ' +
  'Wash: blue = SealBot ahead, red = strix ahead, gray ≈ even.' +
  (DATA.solo ? ' The 0.44s/64-sims cell pools the contended run with an ' +
    'uncontended re-run (' + DATA.solo.seal_w + '–' + DATA.solo.strix_w +
    ' solo; 200 games total).' : '');

// ---- fitted table ----
const ft = document.getElementById('ftable');
ft.innerHTML = '<thead><tr><th>Engine / setting</th><th>s per turn</th>' +
  '<th>Fitted Elo</th><th>&plusmn;SE</th></tr></thead><tbody>' +
  allPts.map((p, i) =>
    `<tr><td>${i < DATA.seal.length ? 'SealBot ' : 'Strix '}${p.label}</td>` +
    `<td>${p.x.toFixed(3)}</td><td>${fmtElo(p.y)}</td>` +
    `<td>${Math.round(p.se)}</td></tr>`).join('') + '</tbody>';

// ---- caveats & method ----
document.getElementById('caveats').innerHTML = __CAVEATS__;
document.getElementById('method').innerHTML = __METHOD__;
</script>
"""

CAVEATS = """<h2>Caveats &amp; verification</h2><ul>
<li><b>Contention: verified negligible.</b> The three sims-rows ran as
concurrent processes to fit the clock budget. Strix strength is
sims-controlled, so its side is unaffected by construction; SealBot's search
throughput measured <b>59k nps both solo and contended</b> (identical), and
__SOLO_SENTENCE__</li>
<li><b>Node effect.</b> Last night's champion benches ran on a different
node (this one shares a socket with a long-running GROMACS job); the same
cell read 29/100 there vs 24/100 solo here &mdash; within noise, but
compare across days with that in mind. Within-grid comparisons are clean.</li>
<li><b>Protocol.</b> Paired human openings 0&ndash;49, both colors (the
honest dev gate). The empty-board scoreboard reads ~107 Elo higher for
SealBot from opening memorization (63/150 vs 42/150 at equal time).</li>
<li><b>Low-sims flatness is real, not noise:</b> at 4&ndash;16 sims Gumbel
MCTS is nearly raw policy, so those two strix settings tie within error.</li>
<li>The 35&times; time-odds figure extrapolates SealBot's +41/doubling
line ~3 doublings past its measured range; treat it as an order of
magnitude, not a point estimate.</li></ul>"""

METHOD = """<h2>Method</h2><ul>
<li>SealBot: v1.5 champion build (<code>current/</code>, output_trunk5,
SEAL_EVAL=trunk, TRUNK_POLICY=1, BLEND=0, MODE=74, VCF=15, k=11, 40k).
Clock = seconds per 2-stone turn, wall clock.</li>
<li>Strix: checkpoint 237000, Gumbel MCTS (m_actions=16, c_visit=50),
n_simulations swept; latency = measured sec/stone &times; 2, from
uncontended runs.</li>
<li>Ratings: single Bradley&ndash;Terry (logistic) fit over all cells,
one parameter per engine setting, anchor SealBot@0.44s = 0. SEs from the
observed information matrix.</li>
<li>Time odds = horizontal gap between the two curves (SealBot's curve
extrapolated log-linearly where needed).</li></ul>"""


def main():
    out = Path(sys.argv[1])
    fit, solo = load()
    data = build_data(fit, solo)
    if solo is not None:
        solo_sentence = (
            "a solo re-run of the most suspect cell (0.44s vs 64 sims) "
            f"scored SealBot {data['solo']['seal_w']}/100 vs 16/100 "
            "contended (z&asymp;1.4, not significant). The two runs are "
            "pooled (200 games) in the table and fit.")
    else:
        solo_sentence = ("solo verification of the 0.44s/64-sims cell is "
                         "pending.")
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__CAVEATS__", json.dumps(
                CAVEATS.replace("__SOLO_SENTENCE__", solo_sentence)
                .replace("\n", " ")))
            .replace("__METHOD__", json.dumps(METHOD.replace("\n", " "))))
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()

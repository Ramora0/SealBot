"""Analyze the tl x sims scaling grid (grid_s{sims}_tl{tl}.json).

Fits a Bradley-Terry model over all cells: one Elo rating per sealbot tl
level and one per strix sims level, anchored at E_seal(0.44) = 0. This
puts both engines on a common strength axis, plotted against wall-clock
seconds per 2-stone turn (sealbot: tl; strix: 2 x measured sec/stone).

Outputs a markdown table + fit summary, and dumps grid_fit.json for
plotting.
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TLS = ["0.11", "0.44", "1.76"]
SIMS = ["4", "16", "64"]
LN10_400 = math.log(10) / 400

# solo (uncontended) strix sec/stone for the time axis, measured on this
# node; grid rows ran in parallel so in-cell measurements are
# GPU-contention-inflated (e.g. s16 read 0.057 contended vs 0.046 solo).
SOLO_SEC_PER_STONE = {"4": 0.029, "16": 0.0463, "64": 0.2077}


def load_cells():
    cells = {}
    for s in SIMS:
        for tl in TLS:
            p = HERE / f"grid_s{s}_tl{tl}.json"
            if p.exists() and p.stat().st_size:
                d = json.loads(p.read_text())
                if "strix_wins" in d:
                    cells[(tl, s)] = d
    # pool the solo verification re-run into the (0.44, 64) cell
    solo = HERE / "solo_s64_tl0.44.json"
    if solo.exists() and solo.stat().st_size and ("0.44", "64") in cells:
        d = json.loads(solo.read_text())
        c = cells[("0.44", "64")]
        c["strix_wins"] += d["strix_wins"]
        c["strix_losses"] += d["strix_losses"]
        c["games"] += d["games"]
        p = c["strix_wins"] / (c["strix_wins"] + c["strix_losses"])
        c["strix_elo_vs_sealbot"] = round(
            400 * math.log10(p / (1 - p)), 1)
        c["pooled_with_solo"] = True
    return cells


def fit_bt(cells):
    """Returns (elo_seal{tl}, elo_strix{sims}, se{param_name})."""
    tls = sorted({tl for tl, _ in cells}, key=float)
    sims = sorted({s for _, s in cells}, key=float)
    # params: seal elos for tls != anchor, then strix elos
    anchor = "0.44" if "0.44" in tls else tls[0]
    free_tls = [t for t in tls if t != anchor]
    names = [f"seal_{t}" for t in free_tls] + [f"strix_{s}" for s in sims]

    # logistic regression via IRLS: p_cell = sigmoid(X theta), theta in
    # natural units (Elo * ln10/400); X has +1 at the strix param, -1 at
    # the sealbot param (anchor tl contributes 0)
    idx = {n: i for i, n in enumerate(names)}
    X, wins, n = [], [], []
    for (tl, s), c in cells.items():
        row = np.zeros(len(names))
        if tl != anchor:
            row[idx[f"seal_{tl}"]] = -1.0
        row[idx[f"strix_{s}"]] = 1.0
        X.append(row)
        wins.append(c["strix_wins"])
        n.append(c["strix_wins"] + c["strix_losses"])
    X, wins, n = np.array(X), np.array(wins, float), np.array(n, float)

    theta = np.zeros(len(names))
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-X @ theta))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        W = n * p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (wins - n * p)
        step = np.linalg.solve(H, g)
        theta += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-X @ theta)), 1e-12, 1 - 1e-12)
    nllv = -float(np.sum(wins * np.log(p) + (n - wins) * np.log(1 - p)))
    cov = np.linalg.inv(X.T @ (X * (n * p * (1 - p))[:, None]))
    se_nat = np.sqrt(np.diag(cov))

    es = {anchor: 0.0}
    es.update({t: float(theta[idx[f"seal_{t}"]] / LN10_400) for t in free_tls})
    ex = {s: float(theta[idx[f"strix_{s}"]] / LN10_400) for s in sims}
    se = {nm: float(se_nat[idx[nm]] / LN10_400) for nm in names}
    return es, ex, se, nllv


def main():
    cells = load_cells()
    if not cells:
        print("no cells yet")
        return
    print(f"{len(cells)}/{len(TLS) * len(SIMS)} cells loaded\n")

    # strix latency per sims level: solo measurement preferred; fall back
    # to the (contention-inflated) in-cell mean
    lat = {}
    for s in SIMS:
        solo = SOLO_SEC_PER_STONE.get(s)
        if solo is None and (("0.11", s) in cells):
            solo = cells[("0.11", s)]["strix_sec_per_stone_mean"]
        vals = [c["strix_sec_per_stone_mean"]
                for (tl, ss), c in cells.items() if ss == s]
        if solo is not None:
            lat[s] = 2 * solo  # sec per 2-stone turn
        elif vals:
            lat[s] = 2 * sum(vals) / len(vals)

    # raw cross table (strix wins /100, from sealbot's POV as losses)
    hdr = "| sealbot tl \\ strix sims | " + " | ".join(
        f"{s} ({lat.get(s, float('nan')):.3f}s/turn)" for s in SIMS) + " |"
    print(hdr)
    print("|" + "---|" * (len(SIMS) + 1))
    for tl in TLS:
        row = [f"| {tl}s "]
        for s in SIMS:
            c = cells.get((tl, s))
            if c:
                sw = c["strix_wins"]
                sb = c["strix_losses"]
                row.append(f"seal {sb}-{sw} ({c['strix_elo_vs_sealbot']:+.0f} strix) ")
            else:
                row.append("... ")
        print("|".join(row) + "|")

    if len(cells) < 4:
        return
    es, ex, se, nllv = fit_bt(cells)
    print("\nBradley-Terry fit (anchor: sealbot @0.44s = 0 Elo):")
    for t in sorted(es, key=float):
        s_ = se.get(f"seal_{t}", float("nan"))
        print(f"  sealbot tl={t:>5}s/turn: {es[t]:+7.1f} Elo  (se {s_:.0f})")
    for s in sorted(ex, key=float):
        s_ = se.get(f"strix_{s}", float("nan"))
        print(f"  strix sims={s:>4} ({lat.get(s, 0):.3f}s/turn): "
              f"{ex[s]:+7.1f} Elo  (se {s_:.0f})")

    # Elo per time-doubling (log-linear fit per engine)
    def slope(pts):
        if len(pts) < 2:
            return float("nan")
        xs = np.log2([p[0] for p in pts])
        ys = [p[1] for p in pts]
        return float(np.polyfit(xs, ys, 1)[0])

    seal_pts = [(float(t), es[t]) for t in es]
    strix_pts = [(lat[s], ex[s]) for s in ex if s in lat]
    m_seal, m_strix = slope(seal_pts), slope(strix_pts)
    print(f"\nElo per time-doubling: sealbot {m_seal:+.0f}, strix {m_strix:+.0f}")

    # time-odds: sealbot time needed to match strix at each sims level,
    # extrapolating sealbot's log-linear curve
    b_seal = float(np.polyfit(np.log2([p[0] for p in seal_pts]),
                              [p[1] for p in seal_pts], 1)[1])
    print("\nEqual-strength frontier (sealbot log-linear extrapolation):")
    for s in sorted(ex, key=float):
        if s not in lat:
            continue
        t_eq = 2 ** ((ex[s] - b_seal) / m_seal) if m_seal else float("nan")
        print(f"  strix sims={s:>4} uses {lat[s]:.3f}s/turn -> sealbot needs "
              f"~{t_eq:.2f}s/turn = {t_eq / lat[s]:.0f}x time odds")

    out = {
        "cells": {f"tl{tl}_s{s}": c for (tl, s), c in cells.items()},
        "elo_seal": es, "elo_strix": ex, "se": se,
        "strix_sec_per_turn": lat,
        "elo_per_doubling": {"sealbot": m_seal, "strix": m_strix},
    }
    (HERE / "grid_fit.json").write_text(json.dumps(out, indent=1))
    print("\nwrote grid_fit.json")


if __name__ == "__main__":
    main()

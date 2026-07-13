"""Measure what actually matters at min-nodes: refutation recall of the
policy tables vs the linear delta, ON TREE-INTERIOR-LIKE POSITIONS.

Takes strong-play positions, perturbs them with 1-2 speculative turns
(mimicking search-tree interiors), then measures with a FRESH strix forward
as oracle:
  A. recall of strix's top move within top-15 of each orderer
     (on-distribution vs perturbed)
  B. recall of FORCED-BLOCK cells (empties of opponent >=4 windows) within
     top-15 — the refutation-specific metric
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / ".." / "nnue"))

from strix_bridge import load_strix, state_from_cells
from policy_features import extract_cell_features
from features import DIRS
import re
import torch

_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def cand_cells(cells):
    occ = {(q, r) for q, r, _ in cells}
    return sorted({(q + dq, r + dr) for q, r, _ in cells
                   for dq, dr in _D2 if (q + dq, r + dr) not in occ})


def load_tables():
    d = torch.load(SCRIPT_DIR / "output_policy" / "policy.pt")
    pw, pc = d["pw"].numpy(), d["pc"].numpy()
    text = open(SCRIPT_DIR / ".." / ".." / "current" / "pattern_data.h").read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    return pw, pc, np.array([float(x) for x in nums])


def rankers(cells, mover, cand, pw, pc, pv):
    win, cls = extract_cell_features([tuple(c) for c in cells], mover, cand)
    pol = pw[win].sum(axis=1) + np.where(cls >= 0, pc[np.maximum(cls, 0)], pc[-1])
    delta = np.zeros(len(cand))
    for s in range(18):
        j = s % 6
        delta += pv[win[:, s] + 3 ** j] - pv[win[:, s]]
    return pol, np.abs(delta)


def forced_blocks(cells, mover, cand):
    """Empty cells of opponent windows with >=4 stones and none of ours."""
    occ = {(q, r): p for q, r, p in cells}
    opp = 2 if mover == 1 else 1
    blocks = set()
    for q, r in list(occ):
        for dq, dr in DIRS:
            for a in range(-5, 1):
                stones = 0
                mine = 0
                empt = []
                for j in range(6):
                    c = (q + (a + j) * dq, r + (a + j) * dr)
                    v = occ.get(c)
                    if v == opp:
                        stones += 1
                    elif v == mover:
                        mine += 1
                    else:
                        empt.append(c)
                if stones >= 4 and mine == 0:
                    blocks.update(empt)
    return blocks & set(cand)


def perturb(cells, mover, rng, pv, pw, pc, n_turns):
    cells = list(cells)
    cur = mover
    for _ in range(n_turns):
        for _ in range(2):
            cand = cand_cells(cells)
            if not cand:
                return None
            pol, delta = rankers(cells, cur, cand, pw, pc, pv)
            if rng.random() < 0.5:
                top = np.argsort(-delta)[:8]
                pick = cand[int(rng.choice(top))]
            else:
                pick = cand[rng.randrange(len(cand))]
            cells.append((pick[0], pick[1], cur))
        cur = 3 - cur
    return cells, cur


def main():
    model, mc, gc = load_strix()
    pw, pc, pv = load_tables()
    rng = random.Random(17)

    with open(SCRIPT_DIR / "strong_play_recs.pkl", "rb") as fh:
        recs = [r[:4] for r in pickle.load(fh)]
    rng.shuffle(recs)
    recs = recs[:500]

    import hexo_rs
    from hexo_a0.graph import axis_states_to_batch

    def oracle_top(cells_movers):
        states, keeps = [], []
        for cells, mover in cells_movers:
            s = state_from_cells(cells, mover, 2, gc)
            if s is None or s.is_terminal():
                keeps.append(None)
            else:
                keeps.append(s)
        live = [s for s in keeps if s is not None]
        tops = {}
        with torch.inference_mode():
            for i in range(0, len(live), 256):
                sub = live[i:i + 256]
                batch, aux = axis_states_to_batch(
                    sub, prune_empty_edges=mc.prune_empty_edges,
                    threat_features=mc.threat_features,
                    relative_stones=mc.relative_stone_encoding,
                    device="cuda:0")
                lg, cnt, _ = model._forward_batch_core(
                    batch, legal_idx=aux.legal_idx, stone_idx=aux.stone_idx,
                    stone_batch=aux.stone_batch)
                lg = lg.float().cpu().numpy()
                cnt = cnt.cpu().numpy()
                off = 0
                for s, n in zip(sub, cnt):
                    lm = list(map(tuple, s.legal_moves()))
                    tops[id(s)] = lm[int(np.argmax(lg[off:off + n]))]
                    off += n
        out = []
        for s in keeps:
            out.append(tops.get(id(s)) if s is not None else None)
        return out

    def measure(tag, batch_positions):
        oracle = oracle_top([(c, m) for c, m in batch_positions])
        pol_hit = del_hit = n = 0
        fb_pol = fb_del = fb_tot = 0
        for (cells, mover), top in zip(batch_positions, oracle):
            if top is None:
                continue
            # translate oracle move back if the bridge translated
            owner = {(q, r): p for q, r, p in cells}
            if owner.get((0, 0)) != 1:
                a = [(q, r) for q, r, p in cells if p == 1][0]
                top = (top[0] + a[0], top[1] + a[1])
            cand = cand_cells(cells)
            if top not in cand:
                continue
            pol, delta = rankers(cells, mover, cand, pw, pc, pv)
            mi = cand.index(top)
            n += 1
            pol_hit += int((pol > pol[mi]).sum()) < 15
            del_hit += int((delta > delta[mi]).sum()) < 15
            for b in forced_blocks(cells, mover, cand):
                bi = cand.index(b)
                fb_tot += 1
                fb_pol += int((pol > pol[bi]).sum()) < 15
                fb_del += int((delta > delta[bi]).sum()) < 15
        print(f"{tag}: n={n}")
        print(f"  strix-top-move in top-15:  policy {pol_hit/n:.1%}   "
              f"delta {del_hit/n:.1%}")
        if fb_tot:
            print(f"  forced-block cells in top-15 (n={fb_tot}): "
                  f"policy {fb_pol/fb_tot:.1%}   delta {fb_del/fb_tot:.1%}")

    base = [(cells, mover) for cells, mover, _, _ in recs]
    measure("REAL positions (on-distribution)", base)

    pert = []
    for cells, mover, _, _ in recs:
        out = perturb(cells, mover, rng, pv, pw, pc, rng.choice([1, 2]))
        if out:
            pert.append(out)
    measure("PERTURBED 1-2 turns (tree-interior-like)", pert)


if __name__ == "__main__":
    main()

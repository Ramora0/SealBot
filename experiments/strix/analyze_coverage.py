"""Candidate-coverage analysis: at strong-play positions where strix chose a
move, (1) was that move even in our D2 candidate set? (2) what rank does the
current linear _move_delta ordering give it (caps: 15 interior / 20 root)?
(3) what rank do the NEW policy tables give it (if trained)?

Needs strong_play_recs.pkl from fidelity_diag.py. SealBot venv.
"""

import os
import pickle
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))

from policy_features import extract_cell_features
from features import POW3_6, DIRS, PAD

_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def load_pv():
    text = open(os.path.join(SCRIPT_DIR, "..", "..", "current",
                             "pattern_data.h")).read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    return np.array([float(x) for x in nums])


def main():
    with open(os.path.join(SCRIPT_DIR, "strong_play_recs.pkl"), "rb") as fh:
        recs = pickle.load(fh)
    pv = load_pv()

    pol = None
    ppath = os.path.join(SCRIPT_DIR, "output_policy", "policy.pt")
    if os.path.exists(ppath):
        import torch
        d = torch.load(ppath)
        pol = (d["pw"].numpy(), d["pc"].numpy())

    n = in_d2 = 0
    lin_ranks, pol_ranks, n_cands = [], [], []
    for rec in recs:
        if len(rec) < 5 or rec[4][0] != "strix_move":
            continue
        cells, mover, ml, mc = rec[:4]
        mv = tuple(rec[4][1])
        occ = {(q, r) for q, r, _ in cells}
        cand = sorted({(q + dq, r + dr) for q, r, _ in cells
                       for dq, dr in _D2 if (q + dq, r + dr) not in occ})
        n += 1
        if mv not in cand:
            continue
        in_d2 += 1
        n_cands.append(len(cand))
        win, cls = extract_cell_features([tuple(c) for c in cells], mover,
                                         cand)
        # linear delta ordering: delta = sum_w pv[pat + 3^k] - pv[pat],
        # digit 1 = mover; k recoverable from window slot (j within dir)
        deltas = np.zeros(len(cand))
        for s in range(18):
            j = s % 6
            deltas += pv[win[:, s] + 3 ** j] - pv[win[:, s]]
        mi = cand.index(mv)
        lin_ranks.append(int((deltas > deltas[mi]).sum()) + 1)
        if pol is not None:
            pw, pc = pol
            sc = pw[win].sum(axis=1) + np.where(cls >= 0, pc[np.maximum(cls, 0)],
                                                pc[-1] if len(pc) > 8548 else 0.0)
            pol_ranks.append(int((sc > sc[mi]).sum()) + 1)

    lin = np.array(lin_ranks)
    print(f"{n} strix moves; in D2 candidate set: {in_d2/n:.1%}")
    print(f"candidates per position: mean {np.mean(n_cands):.0f}")
    print(f"\nlinear _move_delta rank of strix's move:")
    print(f"  mean {lin.mean():.1f}  median {np.median(lin):.0f}")
    for cap in (5, 15, 20):
        print(f"  within top-{cap}: {(lin <= cap).mean():.1%}")
    if pol_ranks:
        pr = np.array(pol_ranks)
        print(f"\npolicy-table rank of strix's move:")
        print(f"  mean {pr.mean():.1f}  median {np.median(pr):.0f}")
        for cap in (5, 15, 20):
            print(f"  within top-{cap}: {(pr <= cap).mean():.1%}")


if __name__ == "__main__":
    main()

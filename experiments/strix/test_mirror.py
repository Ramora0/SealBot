"""Verify the engine's color-mirror identities for the policy path.

The engine computes root-relative window patterns/classes and, for
opponent-to-move nodes, looks up POLICY_W[MIRROR729[p]] / POLICY_C[
CLASS_MIRROR[cls]]. That is only correct if:

    MIRROR729[pattern(root-rel)] == pattern(mover-rel)      per window
    CLASS_MIRROR[class(root-rel)] == class(mover-rel)       per cell

for every position where mover != root. Checks both on random positions.
"""

import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / ".." / "nnue"))

from features import MIRROR729, CLASS_MIRROR
from policy_features import extract_cell_features

_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def random_position(rng, n):
    cells = [(0, 0, 1)]
    occ = {(0, 0)}
    player = 2
    placed = 1
    while placed < n:
        base = rng.choice(list(occ))
        c = (base[0] + rng.choice(_D2)[0], base[1] + rng.choice(_D2)[1])
        if c in occ:
            continue
        occ.add(c)
        cells.append((c[0], c[1], player))
        placed += 1
        if placed % 2 == 1:
            player = 3 - player
    return cells


def main():
    rng = random.Random(11)
    bad_w = bad_c = tot_w = tot_c = 0
    for trial in range(50):
        cells = random_position(rng, rng.randint(4, 30))
        occ = {(q, r) for q, r, _ in cells}
        cand = sorted({(q + dq, r + dr) for q, r, _ in cells
                       for dq, dr in _D2 if (q + dq, r + dr) not in occ})
        # "root" = player 1; "mover" = player 2 (opponent-to-move case)
        win_root, cls_root = extract_cell_features(cells, 1, cand)
        win_mov, cls_mov = extract_cell_features(cells, 2, cand)

        mw = MIRROR729[win_root]
        bad_w += int((mw != win_mov).sum())
        tot_w += mw.size
        valid = (cls_root >= 0) & (cls_mov >= 0)
        mc = CLASS_MIRROR[cls_root[valid]]
        bad_c += int((mc != cls_mov[valid]).sum())
        tot_c += int(valid.sum())
        both_neg = ((cls_root < 0) != (cls_mov < 0)).sum()
        if both_neg:
            print(f"trial {trial}: -1 class mismatch x{both_neg}")

    print(f"window mirror: {bad_w}/{tot_w} mismatches")
    print(f"class mirror:  {bad_c}/{tot_c} mismatches")
    if bad_w == 0 and bad_c == 0:
        print("MIRROR IDENTITIES HOLD — engine mirror path is correct")
    else:
        print("MIRROR BROKEN — this is the -512 bug")


if __name__ == "__main__":
    main()

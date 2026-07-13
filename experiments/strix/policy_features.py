"""Per-candidate-cell features for policy distillation, mirroring engine
conventions exactly (features.py grid code, mover-relative digits).

For an empty candidate cell: the 18 window pattern indices of the windows
containing it (6 anchors x 3 directions, current board = before placing),
and its conjunction class (empty-cell class from _lp codes). These are the
same quantities the engine reads in O(1) from _wp and _lc at search time.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))

from features import (CODEBOOK, CANON_EMPTY, PAD, DIRS, POW3_6, POW3_11,
                      NUM_CLASSES)


def extract_cell_features(cells, mover, cand_cells):
    """cells: [(q,r,player)], mover: 1|2, cand_cells: [(q,r)] empty cells.

    Returns (win_idx [n,18] int32, cls [n] int32); cls -1 when the cell has
    no stones within 5 in any direction (engine scores those 0).
    """
    qs = np.array([c[0] for c in cells] + [c[0] for c in cand_cells])
    rs = np.array([c[1] for c in cells] + [c[1] for c in cand_cells])
    q0, r0 = qs.min() - PAD, rs.min() - PAD
    H = int(qs.max() - q0 + PAD + 1)
    W = int(rs.max() - r0 + PAD + 1)
    grid = np.zeros((H, W), dtype=np.int64)
    for q, r, p in cells:
        grid[q - q0, r - r0] = 1 if p == mover else 2

    # per-direction window-pattern map (pattern of window anchored at cell)
    pats = []
    lps = []
    for dq, dr in DIRS:
        pat = np.zeros((H, W), dtype=np.int64)
        for j in range(6):
            sq, sr = j * dq, j * dr
            src = np.zeros((H, W), dtype=np.int64)
            lo_q, hi_q = max(0, -sq), min(H, H - sq)
            lo_r, hi_r = max(0, -sr), min(W, W - sr)
            src[lo_q:hi_q, lo_r:hi_r] = grid[lo_q + sq:hi_q + sq,
                                             lo_r + sr:hi_r + sr]
            pat += src * POW3_6[j]
        pats.append(pat)
        lp = np.zeros((H, W), dtype=np.int64)
        for u in range(-5, 6):
            sq, sr = u * dq, u * dr
            src = np.zeros((H, W), dtype=np.int64)
            lo_q, hi_q = max(0, -sq), min(H, H - sq)
            lo_r, hi_r = max(0, -sr), min(W, W - sr)
            src[lo_q:hi_q, lo_r:hi_r] = grid[lo_q + sq:hi_q + sq,
                                             lo_r + sr:hi_r + sr]
            lp += src * POW3_11[5 + u]
        lps.append(lp)

    n = len(cand_cells)
    win_idx = np.zeros((n, 18), dtype=np.int32)
    cls = np.full(n, -1, dtype=np.int32)
    for i, (q, r) in enumerate(cand_cells):
        gq, gr = q - q0, r - r0
        k = 0
        for d, (dq, dr) in enumerate(DIRS):
            for j in range(6):
                win_idx[i, k] = pats[d][gq - j * dq, gr - j * dr]
                k += 1
        l0, l1, l2 = lps[0][gq, gr], lps[1][gq, gr], lps[2][gq, gr]
        if l0 or l1 or l2:
            cb0, cb1, cb2 = CODEBOOK[l0], CODEBOOK[l1], CODEBOOK[l2]
            p0 = int(cb0 & 7) * 6 + int(cb0 >> 3)
            p1 = int(cb1 & 7) * 6 + int(cb1 >> 3)
            p2 = int(cb2 & 7) * 6 + int(cb2 >> 3)
            cls[i] = int(CANON_EMPTY[(p0 * 36 + p1) * 36 + p2])
    return win_idx, cls

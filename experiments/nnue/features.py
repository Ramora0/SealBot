"""Numpy feature extraction mirroring the engine's NNUE features exactly.

Encoding is root-relative: digit 1 = mover (root player), digit 2 = opponent.
Must stay in lockstep with current/engine/board.h (_init_eval_arrays) and
bot.h (_conj_class).
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_CB = np.load(os.path.join(SCRIPT_DIR, "codebook.npz"))
CODEBOOK = _CB["codebook"]          # uint8, us | them<<3
CANON_EMPTY = _CB["canon_empty"]    # uint16 [36^3]
CANON_OCC = _CB["canon_occ"]        # uint8  [6^3]
CLASS_MIRROR = _CB["class_mirror"]  # int32  [8548]
MIRROR729 = _CB["mirror729"]        # int32  [729]
NUM_CLASSES = int(_CB["num_classes"])
EMPTY_CLASSES = 8436
OCC_RANKS = 56

DIRS = [(1, 0), (0, 1), (1, -1)]
PAD = 11  # window anchors reach -5, lp reads reach +-5 beyond those cells

POW3_6 = 3 ** np.arange(6, dtype=np.int64)
POW3_11 = 3 ** np.arange(11, dtype=np.int64)


def extract_features(cells, mover):
    """cells: [(q, r, player 1|2)], mover: 1|2 (root player).

    Returns (w_idx, w_cnt, c_idx, c_cnt) int arrays: sparse counts of
    6-window pattern indices and conjunction class indices.
    """
    if not cells:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty(0, np.int64), np.empty(0, np.int64))

    qs = np.array([c[0] for c in cells])
    rs = np.array([c[1] for c in cells])
    ps = np.array([c[2] for c in cells])

    q0, r0 = qs.min() - PAD, rs.min() - PAD
    H = qs.max() - q0 + PAD + 1
    W = rs.max() - r0 + PAD + 1
    grid = np.zeros((H, W), dtype=np.int64)
    # root-relative digits
    digits = np.where(ps == mover, 1, 2)
    grid[qs - q0, rs - r0] = digits

    # ── 6-cell window pattern counts ──
    w_counts = np.zeros(729, dtype=np.int64)
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
        nz = pat[pat > 0]
        if len(nz):
            w_counts += np.bincount(nz, minlength=729)

    # ── 11-cell line patterns per direction ──
    lps = []
    for dq, dr in DIRS:
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
    lp0, lp1, lp2 = lps

    active = (lp0 | lp1 | lp2) != 0
    cb0, cb1, cb2 = CODEBOOK[lp0], CODEBOOK[lp1], CODEBOOK[lp2]

    empty = active & (grid == 0)
    p0 = (cb0 & 7).astype(np.int64) * 6 + (cb0 >> 3)
    p1 = (cb1 & 7).astype(np.int64) * 6 + (cb1 >> 3)
    p2 = (cb2 & 7).astype(np.int64) * 6 + (cb2 >> 3)
    cls_empty = CANON_EMPTY[(p0 * 36 + p1) * 36 + p2].astype(np.int64)

    own0 = np.where(grid == 1, cb0 & 7, cb0 >> 3).astype(np.int64)
    own1 = np.where(grid == 1, cb1 & 7, cb1 >> 3).astype(np.int64)
    own2 = np.where(grid == 1, cb2 & 7, cb2 >> 3).astype(np.int64)
    cls_occ = (EMPTY_CLASSES + (grid == 2) * OCC_RANKS
               + CANON_OCC[(own0 * 6 + own1) * 6 + own2])

    cls = np.where(grid == 0, cls_empty, cls_occ)[active]
    c_counts = np.bincount(cls, minlength=NUM_CLASSES)

    w_idx = np.nonzero(w_counts)[0]
    c_idx = np.nonzero(c_counts)[0]
    return w_idx, w_counts[w_idx], c_idx, c_counts[c_idx]


def net_forward(w_idx, w_cnt, c_idx, c_cnt, move_count, net):
    """Reference forward pass matching bot.h::_leaf_eval exactly."""
    acc = (net["ew"][w_idx] * w_cnt[:, None]).sum(axis=0) \
        + (net["ec"][c_idx] * c_cnt[:, None]).sum(axis=0)
    h = np.clip(acc, 0.0, net["clip"])
    g0 = move_count * 0.02
    x = np.concatenate([h, [g0]])
    hidden = np.maximum(net["w1"] @ x + net["b1"], 0.0)
    return float((net["w2"] @ hidden + net["b2"]) * net["out_scale"])

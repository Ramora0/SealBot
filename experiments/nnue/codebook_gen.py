"""Generate the conjunction-feature codebook shared by engine and trainer.

Line code semantics (per player p, over the 11-cell segment through a cell,
center at digit exponent 5; digit j <-> offset j-5 along the direction):

  The 6 windows through the center are starts s=0..5, cells [s, s+5].
  A window is LIVE for p if it contains no opponent stones.
  m = max p-stone count over live windows.

  EMPTY center:
    L0 dead      : no live window
    L1 space     : m <= 1
    L2 pair      : m == 2
    L3 closed-3  : m == 3 and one opponent stone refutes (a single empty cell,
                   other than the center, lies in every live count-3 window)
    L4 open-3    : m == 3, not refutable by one stone
    L5 win-square: m >= 4  (p completes six this turn through this cell)

  Center occupied by p: code = 0 if no live window else min(m, 5).
  Center occupied by opponent: code = 0 (all windows dead).

Class index layout:
  empty cells   : sorted multiset of 3 pair-codes (pair = us*6+them, 36 states)
                  -> 8436 classes, ids [0, 8436)
  occupied cells: sorted triple of owner codes (6 states) -> 56 ranks,
                  ids 8436 + owner_is_them*56 + rank -> [8436, 8548)

Outputs:
  codebook_data.h  (C tables for the engine)
  codebook.npz     (same tables + mirror permutations for the trainer)
"""

import itertools
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PAT = 3 ** 11          # 177147
CENTER = 5               # digit exponent of the center cell
NUM_EMPTY_CLASSES = None  # filled below
NUM_CLASSES = None


def gen_line_codes():
    """Return codes[177147] uint8, packed us | them << 3."""
    pats = np.arange(N_PAT, dtype=np.int64)
    digits = np.empty((N_PAT, 11), dtype=np.int8)
    x = pats.copy()
    for j in range(11):
        digits[:, j] = x % 3
        x //= 3

    center = digits[:, CENTER]

    codes = np.zeros((2, N_PAT), dtype=np.uint8)  # [player-1][pattern]

    # window stone counts: 6 windows, starts 0..5
    cnt = {1: np.zeros((N_PAT, 6), dtype=np.int8),
           2: np.zeros((N_PAT, 6), dtype=np.int8)}
    for s in range(6):
        w = digits[:, s:s + 6]
        cnt[1][:, s] = (w == 1).sum(axis=1)
        cnt[2][:, s] = (w == 2).sum(axis=1)

    for p in (1, 2):
        o = 3 - p
        live = cnt[o] == 0                       # (N,6)
        mc = np.where(live, cnt[p], -1)          # masked counts
        m = mc.max(axis=1)                       # max live count, -1 if none

        code = np.zeros(N_PAT, dtype=np.uint8)

        occ_own = center == p
        occ_opp = center == o
        empty = center == 0

        # occupied by p: 0 if no live window else min(m, 5)
        code[occ_own] = np.clip(m[occ_own], 0, 5).astype(np.uint8)
        code[occ_own & (m < 0)] = 0
        # occupied by opponent: 0 (default)

        # empty center
        e = empty
        code[e & (m < 0)] = 0                    # L0
        code[e & (m >= 0) & (m <= 1)] = 1        # L1
        code[e & (m == 2)] = 2                   # L2
        code[e & (m >= 4)] = 5                   # L5

        # L3 vs L4: refutability check for m == 3, empty center
        idx3 = np.where(e & (m == 3))[0]
        for pi in idx3:
            d = digits[pi]
            threat_windows = []
            for s in range(6):
                if cnt[o][pi, s] == 0 and cnt[p][pi, s] == 3:
                    # empties of this window excluding the center
                    empties = frozenset(
                        s + k for k in range(6)
                        if d[s + k] == 0 and (s + k) != CENTER)
                    threat_windows.append(empties)
            # single stone refutes iff some cell lies in every threat window
            common = frozenset.intersection(*threat_windows)
            code[pi] = 3 if common else 4
        codes[p - 1] = code

    return (codes[0] | (codes[1] << 3)).astype(np.uint8)


def gen_canon_tables():
    """canon_empty[36^3] -> class id; canon_occ[6^3] -> rank."""
    # empty: multisets of 3 pair-codes from 36 states
    multisets = list(itertools.combinations_with_replacement(range(36), 3))
    ms_rank = {ms: i for i, ms in enumerate(multisets)}
    canon_empty = np.empty(36 ** 3, dtype=np.uint16)
    for a in range(36):
        for b in range(36):
            base = (a * 36 + b) * 36
            for c in range(36):
                canon_empty[base + c] = ms_rank[tuple(sorted((a, b, c)))]

    occ_multisets = list(itertools.combinations_with_replacement(range(6), 3))
    occ_rank = {ms: i for i, ms in enumerate(occ_multisets)}
    canon_occ = np.empty(6 ** 3, dtype=np.uint8)
    for a in range(6):
        for b in range(6):
            for c in range(6):
                canon_occ[(a * 6 + b) * 6 + c] = occ_rank[tuple(sorted((a, b, c)))]

    return canon_empty, canon_occ, multisets, occ_multisets


def gen_mirror_perms(multisets, occ_multisets):
    """Class permutation under us<->them swap, and 729-window mirror."""
    n_empty = len(multisets)
    n_occ = len(occ_multisets)
    num_classes = n_empty + 2 * n_occ

    ms_rank = {ms: i for i, ms in enumerate(multisets)}
    class_mirror = np.empty(num_classes, dtype=np.int32)
    for i, ms in enumerate(multisets):
        sw = tuple(sorted((p % 6) * 6 + (p // 6) for p in ms))  # swap us/them
        class_mirror[i] = ms_rank[sw]
    for r in range(n_occ):
        class_mirror[n_empty + r] = n_empty + n_occ + r          # owner flips
        class_mirror[n_empty + n_occ + r] = n_empty + r

    mirror729 = np.empty(729, dtype=np.int32)
    for i in range(729):
        x, m, p3 = i, 0, 1
        for _ in range(6):
            d = x % 3
            if d == 1: d = 2
            elif d == 2: d = 1
            m += d * p3
            p3 *= 3
            x //= 3
        mirror729[i] = m
    return class_mirror, mirror729, num_classes


def emit_header(path, cb, canon_empty, canon_occ, num_classes):
    def fmt_array(name, ctype, arr):
        vals = ",".join(str(int(v)) for v in arr)
        return f"static const {ctype} {name}[{len(arr)}] = {{{vals}}};\n"

    with open(path, "w") as f:
        f.write("// Generated by codebook_gen.py -- do not edit.\n")
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"static constexpr int CONJ_NUM_CLASSES = {num_classes};\n")
        f.write(f"static constexpr int CONJ_EMPTY_CLASSES = 8436;\n")
        f.write(f"static constexpr int CONJ_OCC_RANKS = 56;\n")
        f.write(f"static constexpr int LP_LEN = 11;\n")
        f.write(f"static constexpr int LP_CENTER = 5;\n\n")
        f.write(fmt_array("LINE_CODEBOOK", "uint8_t", cb))
        f.write(fmt_array("CANON_EMPTY", "uint16_t", canon_empty))
        f.write(fmt_array("CANON_OCC", "uint8_t", canon_occ))
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    print("generating line codes (3^11 patterns)...")
    cb = gen_line_codes()
    canon_empty, canon_occ, multisets, occ_multisets = gen_canon_tables()
    class_mirror, mirror729, num_classes = gen_mirror_perms(
        multisets, occ_multisets)
    print(f"classes: {len(multisets)} empty + 2x{len(occ_multisets)} occupied "
          f"= {num_classes}")

    # code distribution sanity
    us = cb & 7
    them = cb >> 3
    print("us-code histogram:", np.bincount(us, minlength=6).tolist())
    print("them-code histogram:", np.bincount(them, minlength=6).tolist())

    emit_header(os.path.join(SCRIPT_DIR, "codebook_data.h"),
                cb, canon_empty, canon_occ, num_classes)
    np.savez_compressed(
        os.path.join(SCRIPT_DIR, "codebook.npz"),
        codebook=cb, canon_empty=canon_empty, canon_occ=canon_occ,
        class_mirror=class_mirror, mirror729=mirror729,
        num_classes=np.int32(num_classes))
    print("wrote codebook.npz")


if __name__ == "__main__":
    main()

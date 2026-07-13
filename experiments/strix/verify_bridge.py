"""Verify the strix bridge: POV, translation invariance, label agreement.

1. POV: mover with an open four + 2 stones to play -> value ~ +1;
   same board, other side to move -> strongly negative.
2. Translation invariance of the value.
3. Agreement with our deep search labels on sampled gen0_deep positions:
   sign agreement on decided, spearman on quiet — compare with the old
   linear eval and our champion net numbers.
4. Throughput estimate for full-dataset relabeling.
"""

import glob
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells, value_batch

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")


def main():
    model, mc, gc = load_strix()
    print("strix loaded")

    # --- 1. POV sanity ---------------------------------------------------
    # A: open four on the r-axis through the origin; B: scattered pair.
    cells = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (3, -3, 2), (4, -3, 2),
             (-3, 5, 2), (5, 1, 2)]
    s_a = state_from_cells(cells, mover=1, moves_left=2, gc=gc)   # A to win
    s_b = state_from_cells(cells, mover=2, moves_left=2, gc=gc)   # B to defend
    va, vb = value_batch(model, mc, [s_a, s_b])
    print(f"open-four, A(mover) to move: {va:+.3f}  (expect ~+1)")
    print(f"open-four, B(defender) to move: {vb:+.3f}  (expect negative)")
    assert va > 0.5, "POV check failed: winning mover should be strongly +"

    # --- 2. Translation invariance ---------------------------------------
    t = [(q + 3, r - 2, p) for q, r, p in cells]
    vt = value_batch(model, mc, [state_from_cells(t, 1, 2, gc)])[0]
    print(f"translated by (3,-2): {vt:+.3f}  (delta {abs(vt-va):.4f})")

    # --- 3. Agreement with deep labels ------------------------------------
    rng = random.Random(0)
    files = sorted(glob.glob(str(SEAL / "experiments/nnue/data/gen0_deep/*.pkl")))
    rng.shuffle(files)
    recs = []
    for f in files[:30]:
        for g in pickle.load(open(f, "rb")):
            for p in g["positions"]:
                recs.append((p, g["winner"]))
    rng.shuffle(recs)
    recs = recs[:4000]

    states, ys, outs, movers = [], [], [], []
    skipped = 0
    for p, w in recs:
        s = state_from_cells(p["cells"], p["mover"], p["moves_left"], gc)
        if s is None:
            skipped += 1
            continue
        states.append(s)
        ys.append(p["score"])
        outs.append(0 if w == 0 else (1 if w == p["mover"] else -1))
        movers.append(p["mover"])
    t0 = time.time()
    vs = value_batch(model, mc, states)
    dt = time.time() - t0
    v = np.array(vs)
    y = np.array(ys)
    out = np.array(outs)
    print(f"\n{len(v)} positions in {dt:.1f}s ({len(v)/dt:.0f} pos/s), "
          f"{skipped} skipped")

    dec = np.abs(y) > 2000
    quiet = np.abs(y) < 25000
    print(f"sign agreement vs deep label (|y|>2000): "
          f"{np.mean(np.sign(v[dec]) == np.sign(y[dec])):.3f}")
    print(f"sign agreement vs GAME OUTCOME (decided games): "
          f"{np.mean(np.sign(v[out != 0]) == out[out != 0]):.3f}")

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    yq = np.clip(y[quiet], -8000, 8000)
    print(f"spearman vs deep label (quiet): {spearman(v[quiet], yq):.3f}")
    print(f"value distribution: mean {v.mean():+.3f}, std {v.std():.3f}, "
          f"|v|>0.9: {np.mean(np.abs(v) > 0.9):.2%}")

    n_full = 693185 + 460634
    print(f"\nfull relabel estimate: {n_full / (len(v)/dt) / 60:.0f} min")


if __name__ == "__main__":
    main()

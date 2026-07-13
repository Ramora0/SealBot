"""Probe solve_forcing/solve_defense speed + hit rate on gen0 positions,
and agreement with the deep mate labels (|score|>1e7)."""

import glob
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")


def main():
    import hexo_rs
    ck_gc = hexo_rs.GameConfig(win_length=6, placement_radius=6, max_moves=300)

    rng = random.Random(3)
    files = sorted(glob.glob(str(SEAL / "experiments/nnue/data/gen0_deep/*.pkl")))
    rng.shuffle(files)
    recs = []
    for f in files[:10]:
        for g in pickle.load(open(f, "rb")):
            recs.extend(g["positions"])
    rng.shuffle(recs)

    mate = [p for p in recs if p["score"] > 1e7][:300]
    mated = [p for p in recs if p["score"] < -1e7][:300]
    quiet = [p for p in recs if abs(p["score"]) < 25000][:300]

    for name, group, budget in [("mate-for-mover", mate, 200_000),
                                ("mated (opp wins)", mated, 200_000),
                                ("quiet", quiet, 200_000)]:
        t0 = time.time()
        hits = 0
        threat_hits = 0
        for p in group:
            s = state_from_cells(p["cells"], p["mover"], p["moves_left"], ck_gc)
            if s is None:
                continue
            if hexo_rs.solve_forcing(s, 30, budget) is not None:
                hits += 1
            elif hexo_rs.solve_threat(s, 30, budget) is not None:
                threat_hits += 1
        dt = time.time() - t0
        print(f"{name:18s} n={len(group)}  forcing-win {hits/len(group):.2%}  "
              f"opp-threat {threat_hits/len(group):.2%}  "
              f"{1000*dt/len(group):.1f} ms/pos")


if __name__ == "__main__":
    main()

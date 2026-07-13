"""Stratify strix-value agreement with deep labels by label confidence,
move count, and compare against the champion net's static eval on the
SAME positions (via the 3.13 current/ module's eval hook if available,
else the numpy sidecar forward)."""

import glob
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells, value_batch

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")
sys.path.insert(0, str(SEAL / "experiments/nnue"))
from features import extract_features, net_forward  # noqa: E402


def main():
    model, mc, gc = load_strix()

    rng = random.Random(1)
    files = sorted(glob.glob(str(SEAL / "experiments/nnue/data/gen0_deep/*.pkl")))
    rng.shuffle(files)
    recs = []
    for f in files[:40]:
        for g in pickle.load(open(f, "rb")):
            for p in g["positions"]:
                recs.append((p, g["winner"]))
    rng.shuffle(recs)
    recs = recs[:8000]

    net = dict(np.load(SEAL / "current/net_data.h.npz"))
    text = open(SEAL / "current/pattern_data.h").read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    pv_lin = np.array([float(x) for x in nums])

    states, y, champ, mc_arr = [], [], [], []
    for p, w in recs:
        s = state_from_cells(p["cells"], p["mover"], p["moves_left"], gc)
        if s is None:
            continue
        states.append(s)
        y.append(p["score"])
        w_idx, w_cnt, c_idx, c_cnt = extract_features(p["cells"], p["mover"])
        ev = net_forward(w_idx, w_cnt, c_idx, c_cnt, p["move_count"],
                         p["moves_left"] * 0.5, net)
        ev += float(net.get("lin_blend", 0.0)) * float((pv_lin[w_idx] * w_cnt).sum())
        champ.append(ev)
        mc_arr.append(p["move_count"])
    v = np.array(value_batch(model, mc, states))
    y = np.array(y)
    champ = np.array(champ)
    mc_arr = np.array(mc_arr)
    print(f"{len(v)} positions")

    def band(name, mask):
        if mask.sum() < 30:
            print(f"{name}: n={mask.sum()} (skip)")
            return
        sa_strix = np.mean(np.sign(v[mask]) == np.sign(y[mask]))
        sa_champ = np.mean(np.sign(champ[mask]) == np.sign(y[mask]))
        print(f"{name:34s} n={mask.sum():5d}  sign: strix {sa_strix:.3f}  "
              f"champ {sa_champ:.3f}")

    ay = np.abs(y)
    band("MATE labels (|y|>1e7)", ay > 1e7)
    band("near-mate (1e5<|y|<1e7)", (ay > 1e5) & (ay < 1e7))
    band("strong (25k<|y|<1e5)", (ay > 25000) & (ay < 1e5))
    band("decided-quiet (2k<|y|<25k)", (ay > 2000) & (ay < 25000))
    band("early (mc<12, |y|>2k)", (mc_arr < 12) & (ay > 2000))
    band("mid (12<=mc<30, |y|>2k)", (mc_arr >= 12) & (mc_arr < 30) & (ay > 2000))
    band("late (mc>=30, |y|>2k)", (mc_arr >= 30) & (ay > 2000))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    quiet = ay < 25000
    yq = np.clip(y[quiet], -8000, 8000)
    print(f"\nspearman quiet: strix {spearman(v[quiet], yq):.3f}   "
          f"champ {spearman(champ[quiet], yq):.3f}")
    print(f"strix-vs-champ value corr (all): "
          f"{np.corrcoef(v, np.tanh(champ/5000))[0,1]:.3f}")


if __name__ == "__main__":
    main()

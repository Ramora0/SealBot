"""Offline metrics for a trained tiny-net checkpoint on a dataset cache.

Reproduces train.py's val split (default_rng(0), val_frac 0.03) and reports,
on held-out rows: correlation with the strix target, sign/spearman vs the
deep search label, sign vs game outcome — side by side with the champion
hybrid eval computed on the same rows.

Run in the SealBot venv:
    ../../.venv/bin/python eval_offline.py --ckpt ../nnue/output/strix_gen0/net.pt \
        --cache ../nnue/output/strix_gen0/dataset_v2.npz
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")


def forward_rows(weights, d, rows, lin_blend=0.0, pv_lin=None):
    ew, ec = weights["ew"], weights["ec"]
    offs, fidx, fcnt = d["offsets"], d["feat_idx"], d["feat_cnt"]
    n, K = len(rows), ew.shape[1]
    acc = np.zeros((n, K))
    lin = np.zeros(n)
    for k, i in enumerate(rows):
        sl = slice(offs[i], offs[i + 1])
        idx, cnt = fidx[sl], fcnt[sl].astype(np.float64)
        wm = idx < 729
        acc[k] = ew[idx[wm]].T @ cnt[wm] + ec[idx[~wm] - 729].T @ cnt[~wm]
        if pv_lin is not None:
            lin[k] = (pv_lin[idx[wm]] * cnt[wm]).sum()
    x = np.hstack([np.clip(acc, 0, weights["clip"]),
                   (d["move_count"][rows] * 0.02)[:, None],
                   (d["moves_left"][rows] * 0.5)[:, None]])
    h = np.maximum(x @ weights["w1"].T + weights["b1"], 0.0)
    out = h @ weights["w2"] + weights["b2"]
    return out * weights["out_scale"] + lin_blend * lin


def load_ckpt(path):
    sd = torch.load(path, map_location="cpu")
    return {
        "ew": sd["ew.weight"].numpy().astype(np.float64),
        "ec": sd["ec.weight"].numpy().astype(np.float64),
        "w1": sd["w1.weight"].numpy().astype(np.float64),
        "b1": sd["w1.bias"].numpy().astype(np.float64),
        "w2": sd["w2.weight"].numpy().reshape(-1).astype(np.float64),
        "b2": float(sd["w2.bias"].numpy().reshape(-1)[0]),
        "out_scale": float(sd["out_scale"]),
        "clip": float(sd["clip"]),
    }


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def report(tag, ev, strix_score, deep, out):
    dec = np.abs(deep) > 2000
    quiet = np.abs(deep) < 25000
    dq = np.clip(deep[quiet], -8000, 8000)
    played = out != 0.5
    print(f"  {tag}:")
    print(f"    corr vs strix target:      "
          f"{np.corrcoef(ev, strix_score)[0,1]:+.3f}   "
          f"spearman {spearman(ev, strix_score):+.3f}")
    print(f"    vs deep label:  sign(dec) {np.mean(np.sign(ev[dec]) == np.sign(deep[dec])):.3f}   "
          f"spearman(quiet) {spearman(ev[quiet], dq):+.3f}")
    print(f"    sign vs outcome:           "
          f"{np.mean(np.sign(ev[played]) == np.sign(out[played]*2-1)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()

    d = np.load(args.cache)
    n = len(d["score"])
    rng = np.random.default_rng(0)
    val_ids = rng.permutation(n)[:int(n * args.val_frac)]
    rows = np.sort(val_ids[:args.n])
    print(f"{len(rows)} held-out rows of {n}")

    strix_score = d["score"][rows].astype(np.float64)
    deep = d["score_deep"][rows].astype(np.float64) if "score_deep" in d \
        else strix_score
    out = d["outcome"][rows].astype(np.float64)

    w = load_ckpt(args.ckpt)
    ev = forward_rows(w, d, rows)
    report(f"ckpt {args.ckpt}", ev, strix_score, deep, out)

    champ_npz = dict(np.load(SEAL / "champion_frozen/net_data.h.npz"))
    text = open(SEAL / "champion_frozen/pattern_data.h").read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    pv_lin = np.array([float(x) for x in nums])
    cw = {k: champ_npz[k].astype(np.float64) for k in
          ("ew", "ec", "w1", "b1", "w2")}
    cw.update(b2=float(champ_npz["b2"]), out_scale=float(champ_npz["out_scale"]),
              clip=float(champ_npz["clip"]))
    ev_c = forward_rows(cw, d, rows, lin_blend=float(champ_npz["lin_blend"]),
                        pv_lin=pv_lin)
    report("champion hybrid (reference)", ev_c, strix_score, deep, out)


if __name__ == "__main__":
    main()

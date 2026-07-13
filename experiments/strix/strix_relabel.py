"""Relabel SealBot self-play shards with strix single-forward values.

Reads data/<src> shards, writes data/<dst> shards with the same structure:
score := strix_value * 8000 (mover POV), original deep score preserved as
"score_deep". With train.py --loss score --lam 1.0 the target becomes
strix_value * 8 in net-output units.

Usage (hexo venv):
    python strix_relabel.py --src gen0_deep --dst gen0_strix
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells, value_batch

DATA = Path("/users/PAS2836/leedavis/personal/SealBot/experiments/nnue/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale", type=float, default=8000.0)
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    model, mc, gc = load_strix()
    src_dir, dst_dir = DATA / args.src, DATA / args.dst
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(str(src_dir / "*.pkl")))
    total = skipped = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        out_path = dst_dir / os.path.basename(f)
        if out_path.exists():
            continue
        games = pickle.load(open(f, "rb"))
        states, refs = [], []
        for g in games:
            for p in g["positions"]:
                s = state_from_cells(p["cells"], p["mover"], p["moves_left"], gc)
                if s is None:
                    p["_drop"] = True
                    skipped += 1
                    continue
                states.append(s)
                refs.append(p)
        vals = value_batch(model, mc, states, chunk=args.chunk)
        for p, v in zip(refs, vals):
            p["score_deep"] = p["score"]
            p["score"] = float(v) * args.scale
            p["strix_v"] = float(v)
        for g in games:
            g["positions"] = [p for p in g["positions"] if not p.get("_drop")]
        total += len(refs)
        tmp = str(out_path) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(games, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp, out_path)
        if (fi + 1) % 20 == 0 or fi + 1 == len(files):
            rate = total / max(time.time() - t0, 1e-9)
            print(f"[{fi+1}/{len(files)}] {total} pos, {skipped} skipped, "
                  f"{rate:.0f} pos/s", flush=True)
    print(f"DONE {total} positions, {skipped} skipped -> {dst_dir}")


if __name__ == "__main__":
    main()

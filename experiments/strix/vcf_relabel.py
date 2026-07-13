"""Upgrade strix-labeled shards with VCF solver overrides.

For each position: if hexo_rs.solve_forcing proves a fully-forcing win for
the mover, score := +8000 (exact, beats any static value); otherwise the
strix score is kept. solve_forcing releases the GIL, so a thread pool
scales across cores.

Usage (hexo venv):
    python vcf_relabel.py --src gen0_strix --dst gen0_strix_vcf \
        --depth 24 --budget 60000 --threads 14
"""

import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import state_from_cells

DATA = Path("/users/PAS2836/leedavis/personal/SealBot/experiments/nnue/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--budget", type=int, default=60_000)
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()

    import hexo_rs
    gc = hexo_rs.GameConfig(win_length=6, placement_radius=6, max_moves=300)
    src_dir, dst_dir = DATA / args.src, DATA / args.dst
    os.makedirs(dst_dir, exist_ok=True)

    def solve_one(p):
        s = state_from_cells(p["cells"], p["mover"], p["moves_left"], gc)
        if s is None:
            return False
        return hexo_rs.solve_forcing(s, args.depth, args.budget) is not None

    files = sorted(glob.glob(str(src_dir / "*.pkl")))
    total = wins = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for fi, f in enumerate(files):
            out_path = dst_dir / os.path.basename(f)
            if out_path.exists():
                continue
            games = pickle.load(open(f, "rb"))
            positions = [p for g in games for p in g["positions"]]
            for p, won in zip(positions, pool.map(solve_one, positions,
                                                  chunksize=16)):
                if won:
                    p["score"] = 8000.0
                    p["vcf_win"] = True
                total += 1
                wins += bool(won)
            tmp = str(out_path) + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(games, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp, out_path)
            if (fi + 1) % 10 == 0 or fi + 1 == len(files):
                rate = total / max(time.time() - t0, 1e-9)
                print(f"[{fi+1}/{len(files)}] {total} pos, "
                      f"{wins/max(total,1):.1%} vcf wins, {rate:.0f} pos/s",
                      flush=True)
    print(f"DONE {total} positions, {wins} vcf wins -> {dst_dir}")


if __name__ == "__main__":
    main()

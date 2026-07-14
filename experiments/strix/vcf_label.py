"""Stamp proven-forced-win flags onto training positions (v1.4 labels).

For every position in policy_targets/ and human_targets/ shards, probe the
in-engine VCF solver (mover to play): +1 => the mover has a PROVEN forced
win, so the value target should saturate (+1 tanh) regardless of what the
strix single-forward said. Writes sidecar vcf_targets/<shard>.npy int8
arrays aligned to the shard's meta order (1 = mover forced win, 0 = not
proven). trunk_train2 --vcf-labels applies the override at build time.

Run in the SealBot venv:
    ../../.venv/bin/python vcf_label.py --workers 14
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEAL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT = os.path.join(SCRIPT_DIR, "vcf_targets")


def _shard(path):
    sys.path.insert(0, SEAL)
    sys.path.insert(0, os.path.join(SEAL, "current"))
    from game import HexGame, Player
    import minimax_cpp
    bot = minimax_cpp.MinimaxBot(0.1)
    bot.vcf_node_budget = 3000

    d = np.load(path, allow_pickle=True)
    metas = d["meta"]
    flags = np.zeros(len(metas), dtype=np.int8)
    for i, (cells, mover, ml, mc) in enumerate(metas):
        game = HexGame(win_length=6)
        for q, r, p in cells:
            game.board[(int(q), int(r))] = Player(int(p))
        game.current_player = Player(int(mover))
        game.moves_left_in_turn = int(ml)
        game.move_count = int(mc)
        res, _ = bot.forced_win(game, 6)
        if res == 1:
            flags[i] = 1
    tag = ("g_" if "policy_targets" in path else "h_") + \
        os.path.basename(path).replace(".npz", ".npy")
    np.save(os.path.join(OUT, tag), flags)
    return len(metas), int(flags.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    shards = (sorted(glob.glob(os.path.join(SCRIPT_DIR, "policy_targets",
                                            "*.npz")))
              + sorted(glob.glob(os.path.join(SCRIPT_DIR, "human_targets",
                                              "*.npz"))))
    todo = []
    for p in shards:
        tag = ("g_" if "policy_targets" in p else "h_") + \
            os.path.basename(p).replace(".npz", ".npy")
        if not os.path.exists(os.path.join(OUT, tag)):
            todo.append(p)
    print(f"{len(todo)} shards to label")
    import multiprocessing as mp
    t0 = time.time()
    total = wins = 0
    with mp.Pool(args.workers) as pool:
        for i, (n, w) in enumerate(pool.imap_unordered(_shard, todo)):
            total += n
            wins += w
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f"[{i+1}/{len(todo)}] {total} pos, {wins} forced wins "
                      f"({100.0*wins/max(total,1):.1f}%), "
                      f"{time.time()-t0:.0f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

"""Re-label stored positions with deeper searches from the original engine.

Reads shards from --in, replaces each position's "score" with the root score
of a fixed-time search (best/ engine = original SealBot), writes shards with
the same structure to --out. Winner/outcome fields pass through unchanged.

Usage:
    python relabel.py --in data/gen0 --out data/gen0_deep --tl 0.12 --workers 30
"""

import argparse
import glob
import os
import pickle
import sys
import time
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _worker(args):
    shard_path, out_dir, bot_dir, tl = args
    sys.path.insert(0, bot_dir)
    sys.path.insert(0, ROOT_DIR)
    from game import HexGame, Player
    from minimax_cpp import MinimaxBot

    out_path = os.path.join(out_dir, os.path.basename(shard_path))
    if os.path.exists(out_path):
        return os.path.basename(shard_path), 0, 0.0  # resume support

    bot = MinimaxBot(tl)
    with open(shard_path, "rb") as f:
        games = pickle.load(f)

    t0 = time.time()
    n = 0
    for g in games:
        for pos in g["positions"]:
            game = HexGame(win_length=6)
            for q, r, p in pos["cells"]:
                game.board[(q, r)] = Player.A if p == 1 else Player.B
            game.current_player = Player.A if pos["mover"] == 1 else Player.B
            game.moves_left_in_turn = pos["moves_left"]
            game.move_count = pos["move_count"]
            bot.time_limit = tl
            moves = bot.get_move(game)
            if moves:
                pos["score"] = bot.last_score
                pos["depth"] = bot.last_depth
            n += 1

    with open(out_path + ".tmp", "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(out_path + ".tmp", out_path)
    return os.path.basename(shard_path), n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/gen0")
    ap.add_argument("--out", type=str, default="data/gen0_deep")
    ap.add_argument("--bot-dir", type=str, default=os.path.join(ROOT_DIR, "best"))
    ap.add_argument("--tl", type=float, default=0.12)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    in_dir = os.path.join(SCRIPT_DIR, args.inp)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(in_dir, "*.pkl")))
    if args.max_shards:
        shards = shards[:args.max_shards]
    tasks = [(s, out_dir, args.bot_dir, args.tl) for s in shards]
    print(f"relabeling {len(shards)} shards at tl={args.tl}s "
          f"({args.workers} workers)")

    t0 = time.time()
    done = pos_total = 0
    with mp.Pool(args.workers) as pool:
        for name, n, dt in pool.imap_unordered(_worker, tasks):
            done += 1
            pos_total += n
            if done % 10 == 0:
                el = time.time() - t0
                eta = el / done * (len(shards) - done)
                print(f"  {done}/{len(shards)} shards, {pos_total} positions, "
                      f"{el:.0f}s elapsed, ETA {eta/60:.0f}m", flush=True)
    print(f"done: {pos_total} positions in {(time.time()-t0)/60:.0f}m")


if __name__ == "__main__":
    main()

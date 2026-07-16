"""Self-play data generation for eval training.

Plays engine-vs-engine games (best/ module = original SealBot by default),
with randomized openings for diversity, and records every searched position
with the mover's root search score and the final game outcome.

Each worker writes its own shard files, so the run is checkpointed and
partial results are always usable.

Usage:
    python datagen.py --games 10000 --out data/gen0 --workers 32
    python datagen.py --games 200 --out data/pilot      # quick pilot
"""

import argparse
import os
import pickle
import random
import sys
import time
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)

MAX_MOVES = 200

# Radius-2 hex offsets for random-opening placement near existing stones
_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def _random_opening(game, rng, n_stones):
    """Play n_stones random moves through the game API (keeps turn structure)."""
    for _ in range(n_stones):
        if game.game_over:
            return False
        if not game.board:
            cands = [(0, 0)]
        else:
            cands = list({(q + dq, r + dr)
                          for q, r in game.board for dq, dr in _D2
                          if (q + dq, r + dr) not in game.board})
        if not cands:
            return False
        game.make_move(*rng.choice(cands))
    return not game.game_over


def _play_one_game(bot, game_mod, rng, tl_min, tl_max, open_min, open_max,
                   rand_prob):
    """Play one self-play game; return list of position records + winner."""
    HexGame, Player = game_mod.HexGame, game_mod.Player
    game = HexGame(win_length=6)

    n_open = rng.randint(open_min, open_max)
    if not _random_opening(game, rng, n_open):
        return None

    records = []
    total_moves = game.move_count

    while not game.game_over and total_moves < MAX_MOVES:
        if rand_prob > 0 and rng.random() < rand_prob:
            # Off-path injection: play this turn randomly and record
            # nothing; every later position still gets a full-budget
            # label, so the corpus samples states the search tree visits
            # but game paths never reach (gensfen random_move analog).
            k = game.moves_left_in_turn
            if not _random_opening(game, rng, k):
                break
            total_moves += k
            continue
        mover = game.current_player
        bot.time_limit = rng.uniform(tl_min, tl_max)
        moves = bot.get_move(game)
        if not moves:
            break
        # Snapshot position BEFORE the move, with the search's root score
        records.append({
            "cells": [(q, r, 1 if p == Player.A else 2)
                      for (q, r), p in game.board.items()],
            "mover": 1 if mover == Player.A else 2,
            "moves_left": game.moves_left_in_turn,
            "move_count": game.move_count,
            "score": bot.last_score,   # mover's (root) perspective
            "depth": bot.last_depth,
        })
        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                return records, 0  # illegal move: treat as draw, keep data
        total_moves += len(moves)

    winner = 0
    if game.winner == Player.A:
        winner = 1
    elif game.winner == Player.B:
        winner = 2
    return records, winner


def _worker(args):
    (worker_id, n_games, out_dir, bot_dir, tl_min, tl_max,
     open_min, open_max, rand_prob, seed, shard_size) = args

    sys.path.insert(0, bot_dir)
    sys.path.insert(0, ROOT_DIR)
    import game as game_mod
    from minimax_cpp import MinimaxBot

    rng = random.Random(seed)
    bot = MinimaxBot(tl_min)

    shard, shard_idx, games_done, positions = [], 0, 0, 0
    t0 = time.time()

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        path = os.path.join(out_dir, f"w{worker_id:03d}_s{shard_idx:04d}.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(path + ".tmp", path)
        shard = []
        shard_idx += 1

    for gi in range(n_games):
        result = _play_one_game(bot, game_mod, rng, tl_min, tl_max,
                                open_min, open_max, rand_prob)
        if result is None:
            continue
        records, winner = result
        if not records:
            continue
        shard.append({"winner": winner, "positions": records})
        games_done += 1
        positions += len(records)
        if len(shard) >= shard_size:
            flush()

    flush()
    return worker_id, games_done, positions, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--out", type=str, default="data/gen0")
    ap.add_argument("--bot-dir", type=str, default=os.path.join(ROOT_DIR, "best"),
                    help="Engine module dir (default: best/ = original)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tl-min", type=float, default=0.02)
    ap.add_argument("--tl-max", type=float, default=0.05)
    ap.add_argument("--open-min", type=int, default=2)
    ap.add_argument("--open-max", type=int, default=10)
    ap.add_argument("--rand-move-prob", type=float, default=0.0,
                    help="Per-turn probability of playing a random "
                         "unrecorded turn (off-path state injection)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard-size", type=int, default=200)
    args = ap.parse_args()

    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    per_worker = (args.games + args.workers - 1) // args.workers
    tasks = [(w, per_worker, out_dir, args.bot_dir, args.tl_min, args.tl_max,
              args.open_min, args.open_max, args.rand_move_prob,
              args.seed + 7919 * w, args.shard_size)
             for w in range(args.workers)]

    print(f"Generating ~{per_worker * args.workers} games "
          f"({args.workers} workers, tl {args.tl_min}-{args.tl_max}s, "
          f"openings {args.open_min}-{args.open_max} stones)")
    print(f"Output: {out_dir}")

    t0 = time.time()
    total_games = total_pos = 0
    with mp.Pool(args.workers) as pool:
        for wid, g, p, dt in pool.imap_unordered(_worker, tasks):
            total_games += g
            total_pos += p
            elapsed = time.time() - t0
            print(f"  worker {wid:3d} done: {g} games, {p} positions "
                  f"({dt:.0f}s) | cumulative {total_games} games, "
                  f"{total_pos} positions, {elapsed:.0f}s", flush=True)

    print(f"\nDone: {total_games} games, {total_pos} positions "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

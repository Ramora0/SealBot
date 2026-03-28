#!/usr/bin/env python3
"""Benchmark test: best/ vs current/ at fixed depth on random positions.

Generates reproducible random mid-game positions and runs both engines
at a fixed search depth (default 4), comparing move choices and speed.

Usage:
    python test_depth.py                # 10 positions, depth 4
    python test_depth.py -n 20 -d 6     # 20 positions, depth 6
    python test_depth.py --seed 123     # custom seed
"""

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.dirname(__file__))
CURRENT_DIR = os.path.join(ROOT, "current")
BEST_DIR = os.path.join(ROOT, "best")

sys.path.insert(0, ROOT)
from game import HexGame, Player

# Hex neighbor offsets (distance <= 2)
_D2_OFFSETS = [
    (dq, dr)
    for dq in range(-2, 3)
    for dr in range(-2, 3)
    if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)
]


def generate_position(rng, num_stones):
    """Play num_stones random moves and return the game if not over."""
    game = HexGame(win_length=6)
    for _ in range(num_stones):
        if game.game_over:
            return None
        if not game.board:
            candidates = [(0, 0)]
        else:
            candidates = []
            for q, r in game.board:
                for dq, dr in _D2_OFFSETS:
                    nb = (q + dq, r + dr)
                    if nb not in game.board:
                        candidates.append(nb)
            candidates = list(set(candidates))
        if not candidates:
            return None
        game.make_move(*rng.choice(candidates))
    if game.game_over:
        return None
    return game


def generate_positions(num_positions, seed=42):
    """Generate reproducible random mid-game positions."""
    rng = random.Random(seed)
    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < num_positions * 10:
        attempts += 1
        num_stones = rng.randint(8, 20)
        game = generate_position(rng, num_stones)
        if game is not None:
            positions.append(game)
    return positions


def serialize_position(game):
    """Convert a HexGame to a JSON-serializable dict."""
    cells = []
    for (q, r), player in game.board.items():
        cells.append({"q": q, "r": r, "p": 1 if player == Player.A else 2})
    return {
        "cells": cells,
        "cur_player": 1 if game.current_player == Player.A else 2,
        "moves_left": game.moves_left_in_turn,
        "move_count": game.move_count,
    }


# Subprocess worker: loads one bot, runs all positions at fixed depth.
# Reads JSON positions from stdin, writes JSON results to stdout.
_WORKER = textwrap.dedent(r'''
    import sys, json, time

    bot_dir = sys.argv[1]
    max_depth = int(sys.argv[2])
    root_dir = sys.argv[3]

    sys.path.insert(0, root_dir)
    sys.path.insert(0, bot_dir)

    import minimax_cpp
    from game import HexGame, Player

    positions = json.loads(sys.stdin.read())
    bot = minimax_cpp.MinimaxBot(999.0)
    bot.max_depth = max_depth

    results = []
    for pos in positions:
        game = HexGame(win_length=6)
        game.board = {}
        for c in pos["cells"]:
            game.board[(c["q"], c["r"])] = Player.A if c["p"] == 1 else Player.B
        game.current_player = Player.A if pos["cur_player"] == 1 else Player.B
        game.moves_left_in_turn = pos["moves_left"]
        game.move_count = pos["move_count"]

        bot._nodes = 0
        t0 = time.perf_counter()
        moves = bot.get_move(game)
        elapsed = time.perf_counter() - t0

        results.append({
            "moves": [[q, r] for q, r in moves],
            "time_ms": elapsed * 1000,
            "depth": bot.last_depth,
            "score": bot.last_score,
            "nodes": bot._nodes,
        })

    print(json.dumps(results))
''')


def run_bot(bot_dir, positions_json, max_depth):
    """Run a bot subprocess and return parsed results."""
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, bot_dir, str(max_depth), ROOT],
        input=positions_json, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"Error running bot in {bot_dir}:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return None
    return json.loads(proc.stdout.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-depth benchmark: best/ vs current/")
    parser.add_argument("-n", "--num-positions", type=int, default=10,
                        help="Number of random positions (default: 10)")
    parser.add_argument("-d", "--depth", type=int, default=4,
                        help="Search depth (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for position generation (default: 42)")
    args = parser.parse_args()

    # Check builds exist
    for d, label in [(CURRENT_DIR, "current"), (BEST_DIR, "best")]:
        found = any(f.startswith("minimax_cpp") and (f.endswith(".so") or f.endswith(".pyd"))
                     for f in os.listdir(d))
        if not found:
            print(f"Error: no minimax_cpp .so in {d}/")
            print("  Run: make build")
            sys.exit(1)

    # Generate positions
    print(f"Generating {args.num_positions} random positions (seed={args.seed})...")
    positions = generate_positions(args.num_positions, seed=args.seed)
    if len(positions) < args.num_positions:
        print(f"  Warning: only generated {len(positions)} non-terminal positions")
    serialized = json.dumps([serialize_position(g) for g in positions])

    # Run both bots
    print(f"Running best/ at depth {args.depth}...")
    best_results = run_bot(BEST_DIR, serialized, args.depth)
    if best_results is None:
        sys.exit(1)

    print(f"Running current/ at depth {args.depth}...")
    current_results = run_bot(CURRENT_DIR, serialized, args.depth)
    if current_results is None:
        sys.exit(1)

    # Compare results
    n = len(positions)
    match_count = 0
    best_total_ms = 0.0
    curr_total_ms = 0.0
    best_total_nodes = 0
    curr_total_nodes = 0

    hdr = f"{'Pos':>3}  {'Stones':>6}  {'Player':>6}  " \
          f"{'best ms':>8}  {'curr ms':>8}  {'Speedup':>7}  " \
          f"{'best nodes':>10}  {'curr nodes':>10}  " \
          f"{'best depth':>10}  {'curr depth':>10}  " \
          f"{'Match':>5}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"  Fixed-depth benchmark: best/ vs current/ (depth={args.depth})")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for i in range(n):
        b = best_results[i]
        c = current_results[i]
        game = positions[i]
        stones = len(game.board)
        player = "A" if game.current_player == Player.A else "B"

        moves_match = b["moves"] == c["moves"]
        if moves_match:
            match_count += 1

        best_total_ms += b["time_ms"]
        curr_total_ms += c["time_ms"]
        best_total_nodes += b["nodes"]
        curr_total_nodes += c["nodes"]

        speedup = b["time_ms"] / c["time_ms"] if c["time_ms"] > 0 else float("inf")

        mark = "Y" if moves_match else "N"
        if not moves_match:
            # Check if scores are close (different move, same eval = acceptable)
            if abs(b["score"] - c["score"]) < 0.01:
                mark = "~"  # different move but same score

        print(f"{i:3d}  {stones:6d}  {player:>6s}  "
              f"{b['time_ms']:8.1f}  {c['time_ms']:8.1f}  {speedup:7.2f}x  "
              f"{b['nodes']:10d}  {c['nodes']:10d}  "
              f"{b['depth']:10d}  {c['depth']:10d}  "
              f"{mark:>5s}")

    print(sep)

    # Summary
    speedup_total = best_total_ms / curr_total_ms if curr_total_ms > 0 else float("inf")
    nps_best = best_total_nodes / (best_total_ms / 1000) if best_total_ms > 0 else 0
    nps_curr = curr_total_nodes / (curr_total_ms / 1000) if curr_total_ms > 0 else 0

    print(f"\n  Positions tested:    {n}")
    print(f"  Move agreement:      {match_count}/{n} "
          f"({100 * match_count / n:.0f}%)")
    print()
    print(f"  best/  total time:   {best_total_ms:8.1f} ms  "
          f"({best_total_ms / n:6.1f} ms/pos)  "
          f"{nps_best:,.0f} nodes/s")
    print(f"  current/ total time: {curr_total_ms:8.1f} ms  "
          f"({curr_total_ms / n:6.1f} ms/pos)  "
          f"{nps_curr:,.0f} nodes/s")
    print(f"  Speedup (best/curr): {speedup_total:.2f}x")

    # Flag disagreements with different scores
    disagree = []
    for i in range(n):
        b = best_results[i]
        c = current_results[i]
        if b["moves"] != c["moves"] and abs(b["score"] - c["score"]) >= 0.01:
            disagree.append((i, b, c))

    if disagree:
        print(f"\n  SCORE DISAGREEMENTS ({len(disagree)}):")
        for i, b, c in disagree:
            print(f"    pos {i}: best={b['moves']} score={b['score']:.2f}  "
                  f"curr={c['moves']} score={c['score']:.2f}")

    print(f"\n{'=' * len(hdr)}")


if __name__ == "__main__":
    main()

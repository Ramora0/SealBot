"""Fixed-depth value-bias probe: does interior-policy selection (mode 10)
back up systematically more optimistic root values than delta selection
(mode 2) on identical positions? Optimism = dropped opponent resources.

Runs both modes in-process via two env-configured subprocesses; compares
root scores position-by-position and against deep labels.
"""

import glob
import json
import os
import pickle
import random
import subprocess
import sys

SEAL = "/users/PAS2836/leedavis/personal/SealBot"

WORKER = r'''
import glob, json, os, pickle, random, sys
sys.path.insert(0, "%(seal)s")
sys.path.insert(0, "%(seal)s/current")
from game import HexGame, Player
import minimax_cpp

rng = random.Random(9)
files = sorted(glob.glob("%(seal)s/experiments/nnue/data/gen0_deep/*.pkl"))
rng.shuffle(files)
positions = []
for g in pickle.load(open(files[0], "rb")):
    for p in g["positions"]:
        if 12 <= p["move_count"] <= 44 and abs(p["score"]) < 25000:
            positions.append(p)
rng.shuffle(positions)
positions = positions[:120]

bot = minimax_cpp.MinimaxBot(3.0)
bot.max_depth = %(depth)d
out = []
for p in positions:
    game = HexGame(win_length=6)
    for q, r, pl in p["cells"]:
        game.board[(q, r)] = Player(pl)
    game.current_player = Player(p["mover"])
    game.moves_left_in_turn = p["moves_left"]
    game.move_count = p["move_count"]
    bot.get_move(game)
    out.append((bot.last_score, p["score"]))
print(json.dumps(out))
'''


def run(mode, depth):
    env = dict(os.environ, SEAL_POLICY_MODE=str(mode))
    r = subprocess.run(
        [sys.executable, "-c", WORKER % {"seal": SEAL, "depth": depth}],
        capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stderr[-500:])
        raise SystemExit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    import numpy as np
    for depth in (2, 3):
        a = np.array(run(2, depth), dtype=float)
        b = np.array(run(10, depth), dtype=float)
        assert (a[:, 1] == b[:, 1]).all()
        va, vb, deep = a[:, 0], b[:, 0], a[:, 1]
        d = vb - va
        quiet = (np.abs(va) < 25000) & (np.abs(vb) < 25000)
        print(f"depth {depth}: n={quiet.sum()} quiet")
        print(f"  mode10 - mode2 value: mean {d[quiet].mean():+8.0f}  "
              f"median {np.median(d[quiet]):+8.0f}")
        print(f"  |diff|>2000: {(np.abs(d[quiet])>2000).mean():.1%}  "
              f"(of those, mode10 higher: "
              f"{(d[quiet][np.abs(d[quiet])>2000]>0).mean():.1%})")
        dec = np.abs(deep) > 2000
        print(f"  sign agreement w/ deep label: mode2 "
              f"{np.mean(np.sign(va[dec & quiet])==np.sign(deep[dec & quiet])):.3f}  "
              f"mode10 {np.mean(np.sign(vb[dec & quiet])==np.sign(deep[dec & quiet])):.3f}")


if __name__ == "__main__":
    main()

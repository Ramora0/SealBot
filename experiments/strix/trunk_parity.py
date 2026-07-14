"""Engine-vs-python parity for the trunk port.

1. Value parity: bot.eval_position (SEAL_EVAL=trunk, blend 0) must equal
   Trunk2.value * 1000 on gen0 positions (mover POV = root POV here).
2. Incremental parity: acc_drift after a real search (make/undo + TimeUp
   rollback) must be ~float epsilon.

Run in the SealBot venv WITH env set:
    SEAL_EVAL=trunk SEAL_TRUNK_BLEND=0 ../../.venv/bin/python trunk_parity.py
"""

import glob
import os
import pickle
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEAL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SEAL)
sys.path.insert(0, os.path.join(SEAL, "current"))
sys.path.insert(0, SCRIPT_DIR)

from game import HexGame, Player
import minimax_cpp
from trunk_train2 import Trunk2
import trunk_train

CKPT = os.environ.get("TRUNK_CKPT",
                      os.path.join(SCRIPT_DIR, "output_trunk2", "trunk.pt"))


def main():
    assert os.environ.get("SEAL_EVAL") == "trunk", "run with SEAL_EVAL=trunk"
    ck = torch.load(CKPT, map_location="cpu")
    model = Trunk2(k=ck.get("K", 32), h=ck.get("H", 32), hp=ck.get("HP", 32))
    model.load_state_dict(ck["state"])
    model.eval()

    rng = random.Random(3)
    files = sorted(glob.glob(os.path.join(
        SEAL, "experiments", "nnue", "data", "gen0_strix", "*.pkl")))
    rng.shuffle(files)
    positions = []
    for p in pickle.load(open(files[0], "rb"))[0:40]:
        for pos in p["positions"]:
            positions.append(pos)
    rng.shuffle(positions)
    positions = positions[:200]

    bot = minimax_cpp.MinimaxBot(0.2)
    diffs = []
    for p in positions:
        game = HexGame(win_length=6)
        for q, r, pl in p["cells"]:
            game.board[(q, r)] = Player(pl)
        game.current_player = Player(p["mover"])
        game.moves_left_in_turn = p["moves_left"]
        game.move_count = p["move_count"]
        ev = bot.eval_position(game)

        trip, wi, wc, _ = trunk_train.extract(
            [tuple(c) for c in p["cells"]], p["mover"], [(0, 0)])
        with torch.no_grad():
            acc = model.accum(
                torch.from_numpy(trip.astype(np.int64)),
                torch.zeros(len(trip), dtype=torch.int64), 1,
                torch.from_numpy(wi.astype(np.int64)),
                torch.from_numpy(wc.astype(np.float32)),
                torch.tensor([0, len(wi)]))
            v = model.value_from_acc(
                acc,
                torch.tensor([p["move_count"] * 0.02], dtype=torch.float32),
                torch.tensor([p["moves_left"] * 0.5], dtype=torch.float32))
        diffs.append(abs(ev - float(v) * 1000.0))
    diffs = np.array(diffs)
    print(f"value parity over {len(diffs)}: max |diff| {diffs.max():.4f} "
          f"mean {diffs.mean():.5f} (engine units, scale ~8000)")

    worst = 0.0
    for p in positions[:5]:
        game = HexGame(win_length=6)
        for q, r, pl in p["cells"]:
            game.board[(q, r)] = Player(pl)
        game.current_player = Player(p["mover"])
        game.moves_left_in_turn = p["moves_left"]
        game.move_count = p["move_count"]
        worst = max(worst, bot.acc_drift(game, 0.3))
    print(f"acc2 drift after search (5 pos, 0.3s each): {worst:.2e}")


if __name__ == "__main__":
    main()

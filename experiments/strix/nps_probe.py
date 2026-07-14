"""Measure nodes/sec and reached depth per SEAL_POLICY_MODE on fixed
midgame positions. Run once per mode (env set by caller)."""

import glob
import os
import pickle
import random
import sys
import time

sys.path.insert(0, "/users/PAS2836/leedavis/personal/SealBot")
sys.path.insert(0, "/users/PAS2836/leedavis/personal/SealBot/current")

from game import HexGame, Player
import minimax_cpp

rng = random.Random(4)
files = sorted(glob.glob(
    "/users/PAS2836/leedavis/personal/SealBot/experiments/nnue/data/gen0_deep/*.pkl"))
rng.shuffle(files)
positions = []
for g in pickle.load(open(files[0], "rb")):
    for p in g["positions"]:
        if 14 <= p["move_count"] <= 40:
            positions.append(p)
rng.shuffle(positions)
positions = positions[:30]

bot = minimax_cpp.MinimaxBot(0.4)
nodes = 0
depths = []
t0 = time.time()
for p in positions:
    game = HexGame(win_length=6)
    for q, r, pl in p["cells"]:
        game.board[(q, r)] = Player(pl)
    game.current_player = Player(p["mover"])
    game.moves_left_in_turn = p["moves_left"]
    game.move_count = p["move_count"]
    bot.get_move(game)
    nodes += getattr(bot, "work_nodes", bot._nodes)
    depths.append(bot.last_depth)
dt = time.time() - t0
mode = os.environ.get("SEAL_POLICY_MODE", "default(2)")
print(f"mode {mode}: {nodes/dt/1000:.0f}k work-nps "
      f"(search+solver, all threads), "
      f"avg depth {sum(depths)/len(depths):.2f} over {len(positions)} pos")

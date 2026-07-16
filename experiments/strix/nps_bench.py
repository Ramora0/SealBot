"""NPS bench: fixed midgame position, fixed tl, report nodes + depth.
Usage: python nps_bench.py <bot_dir> <blob> [tl]"""

import os
import sys
import time

BOT_DIR = os.path.abspath(sys.argv[1])
os.environ["SEAL_EVAL"] = "mixnet"
os.environ["SEAL_MIXNET_BLOB"] = os.path.abspath(sys.argv[2])
os.environ["SEAL_TRUNK_BLEND"] = "0"
os.environ["SEAL_POLICY_MODE"] = "74"
os.environ["SEAL_VCF"] = "11"
os.environ["SEAL_VCF_K"] = "11"
os.environ["SEAL_VCF_BUDGET"] = "25000"
os.environ["SEAL_SMP_MODE"] = "2"
os.environ["SEAL_THREADS"] = "1"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BOT_DIR)
sys.path.insert(0, REPO)

from game import HexGame
from minimax_cpp import MinimaxBot

MOVES = [(0, 0), (1, 0), (-1, 1), (2, -1), (0, 1), (1, 1), (-2, 2),
         (3, -1), (0, 2), (2, 0), (-1, 3), (4, -2), (1, 2), (3, 0)]

tl = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
g = HexGame(win_length=6)
for q, r in MOVES:
    g.make_move(q, r)

b = MinimaxBot(time_limit=tl)
b.get_move(g)  # warm-up (tables, TT)
tot_nodes = 0
t0 = time.perf_counter()
for _ in range(3):
    b.get_move(g)
    tot_nodes += b._nodes
dt = time.perf_counter() - t0
print(f"{os.path.basename(BOT_DIR)}: {tot_nodes} nodes in {dt:.2f}s "
      f"= {tot_nodes/dt:.0f} nps, depth {b.last_depth}")

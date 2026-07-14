"""Replay one record's adversarial playout with full logging to hunt down
any false +1. Usage: debug_rec.py <rec_idx> [trials] [def_time]"""
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "current"))

from game import HexGame, Player          # noqa: E402
from minimax_cpp import MinimaxBot        # noqa: E402

RECS = "/users/PAS2836/leedavis/personal/SealBot/experiments/strix/strong_play_recs.pkl"
IDX = int(sys.argv[1])
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DEF_TIME = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

with open(RECS, "rb") as f:
    recs = pickle.load(f)
cells, mover, moves_left, move_count = tuple(recs[IDX][:4])
print(f"rec {IDX}: mover={mover} moves_left={moves_left} "
      f"stones={len(cells)}")

bot = MinimaxBot(0.05)
defender_bot = MinimaxBot(DEF_TIME)


def make_game():
    g = HexGame(win_length=6)
    for q, r, pl in cells:
        g.board[(q, r)] = Player(pl)
    g.current_player = Player(mover)
    g.moves_left_in_turn = moves_left
    g.move_count = move_count
    return g


for trial in range(TRIALS):
    g = make_game()
    attacker = Player(mover)
    log = []
    ok = True
    guard = 0
    while not g.game_over and guard < 200:
        guard += 1
        if g.current_player == attacker:
            r, mv = bot.forced_win(g, 8)
            log.append(("A-solve", r, mv, bot.vcf_nodes))
            if r != 1 or not mv:
                ok = False
                break
            for q, rr in mv:
                if g.game_over:
                    break
                assert g.make_move(q, rr), f"invalid {(q, rr)}"
        else:
            moves = defender_bot.get_move(g)
            log.append(("D-move", moves))
            for q, rr in moves:
                if g.game_over:
                    break
                if not g.make_move(q, rr):
                    log.append(("D-invalid", (q, rr)))
    won = g.winner == attacker
    print(f"trial {trial}: winner={g.winner} ok={ok} steps={len(log)}")
    if not won:
        print("  initial cells:", cells)
        for step in log:
            print("  ", step)
        break
else:
    print("all trials: attacker won every playout")

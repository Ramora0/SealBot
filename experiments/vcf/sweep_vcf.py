"""Sweep forced_win over real positions from strong_play_recs.pkl.

Reports solve counts at max_turns 3/5/8 and per-call timing, then verifies
every +1 by an adversarial playout: the attacker follows forced_win's turn
each time it moves, the defender plays MinimaxBot with generous time; the
attacker must actually win, and forced_win must stay +1 the whole way.

Run:  .venv/bin/python experiments/vcf/sweep_vcf.py [n_positions] [def_time]
"""
import os
import pickle
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "current"))

from game import HexGame, Player          # noqa: E402
from minimax_cpp import MinimaxBot        # noqa: E402

RECS = "/users/PAS2836/leedavis/personal/SealBot/experiments/strix/strong_play_recs.pkl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
DEF_TIME = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


def make_game(cells, mover, moves_left, move_count):
    g = HexGame(win_length=6)
    for q, r, pl in cells:
        g.board[(q, r)] = Player(pl)
    g.current_player = Player(mover)
    g.moves_left_in_turn = moves_left
    g.move_count = move_count
    return g


with open(RECS, "rb") as f:
    recs = pickle.load(f)

# Deterministic stride sample of N positions across the file.
idxs = sorted({round(i * (len(recs) - 1) / (N - 1)) for i in range(N)})
# Some records carry extra annotations (e.g. ('strix_move', ...)); the first
# four fields are always (cells, mover, moves_left, move_count).
sample = [tuple(recs[i][:4]) for i in idxs]
print(f"{len(recs)} recs, sweeping {len(sample)} positions "
      f"(stride sample), defender time {DEF_TIME}s")

bot = MinimaxBot(0.05)

results = {}          # k -> list of (idx, result, ms, nodes)
for k in (3, 5, 8):
    rows = []
    for i, (cells, mover, moves_left, move_count) in zip(idxs, sample):
        g = make_game(cells, mover, moves_left, move_count)
        t0 = time.perf_counter()
        r, mv = bot.forced_win(g, k)
        ms = (time.perf_counter() - t0) * 1e3
        rows.append((i, r, ms, bot.vcf_nodes, mv))
    results[k] = rows
    times = [x[2] for x in rows]
    wins = sum(1 for x in rows if x[1] == 1)
    nowin = sum(1 for x in rows if x[1] == -1)
    unk = sum(1 for x in rows if x[1] == 0)
    times_sorted = sorted(times)
    print(f"max_turns={k}: +1={wins}  -1={nowin}  0={unk} | "
          f"time ms mean={statistics.mean(times):.3f} "
          f"median={statistics.median(times):.3f} "
          f"p99={times_sorted[int(0.99 * (len(times) - 1))]:.3f} "
          f"max={max(times):.3f} | "
          f"max nodes={max(x[3] for x in rows)}")

# ── Verification of every +1 at max_turns=8 by adversarial playout ──
print("\n== Playout verification of +1 positions (max_turns=8) ==")
defender_bot = MinimaxBot(DEF_TIME)
wins8 = [(i, mv) for i, r, ms, nd, mv in results[8] if r == 1]
rec_by_idx = {i: tuple(recs[i][:4]) for i in idxs}

verified = failed = 0
fail_detail = []
opp_win_after = 0
for i, first_mv in wins8:
    cells, mover, moves_left, move_count = rec_by_idx[i]
    g = make_game(cells, mover, moves_left, move_count)
    attacker = Player(mover)

    # Quick check: after the winning first turn, the OPPONENT must not have
    # a forced win of their own.
    g2 = make_game(cells, mover, moves_left, move_count)
    for q, r in first_mv:
        if not g2.game_over:
            assert g2.make_move(q, r)
    if not g2.game_over:
        ro, _ = bot.forced_win(g2, 8)
        if ro == 1:
            opp_win_after += 1

    # Full adversarial playout with DECREASING horizon: if win-in-8 is a
    # real proof, then after our proof turn and ANY defender reply the
    # position must be +1 at k-1 (non-covering defender replies leave an
    # instant win, covering replies are inside the proof tree). The
    # attacker replays forced_win's entire returned turn each time
    # (a builder second stone need not itself be a threat move, so
    # re-solving mid-turn is not part of the contract).
    bot.vcf_node_budget = 2000000   # remove budget noise for verification
    ok = True
    att_turns = 0
    k = 8
    guard = 0
    while not g.game_over and guard < 200:
        guard += 1
        if g.current_player == attacker:
            att_turns += 1
            r, mv = bot.forced_win(g, k)
            if r != 1 or not mv:
                ok = False
                fail_detail.append((i, f"+1 lost at horizon k={k} "
                                       f"(r={r}) on attacker turn {att_turns}"))
                break
            k -= 1
            bad = False
            for q, rr in mv:
                if g.game_over:
                    break
                if not g.make_move(q, rr):
                    ok = False
                    bad = True
                    fail_detail.append((i, f"invalid winning move {(q, rr)}"))
                    break
            if bad:
                break
        else:
            moves = defender_bot.get_move(g)
            for q, rr in moves:
                if g.game_over:
                    break
                if not g.make_move(q, rr):
                    # extremely defensive: play any empty adjacent cell
                    placed = False
                    for (bq, br) in list(g.board):
                        for dq in (-1, 0, 1):
                            for dr in (-1, 0, 1):
                                if g.is_valid_move(bq + dq, br + dr):
                                    g.make_move(bq + dq, br + dr)
                                    placed = True
                                    break
                            if placed:
                                break
                        if placed:
                            break
    bot.vcf_node_budget = 5000
    if ok and g.winner == attacker and att_turns <= 8:
        verified += 1
    elif ok:
        failed += 1
        fail_detail.append((i, f"playout ended winner={g.winner} "
                               f"att_turns={att_turns}"))
    else:
        failed += 1

print(f"+1 positions: {len(wins8)}; playout-verified wins: {verified}; "
      f"failures: {failed}; opponent +1 after our winning turn: "
      f"{opp_win_after}")
for i, msg in fail_detail[:10]:
    print(f"  rec {i}: {msg}")
sys.exit(1 if failed or opp_win_after else 0)

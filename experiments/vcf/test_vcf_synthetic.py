"""Synthetic unit tests for the threat-space forced-win solver (vcf.h).

Connect6 threat facts used to construct the positions:
- A clean 4-in-6-window (0 defender stones) is a win-in-1: the attacker
  fills both empties with their 2 stones.
- An OPEN four (.XXXX.) forces BOTH defender stones (the three clean
  windows require the pair of end cells to cover them all).
- A forced win deeper than 1 turn requires every attacker turn to consume
  both defender stones (min hitting set == 2) and terminate with a turn
  after which the min hitting set is >= 3.

Run:  .venv/bin/python experiments/vcf/test_vcf_synthetic.py
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "current"))

from game import HexGame, Player          # noqa: E402
from minimax_cpp import MinimaxBot        # noqa: E402


def make_game(cells, mover=1, moves_left=2):
    g = HexGame(win_length=6)
    for q, r, pl in cells:
        g.board[(q, r)] = Player(pl)
    g.current_player = Player(mover)
    g.moves_left_in_turn = moves_left
    g.move_count = len(cells)
    return g


def apply_turn(game, moves):
    for q, r in moves:
        if game.game_over:
            break
        assert game.is_valid_move(q, r), f"invalid move {(q, r)}"
        game.make_move(q, r)


BOT = MinimaxBot(0.05)
PASS = 0
FAIL = []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def fw(game, k):
    return BOT.forced_win(game, k)


print("== Win in 1 ==")
# One clean (open) four: .XXXX. -> fill any clean window's empties.
g = make_game([(q, 0, 1) for q in range(4)] + [(10, 10, 2), (11, 10, 2)])
r, mv = fw(g, 1)
check("open four is +1 at max_turns=1", r == 1, f"r={r}")
if r == 1:
    apply_turn(g, mv)
    check("returned turn completes 6", g.winner == Player.A,
          f"winner={g.winner} moves={mv}")

# Two separate open fours (also win in 1).
g = make_game([(q, 0, 1) for q in range(4)] + [(q, 8, 1) for q in range(4)]
              + [(20, 20, 2), (21, 20, 2)])
r, mv = fw(g, 1)
check("two open fours is +1 at max_turns=1", r == 1, f"r={r}")
if r == 1:
    apply_turn(g, mv)
    check("returned turn completes 6 (double four)", g.winner == Player.A)

# Five in a row, one stone missing, mid-turn (stones_left == 1).
g = make_game([(q, 0, 1) for q in range(5)] + [(10, 10, 2), (11, 10, 2),
                                               (12, 10, 2)],
              mover=1, moves_left=1)
r, mv = fw(g, 1)
check("five-in-row, 1 stone left, +1", r == 1, f"r={r}")
if r == 1:
    apply_turn(g, mv)
    check("single stone completes 6", g.winner == Player.A, f"moves={mv}")

print("== Win in 2 ==")
# Crossing open threes: (3,0) completes a d0-line (0,0),(1,0),(2,0) and a
# d1-line (3,-3),(3,-2),(3,-1) -> one turn makes two open fours (hitting
# set 4) -> proven win in 2 turns.
cross = ([(q, 0, 1) for q in range(3)] + [(3, r, 1) for r in (-3, -2, -1)]
         + [(15, 15, 2), (16, 15, 2)])
g = make_game(cross)
r1, _ = fw(g, 1)
check("crossing threes not +1 at max_turns=1", r1 != 1, f"r={r1}")
g = make_game(cross)
r2, mv = fw(g, 2)
check("crossing threes +1 at max_turns=2", r2 == 1, f"r={r2}")
if r2 == 1:
    # after the winning turn the DEFENDER must have no forced win
    g2 = make_game(cross)
    apply_turn(g2, mv)
    if not g2.game_over:
        ro, _ = fw(g2, 8)
        check("defender has no forced win after winning turn", ro != 1,
              f"r={ro}")

# Same position with colors swapped (attacker = B).
crossB = [(q, r, 3 - p) for q, r, p in cross]
g = make_game(crossB, mover=2)
r2b, _ = fw(g, 2)
check("crossing threes (B attacker) +1 at max_turns=2", r2b == 1, f"r={r2b}")

# Mid-turn (1 stone) at the intersection also wins.
g = make_game(cross, mover=1, moves_left=1)
r, _ = fw(g, 2)
check("crossing threes, 1 stone left, +1 at max_turns=2", r == 1, f"r={r}")

print("== Win in 3 (forcing chain) ==")
# Turn 1: (3,0) makes an open four from (0..2,0)  [forces both def stones]
#         + builder (2,20) turning the pair (0,20),(1,20) into an open three
#           AND the d1-pair (2,18),(2,19) into an open three (intersection).
# Turn 2: (3,20) + (2,21) complete BOTH threes to open fours -> hitting 4.
# Turn 3: completes six. No 2-turn bypass exists: only the r=0 line can
# four with a single stone; every other four needs two stones.
chain = ([(q, 0, 1) for q in range(3)]            # open three
         + [(0, 20, 1), (1, 20, 1)]               # d0 pair
         + [(2, 18, 1), (2, 19, 1)]               # d1 pair (crosses (2,20))
         + [(30, 30, 2), (31, 30, 2)])
g = make_game(chain)
r2, _ = fw(g, 2)
check("chain not +1 at max_turns=2", r2 != 1, f"r={r2}")
g = make_game(chain)
t0 = time.perf_counter()
r3, mv3 = fw(g, 3)
dt = (time.perf_counter() - t0) * 1e3
check("chain +1 at max_turns=3", r3 == 1, f"r={r3}")
print(f"        (chain solve: {dt:.3f} ms, {BOT.vcf_nodes} nodes, "
      f"first turn {mv3})")
g = make_game(chain)
r8, _ = fw(g, 8)
check("chain +1 at max_turns=8 too", r8 == 1, f"r={r8}")

print("== Known non-wins ==")
# A single open three: any completion is a lone open four/five, defender
# always covers, nothing remains.
g = make_game([(q, 0, 1) for q in range(3)] + [(15, 15, 2), (16, 15, 2)])
r, _ = fw(g, 8)
check("single open three is not +1", r != 1, f"r={r}")

# A single (closed) four: O X X X X . . -> only one clean window, min
# hitting set 1 (free defender stone) -> conservatively not a proven win?
# NO: a clean 4-window is a win-in-1 (attacker moves first!).
g = make_game([(-1, 0, 2)] + [(q, 0, 1) for q in range(4)])
r, mv = fw(g, 1)
check("closed four still +1 (attacker moves first)", r == 1, f"r={r}")
if r == 1:
    apply_turn(g, mv)
    check("closed four turn completes 6", g.winner == Player.A)

# Defensible double: two CLOSED threes far apart. Completing both gives
# two closed fours = min hitting set 2 -> defender covers -> dead.
g = make_game([(-1, 0, 2)] + [(q, 0, 1) for q in range(3)]
              + [(-1, 10, 2)] + [(q, 10, 1) for q in range(3)])
r, _ = fw(g, 8)
check("two closed threes (defensible double) not +1", r != 1, f"r={r}")

# Defender counter-threat: attacker has the crossing-threes win, but the
# defender holds an open four -> v1 must return 0 (unknown), never +1.
g = make_game(cross + [(q, 40, 2) for q in range(4)])
r, _ = fw(g, 8)
check("defender open four forces result 0", r == 0, f"r={r}")

# A collinear pair spawns forcing lines (each turn can build an open four)
# that never terminate -> 0 (unknown after depth cutoff), never +1.
g = make_game([(0, 0, 1), (1, 0, 1), (10, 10, 2), (11, 10, 2)])
r, _ = fw(g, 5)
check("lone pair: unknown (0), never +1", r == 0, f"r={r}")

# No threat-building structure at all -> -1 (proven: no forcing lines).
g = make_game([(0, 0, 1), (10, 10, 2)])
r, _ = fw(g, 5)
check("isolated stone gives -1 (no forcing lines)", r == -1, f"r={r}")

print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)

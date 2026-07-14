"""Post-mortem for recorded strix bench games (<out>.games.pkl).

For every SealBot loss, replay the game and find the EARLIEST strix-to-move
position where the VCF solver (high budget) proves a strix forced win.
The gap between that turn and the end of the game classifies the loss:

  gap <= 1 turn  : positional squeeze — we were ground down, no tactical
                   save existed until the very end (needs better value net)
  gap 2-3 turns  : late tactical entry — our in-game defense filter budget
                   or k was too small to see the trap coming
  gap >= 4 turns : deep tactical blindness — we walked far into proven-lost
                   territory (raise SEAL_VCF_K / budgets, defense filter)

Also reports: losses where we were already lost by move 20 (opening/early
strategy problem), and win/loss length distributions.

Run in the SealBot venv:
    ../../.venv/bin/python analyze_losses.py bench_vcf14_150.json.games.pkl
"""

import pickle
import sys

SEAL = "/users/PAS2836/leedavis/personal/SealBot"
sys.path.insert(0, SEAL)
sys.path.insert(0, SEAL + "/current")

from game import HexGame, Player
import minimax_cpp


def replay_positions(seq):
    """Yield (game_state_snapshot, mover, move_index) before each move."""
    game = HexGame(win_length=6)
    out = []
    for i, (q, r, pl) in enumerate(seq):
        cells = [(qq, rr, p.value) for (qq, rr), p in game.board.items()]
        out.append((cells, game.current_player.value,
                    game.moves_left_in_turn, game.move_count, i))
        if game.game_over:
            break
        game.make_move(q, r)
    return out


def main():
    path = sys.argv[1]
    max_k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    with open(path, "rb") as fh:
        games = pickle.load(fh)

    bot = minimax_cpp.MinimaxBot(0.1)
    bot.vcf_node_budget = 60000

    losses = [g for g in games if not g["sealbot_won"]]
    wins = [g for g in games if g["sealbot_won"]]
    print(f"{len(games)} games: {len(wins)} sealbot wins, "
          f"{len(losses)} losses")
    print(f"win lengths: {sorted(len(g['seq']) for g in wins)[:20]}")

    gaps, early_lost, no_proof = [], 0, 0
    for g in losses:
        seq = g["seq"]
        strix_is_p1 = g["hexo_is_a"]
        strix_val = 1 if strix_is_p1 else 2
        positions = replay_positions(seq)
        first_lost_move = None
        # walk FORWARD over strix-to-move positions; stop at first proven win
        for cells, mover, ml, mc, idx in positions:
            if mover != strix_val or mc < 8:
                continue
            game = HexGame(win_length=6)
            for q, r, p in cells:
                game.board[(q, r)] = Player(p)
            game.current_player = Player(mover)
            game.moves_left_in_turn = ml
            game.move_count = mc
            res, _ = bot.forced_win(game, max_k)
            if res == 1:
                first_lost_move = mc
                break
        total = len(seq)
        if first_lost_move is None:
            no_proof += 1
        else:
            # turns (2 stones) between provably-lost entry and game end
            gap = (total - first_lost_move) / 4.0  # /4: 2 stones x 2 sides
            gaps.append(gap)
            if first_lost_move <= 20:
                early_lost += 1

    print(f"\nlosses with a provable strix win found: {len(gaps)} "
          f"(no proof found: {no_proof})")
    if gaps:
        import statistics
        b1 = sum(1 for x in gaps if x <= 1)
        b2 = sum(1 for x in gaps if 1 < x <= 3)
        b3 = sum(1 for x in gaps if x > 3)
        print(f"gap distribution (turns lost before end): "
              f"<=1: {b1}  2-3: {b2}  >=4: {b3}")
        print(f"mean gap {statistics.fmean(gaps):.1f}, "
              f"max {max(gaps):.1f}")
        print(f"provably lost by move 20 (opening problem): {early_lost}")
    print("\nInterpretation: >=4 bucket large -> raise SEAL_VCF_K/budgets;"
          "\n<=1 bucket large -> value net / positional play is the gap;"
          "\nno-proof large -> strix wins without forcing lines (positional).")


if __name__ == "__main__":
    main()

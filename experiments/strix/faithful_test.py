"""Bridge faithfulness: from_state-built positions must produce the same
strix value as the same position reached through apply_move. Replays real
hexo games (random legal rollouts) and compares at every mover-turn point."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells, value_batch


def main():
    import hexo_rs
    model, mc, gc = load_strix()
    rng = random.Random(7)

    worst = 0.0
    n = 0
    for trial in range(40):
        gs = hexo_rs.GameState(gc)
        # random legal rollout of random length
        for _ in range(rng.randint(2, 40)):
            if gs.is_terminal():
                break
            moves = gs.legal_moves()
            q, r = moves[rng.randrange(len(moves))]
            gs.apply_move(q, r)
        if gs.is_terminal():
            continue
        # rebuild via from_state using the SealBot-side representation
        stones = gs.placed_stones()
        cells = [(s[0][0], s[0][1], 1 if s[1] == "P1" else 2)
                 if isinstance(s[0], tuple) else None for s in stones]
        if cells and cells[0] is None:
            # placed_stones may return flat tuples (q, r, player)
            cells = [(s[0], s[1], 1 if s[2] == "P1" else 2) for s in stones]
        mover = 1 if gs.current_player() == "P1" else 2
        ml = gs.moves_remaining_this_turn()
        rebuilt = state_from_cells(cells, mover, ml, gc)
        v1, v2 = value_batch(model, mc, [gs, rebuilt])
        worst = max(worst, abs(v1 - v2))
        n += 1
    print(f"{n} positions, worst |v_incremental - v_from_state| = {worst:.2e}")
    assert worst < 1e-5, "BRIDGE UNFAITHFUL"
    print("BRIDGE FAITHFUL")


if __name__ == "__main__":
    main()

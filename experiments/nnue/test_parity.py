"""Parity tests: numpy feature extraction / forward pass vs the C++ engine.

1. feature_counts: engine's window + conjunction counts == numpy extraction
2. eval_position: engine _leaf_eval == numpy net_forward (sidecar weights)
3. acc_drift: accumulator after a real search == fresh recompute
"""

import os
import random
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "current"))

from game import HexGame, Player
import minimax_cpp
from features import extract_features, net_forward

_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def random_position(rng, n_stones):
    game = HexGame(win_length=6)
    for _ in range(n_stones):
        if game.game_over:
            return None
        if not game.board:
            cands = [(0, 0)]
        else:
            cands = list({(q + dq, r + dr) for q, r in game.board
                          for dq, dr in _D2 if (q + dq, r + dr) not in game.board})
        game.make_move(*rng.choice(cands))
    return None if game.game_over else game


def main():
    rng = random.Random(99)
    bot = minimax_cpp.MinimaxBot(0.05)
    net_npz = np.load(os.path.join(ROOT_DIR, "current", "net_data.h.npz"))
    net = {k: net_npz[k] for k in net_npz.files}
    lin_blend = float(net.get("lin_blend", 0.0))

    import re
    text = open(os.path.join(ROOT_DIR, "current", "pattern_data.h")).read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    pv_lin = np.array([float(x) for x in nums])

    n_feat = n_eval = 0
    worst_eval = 0.0
    for trial in range(60):
        game = random_position(rng, rng.randint(1, 40))
        if game is None:
            continue
        mover = 1 if game.current_player == Player.A else 2
        cells = [(q, r, 1 if p == Player.A else 2)
                 for (q, r), p in game.board.items()]

        fc = bot.feature_counts(game)
        w_idx, w_cnt, c_idx, c_cnt = extract_features(cells, mover)
        eng_w = dict(fc["w"])
        eng_c = dict(fc["c"])
        np_w = {int(i): int(n) for i, n in zip(w_idx, w_cnt)}
        np_c = {int(i): int(n) for i, n in zip(c_idx, c_cnt)}
        if eng_w != np_w:
            only_e = {k: v for k, v in eng_w.items() if np_w.get(k) != v}
            only_n = {k: v for k, v in np_w.items() if eng_w.get(k) != v}
            print(f"WINDOW MISMATCH trial {trial}: engine-only {list(only_e.items())[:5]} "
                  f"numpy-only {list(only_n.items())[:5]}")
            sys.exit(1)
        if eng_c != np_c:
            only_e = {k: v for k, v in eng_c.items() if np_c.get(k) != v}
            only_n = {k: v for k, v in np_c.items() if eng_c.get(k) != v}
            print(f"CONJ MISMATCH trial {trial}: engine-only {list(only_e.items())[:5]} "
                  f"numpy-only {list(only_n.items())[:5]}")
            sys.exit(1)
        n_feat += 1

        ev_engine = bot.eval_position(game)
        tempo = game.moves_left_in_turn * 0.5   # root is the mover here
        ev_numpy = net_forward(w_idx, w_cnt, c_idx, c_cnt,
                               game.move_count, tempo, net)
        ev_numpy += lin_blend * float((pv_lin[w_idx] * w_cnt).sum())
        d = abs(ev_engine - ev_numpy) / max(1.0, abs(ev_numpy))
        worst_eval = max(worst_eval, d)
        n_eval += 1

    print(f"feature parity: {n_feat} positions OK")
    print(f"eval parity:    {n_eval} positions, worst rel diff {worst_eval:.2e}")
    # float32 engine vs float64 numpy under icpc fast-fp: small noise is fine
    # (scales with weight magnitude; trained nets sit ~1e-3 relative)
    assert worst_eval < 2e-2, "eval parity FAILED"

    worst_drift = 0.0
    for trial in range(8):
        game = random_position(rng, rng.randint(6, 30))
        if game is None:
            continue
        worst_drift = max(worst_drift, bot.acc_drift(game, 0.1))
    print(f"acc drift after search: worst {worst_drift:.2e}")
    # absolute drift scales with embedding magnitude; trained accs are O(10)
    assert worst_drift < 5e-2, "acc drift FAILED"
    print("ALL PARITY TESTS PASSED")


if __name__ == "__main__":
    main()

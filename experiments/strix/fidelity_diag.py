"""Fidelity diagnostic: does the tiny distill net track strix on STRONG-play
positions as well as it does on gen0 (weak self-play + random openings)?

Plays distill-vs-strix games (sims configurable), records every non-terminal
position, then compares tiny-net eval vs strix forward value on those
positions, and the same on a gen0 sample. If fidelity collapses on strong
play, the feature set / architecture is the ceiling; if it holds, the gap
is labels + distribution.
"""

import glob
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells, value_batch

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")
BOT_DIR = SEAL / "distill_frozen"

sys.path.insert(0, str(SEAL))
sys.path.insert(0, str(BOT_DIR))
sys.path.insert(0, str(SEAL / "experiments/nnue"))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def tiny_eval_batch(recs):
    """Tiny-net eval (distill_frozen weights, incl. lin blend) via numpy."""
    import re
    from features import extract_features
    net = dict(np.load(BOT_DIR / "net_data.h.npz"))
    text = open(BOT_DIR / "pattern_data.h").read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    pv_lin = np.array([float(x) for x in nums])
    out = []
    for cells, mover, ml, mc in recs:
        w_idx, w_cnt, c_idx, c_cnt = extract_features(cells, mover)
        acc = net["ew"][w_idx].T @ w_cnt + net["ec"][c_idx].T @ c_cnt
        x = np.concatenate([np.clip(acc, 0, float(net["clip"])),
                            [mc * 0.02, ml * 0.5]])
        h = np.maximum(net["w1"] @ x + net["b1"], 0.0)
        ev = float(net["w2"] @ h + net["b2"]) * float(net["out_scale"])
        ev += float(net["lin_blend"]) * float((pv_lin[w_idx] * w_cnt).sum())
        out.append(ev)
    return np.array(out)


def play_and_record(model, mc_cfg, gc, n_games, sims, seal_tl):
    import hexo_rs
    import minimax_cpp
    from game import HexGame, Player as SBPlayer
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.sealbot_eval import BatchInferenceServer
    import torch

    graph_fn = lambda g: game_to_axis_graph(
        g, prune_empty_edges=mc_cfg.prune_empty_edges,
        threat_features=mc_cfg.threat_features,
        relative_stones=mc_cfg.relative_stone_encoding)
    device = torch.device("cuda:0")
    server = BatchInferenceServer(model, device, graph_fn)
    mcts = hexo_rs.MCTSConfig(n_simulations=sims, m_actions=16,
                              c_visit=50, c_scale=1.0)

    def hexdist(a, b):
        dq, dr = a[0] - b[0], a[1] - b[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    recs = []
    try:
        for gi in range(n_games):
            seal = minimax_cpp.MinimaxBot(time_limit=seal_tl)
            hexo_is_a = gi % 2 == 0
            game = HexGame(win_length=6)
            gs = hexo_rs.GameState(gc)
            moves = 0
            while not game.game_over and moves < 200:
                if game.move_count == 0:
                    game.make_move(0, 0)
                    moves += 1
                    gs = hexo_rs.GameState(gc)
                    continue
                # record position (mover POV fields)
                cells = [(q, r, 1 if p == SBPlayer.A else 2)
                         for (q, r), p in game.board.items()]
                mover = 1 if game.current_player == SBPlayer.A else 2
                recs.append((cells, mover, game.moves_left_in_turn,
                             game.move_count))
                is_hexo = ((hexo_is_a and game.current_player == SBPlayer.A)
                           or (not hexo_is_a and game.current_player == SBPlayer.B))
                if is_hexo:
                    a, _ = hexo_rs.gumbel_mcts(gs, server.eval_fn, mcts)
                    recs[-1] = recs[-1] + (("strix_move", a),)
                    gs.apply_move(a[0], a[1])
                    game.make_move(a[0], a[1])
                    moves += 1
                else:
                    pair = seal.get_move(game)
                    pair = pair if seal.pair_moves else [pair]
                    for m in pair:
                        if game.game_over:
                            break
                        m = tuple(m)
                        legal = set(map(tuple, gs.legal_moves()))
                        if m not in legal:
                            m = min(legal, key=lambda c: (hexdist(c, m), c))
                        game.make_move(m[0], m[1])
                        moves += 1
                        gs.apply_move(m[0], m[1])
            print(f"game {gi+1}/{n_games}: {moves} moves", flush=True)
    finally:
        server.stop()
    return recs


def compare(tag, recs, model, mc_cfg, gc):
    states, keep = [], []
    for rec in recs:
        cells, mover, ml, mc = rec[:4]
        s = state_from_cells(cells, mover, ml, gc)
        if s is not None:
            states.append(s)
            keep.append((cells, mover, ml, mc))
    v_strix = np.array(value_batch(model, mc_cfg, states))
    v_tiny = tiny_eval_batch(keep) / 8000.0   # back to strix scale
    dec = np.abs(v_strix) > 0.25
    print(f"\n{tag}: n={len(keep)}")
    print(f"  pearson {np.corrcoef(v_tiny, v_strix)[0,1]:.3f}   "
          f"spearman {spearman(v_tiny, v_strix):.3f}")
    print(f"  sign agreement (|v_strix|>0.25): "
          f"{np.mean(np.sign(v_tiny[dec]) == np.sign(v_strix[dec])):.3f}")


def main():
    model, mc_cfg, gc = load_strix()

    print("playing recorded distill-vs-strix games (sims=16)...")
    t0 = time.time()
    strong = play_and_record(model, mc_cfg, gc, n_games=24, sims=16,
                             seal_tl=0.44)
    print(f"{len(strong)} strong-play positions in {time.time()-t0:.0f}s")
    with open(Path(__file__).parent / "strong_play_recs.pkl", "wb") as fh:
        pickle.dump(strong, fh)

    rng = random.Random(5)
    files = sorted(glob.glob(str(SEAL / "experiments/nnue/data/gen0_deep/*.pkl")))
    rng.shuffle(files)
    weak = []
    for f in files[:6]:
        for g in pickle.load(open(f, "rb")):
            for p in g["positions"]:
                weak.append((p["cells"], p["mover"], p["moves_left"],
                             p["move_count"]))
    rng.shuffle(weak)
    weak = weak[:len(strong)]

    compare("gen0 (weak play, training distribution)", weak, model, mc_cfg, gc)
    compare("STRONG play (distill vs strix)", strong, model, mc_cfg, gc)


if __name__ == "__main__":
    main()

"""Benchmark a SealBot build against hexo-strix at equal time controls.

strix = Gumbel MCTS (GPU, budget in simulations); SealBot = C++ minimax
(budget in seconds per 2-stone turn). Equal time: sealbot_tl = 2 x strix
measured sec/stone.

Rules bridge: SealBot's native game has no placement radius, HeXO requires
moves within radius 6 of an existing stone. A HeXO-illegal SealBot move is
substituted with the nearest HeXO-legal cell in BOTH mirrored boards
(SealBot re-reads the position every get_move, so this is transparent to
it); occurrences are counted and reported.

Which SealBot build plays is fixed by pre-importing minimax_cpp from
--bot-dir. Run inside the hexo-strix venv (python 3.13):
    .venv/bin/python bench_vs_strix.py --bot-dir best --games 150 --tl 0.44
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

SEALBOT_ROOT = Path("/users/PAS2836/leedavis/personal/SealBot")
STRIX_ROOT = Path("/users/PAS2836/leedavis/personal/hexo-strix")
CKPT = STRIX_ROOT / "checkpoint_00237000.pt"


def hex_dist(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def play_game(game_idx, sealbot_cls, tl, gc, gc_dict, eval_fn, mcts_config,
              strix_times, subs, record=None):
    import hexo_rs
    from game import HexGame, Player as SBPlayer

    sealbot = sealbot_cls(time_limit=tl)
    hexo_is_a = (game_idx % 2 == 0)
    game = HexGame(win_length=gc_dict["win_length"])
    gs = hexo_rs.GameState(gc)
    moves = 0
    seq = []  # (q, r, player_int) in play order

    while not game.game_over and moves < gc_dict["max_moves"]:
        is_hexo_turn = ((hexo_is_a and game.current_player == SBPlayer.A)
                        or (not hexo_is_a and game.current_player == SBPlayer.B))
        if game.move_count == 0:
            game.make_move(0, 0)
            seq.append((0, 0, 1))
            moves += 1
            gs = hexo_rs.GameState(gc)
            continue
        if is_hexo_turn:
            t0 = time.perf_counter()
            action, _ = hexo_rs.gumbel_mcts(gs, eval_fn, mcts_config)
            strix_times.append(time.perf_counter() - t0)
            gs.apply_move(action[0], action[1])
            seq.append((action[0], action[1],
                        1 if game.current_player == SBPlayer.A else 2))
            game.make_move(action[0], action[1])
            moves += 1
        else:
            result = sealbot.get_move(game)
            pair = result if sealbot.pair_moves else [result]
            for m in pair:
                if game.game_over:
                    break
                m = tuple(m)
                legal = set(map(tuple, gs.legal_moves()))
                if m not in legal:
                    sub = min(legal, key=lambda c: (hex_dist(c, m), c))
                    subs.append((game_idx, m, sub))
                    m = sub
                seq.append((m[0], m[1],
                            1 if game.current_player == SBPlayer.A else 2))
                game.make_move(m[0], m[1])
                moves += 1
                gs.apply_move(m[0], m[1])

    is_win = (game.winner != SBPlayer.NONE) and (
        (hexo_is_a and game.winner == SBPlayer.A)
        or (not hexo_is_a and game.winner == SBPlayer.B))
    is_draw = game.winner == SBPlayer.NONE
    if record is not None:
        record.append({"game_idx": game_idx, "hexo_is_a": hexo_is_a,
                       "seq": seq, "winner": int(game.winner.value)
                       if game.winner != SBPlayer.NONE else 0,
                       "sealbot_won": bool(is_win)})
    return {"moves": moves, "is_win": is_win,
            "is_loss": (not is_win) and (not is_draw), "is_draw": is_draw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-dir", required=True, help="best or current")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--tl", type=float, default=0.44,
                    help="SealBot seconds per 2-stone turn")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--m-actions", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--record", action="store_true",
                    help="dump full game sequences to <out>.games.pkl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    bot_dir = (SEALBOT_ROOT / args.bot_dir).resolve()
    sys.path.insert(0, str(SEALBOT_ROOT))
    sys.path.insert(0, str(bot_dir))
    import minimax_cpp
    print(f"minimax_cpp from: {minimax_cpp.__file__}", flush=True)

    import torch
    import hexo_rs
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.sealbot_eval import BatchInferenceServer, _win_rate_stats

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ck["model_config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HeXONet(mc).to(device)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"model loaded on {device}", flush=True)

    gc = hexo_rs.GameConfig(**ck["game_config"])
    gc_dict = {"win_length": gc.win_length,
               "placement_radius": gc.placement_radius,
               "max_moves": gc.max_moves}
    graph_fn = lambda g: game_to_axis_graph(
        g, prune_empty_edges=mc.prune_empty_edges,
        threat_features=mc.threat_features,
        relative_stones=mc.relative_stone_encoding)
    server = BatchInferenceServer(model, device, graph_fn)
    mcts_config = hexo_rs.MCTSConfig(n_simulations=args.sims,
                                     m_actions=args.m_actions,
                                     c_visit=50, c_scale=1.0)

    strix_times, subs, results = [], [], []
    record = [] if args.record else None
    t_start = time.time()
    try:
        for i in range(args.games):
            results.append(play_game(i, minimax_cpp.MinimaxBot, args.tl, gc,
                                     gc_dict, server.eval_fn, mcts_config,
                                     strix_times, subs, record=record))
            if (i + 1) % 10 == 0 or i + 1 == args.games:
                w = sum(r["is_win"] for r in results)
                l = sum(r["is_loss"] for r in results)
                print(f"[{i+1}/{args.games}] strix {w}W-{l}L "
                      f"({time.time()-t_start:.0f}s, {len(subs)} subs)",
                      flush=True)
    finally:
        server.stop()

    if record is not None and args.out:
        import pickle
        with open(args.out + ".games.pkl", "wb") as fh:
            pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"recorded {len(record)} games -> {args.out}.games.pkl")

    wins = sum(r["is_win"] for r in results)
    losses = sum(r["is_loss"] for r in results)
    draws = sum(r["is_draw"] for r in results)
    win_rate, ci_lo, ci_hi, elo = _win_rate_stats(wins, losses, draws)
    mean = statistics.fmean(strix_times) if strix_times else 0.0
    summary = {
        "bot_dir": args.bot_dir, "games": args.games,
        "sealbot_tl_per_turn": args.tl, "strix_sims": args.sims,
        "strix_sec_per_stone_mean": round(mean, 4),
        "strix_sec_per_turn_est": round(2 * mean, 4),
        "strix_wins": wins, "strix_losses": losses, "draws": draws,
        "strix_win_rate": round(win_rate, 4),
        "strix_elo_vs_sealbot": round(elo, 1),
        "ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "mean_game_length": round(statistics.fmean(r["moves"] for r in results), 1),
        "illegal_substitutions": len(subs),
        "sub_examples": [f"{g}: {m}->{s}" for g, m, s in subs[:10]],
    }
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

"""Reproduce the out-of-range crash: current/ SealBot vs strix, dump state."""
import sys
import time
from pathlib import Path

SEALBOT_ROOT = Path("/users/PAS2836/leedavis/personal/SealBot")
STRIX_ROOT = Path("/users/PAS2836/leedavis/personal/hexo-strix")

sys.path.insert(0, str(SEALBOT_ROOT))
sys.path.insert(0, str(SEALBOT_ROOT / "current"))
import minimax_cpp
print("minimax_cpp:", minimax_cpp.__file__, flush=True)

import torch
import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.graph import game_to_axis_graph
from hexo_a0.sealbot_eval import BatchInferenceServer
from game import HexGame, Player as SBPlayer

ck = torch.load(STRIX_ROOT / "checkpoint_00237000.pt", map_location="cpu",
                weights_only=False)
mc = ModelConfig(**ck["model_config"])
device = torch.device("cuda:0")
model = HeXONet(mc).to(device)
model.load_state_dict({k.removeprefix("_orig_mod."): v
                       for k, v in ck["model_state_dict"].items()}, strict=False)
model.eval()

gc = hexo_rs.GameConfig(**ck["game_config"])
graph_fn = lambda g: game_to_axis_graph(
    g, prune_empty_edges=mc.prune_empty_edges,
    threat_features=mc.threat_features, relative_stones=mc.relative_stone_encoding)
server = BatchInferenceServer(model, device, graph_fn)
mcts_config = hexo_rs.MCTSConfig(n_simulations=64, m_actions=16,
                                 c_visit=50, c_scale=1.0)

for game_idx in range(6):
    sealbot = minimax_cpp.MinimaxBot(time_limit=0.44)
    hexo_is_a = (game_idx % 2 == 0)
    game = HexGame(win_length=6)
    gs = hexo_rs.GameState(gc)
    moves = 0
    try:
        while not game.game_over and moves < 300:
            is_hexo_turn = ((hexo_is_a and game.current_player == SBPlayer.A)
                            or (not hexo_is_a and game.current_player == SBPlayer.B))
            if is_hexo_turn:
                if game.move_count == 0:
                    gs = hexo_rs.GameState(gc)
                    game.make_move(0, 0)
                    moves += 1
                    continue
                action, _ = hexo_rs.gumbel_mcts(gs, server.eval_fn, mcts_config)
                gs.apply_move(action[0], action[1])
                game.make_move(action[0], action[1])
                moves += 1
            else:
                if game.move_count == 0:
                    game.make_move(0, 0)
                    moves += 1
                    gs = hexo_rs.GameState(gc)
                    continue
                result = sealbot.get_move(game)
                pair = result if sealbot.pair_moves else [result]
                for m in pair:
                    if game.game_over:
                        break
                    try:
                        game.make_move(m[0], m[1])
                        moves += 1
                        gs.apply_move(m[0], m[1])
                    except ValueError as e:
                        print(f"\nGAME {game_idx} CRASH at move {moves}: {e}")
                        print(f"sealbot pair: {pair}, failing move: {m}")
                        sb = sorted(game.board.items())
                        hx = sorted([tuple(s) for s in gs.placed_stones()])
                        print(f"sealbot board ({len(sb)}): {sb}")
                        print(f"hexo board    ({len(hx)}): {hx}")
                        sb_set = {c for c, _ in sb}
                        hx_set = {(s[0][0], s[0][1]) if isinstance(s[0], tuple) else (s[0], s[1]) for s in hx}
                        print("only sealbot:", sb_set - hx_set)
                        print("dist to nearest sealbot stone:",
                              min(max(abs(m[0]-q), abs(m[1]-r), abs(m[0]+m[1]-q-r))
                                  for q, r in sb_set if (q, r) != (m[0], m[1])))
                        raise SystemExit(1)
        print(f"game {game_idx}: over={game.game_over} winner={game.winner} moves={moves}")
    except SystemExit:
        raise
server.stop()
print("no crash in 6 games")

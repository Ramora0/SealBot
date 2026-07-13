---
name: strix-distillation
description: hexo-strix (GNN AlphaZero in ../hexo-strix) distilled into SealBot tiny net — pure value distill beat the NNUE champion 91-8 (+413); strix beats all seals ~-700..-990 Elo at equal time
metadata:
  type: project
---

2026-07-13, branch `nnue-eval`, follows [[nnue-eval-branch-status]]. hexo-strix
(`~/personal/hexo-strix`, checkpoint_00237000.pt, GINE h128 L4 JK-cat axis-graph
GNN, value head = tanh scalar, side-to-move POV) is the strongest hexo engine;
SealBot is its standard eval opponent (same game: win_length 6, but HeXO adds
placement radius 6 — SealBot's intentional "colony candidate" far moves are
HeXO-illegal, bench substitutes nearest legal).

Infrastructure in `experiments/strix/` (run in hexo venv, python 3.13; 3.13 ABI
.so built alongside 3.10 in each bot dir): bench_vs_strix.py (equal time = strix
0.22 s/stone at 64 sims → sealbot 0.44 s/turn), strix_bridge.py (from_state +
axis_states_to_batch, 3.5k pos/s on V100, POV/translation/faithfulness verified),
strix_relabel.py, eval_offline.py. `evaluate.py --opp-dir` gates vs any build
dir; champion snapshot in `champion_frozen/`.

Results:
- Equal-time vs strix (150 g): original 0-150 (≈−990), champion 2-148 (≈−710).
- Strix raw fwd value vs OUR deep labels looks WEAK (sign 0.63, coin-flip on
  random openings, misses search-proved mates) but beats champion at predicting
  game outcomes (0.673 vs 0.601) — long-horizon judgment vs tactics split.
- Pure distill (target strix_v*8, Huber, lam 1.0, gen0 positions): tiny net
  reproduces teacher at 0.939 corr; gates +235 vs original, **+413 vs champion
  (91-8-1)**. Non-transitive: distill>champ 91%, champ>orig 87%, distill>orig 79.5%.

Key lesson: offline agreement with a search teacher's labels is NOT the decisive
metric — the engine supplies tactics; long-horizon value quality wins games.

Open: handicap ladder (strix sims 16/4) for resolution; mixdeep (strix+deep mix)
net; blend sweep; stage-2 distill of strix SEARCH values (batched_gumbel_mcts,
solve_forcing VCF labels); gen1_strix data unused.

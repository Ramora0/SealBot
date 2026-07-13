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

POLICY-ROOT breakthrough (later same day): strix policy distilled into
PW[729]+PC[8548] ordering tables (listwise KL over D2 candidates). Used at the
ROOT ONLY = +402 Elo over the distill leader (91%, n=300); used at interior
nodes = −512. ROOT-CAUSED (not a bug; mirror identities verified exact):
minimax pruning asymmetry — dropping OUR alternatives is suboptimal, dropping
the OPPONENT'S refutations is unsound optimism that compounds per min-node.
Side-split bisect: our-side interior +67(ns), opponent-side −512, combos
interfere (+83), min-node union widening loses to breadth (+16). Delta stays
in-tree as the refutation guard (forcing replies always have huge window
deltas = ~100% refutation recall) until the value net can score policy-shaped
replies. Engine knobs: SEAL_POLICY_MODE (2=root-only default, bits:
1=our-interior 2=root 4=threat 8=opp-interior 16=min-union),
SEAL_CAND_CAP/ROOT_CAP/DELTA_KEEP. Equal-time vs strix: policy-root 11/150
(7.3%) vs distill 3/150, champion 2/150, original 0/150.
Diagnosis that led there: 23% of strix's moves fell below the interior cap
under old ordering; value fidelity collapses off-distribution (0.86→0.68).
Basis battery: cellnl (per-cell elementwise NL over summed raw 3^11 line
embeddings, no codebook) is the best value basis (0.943) — next value net.
Mixdeep (avg of strix+deep targets) was a hard failure (−552 vs distill):
never average incompatible teachers; override selectively (VCF ±8) instead.

Open: cellnl engine adoption; DAgger value data from strong play; VCF label
overrides; root-cap sweep at mode 2; search-value (root Q) distillation;
gen1_strix data unused.

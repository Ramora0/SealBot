# Strix Distillation — Experiment Log

Goal (2026-07-13): distill hexo-strix (`../hexo-strix`, GNN AlphaZero,
checkpoint_00237000, GINE h128 L4 JK-cat, axis graphs, threat features,
relative stones) into the tiny SealBot NNUE net; benchmark SealBots vs
strix at equal time controls.

## Infrastructure

- `bench_vs_strix.py` — SealBot (any build dir) vs strix (Gumbel MCTS on
  the V100, budget in sims). Equal time: strix measured **0.22 s/stone**
  at sims=64 → sealbot_tl = **0.44 s per 2-stone turn**.
  Rules bridge: HeXO requires placement within radius 6 of a stone;
  SealBot's board is unbounded and its movegen has an intentional
  far-away "colony candidate" (movegen.h) — HeXO-illegal moves get
  substituted with the nearest legal cell in both mirrored boards
  (counted; ~8/150 games for the NNUE champion, 0 for original).
- `strix_bridge.py` — SealBot cells → `GameState.from_state` (A→P1;
  datagen always opens at (0,0)=A) → `axis_states_to_batch` (Rust rayon)
  → value-only `_forward_batch_core`. **3.5k pos/s** on the V100.
  Verified: POV (+1.0 winning mover / −0.62 defender), translation
  invariance (exact), from_state ≡ apply_move (5e-7).
- `strix_relabel.py` — shards → strix-labeled shards
  (`score = strix_v * 8000`, deep label kept as `score_deep`).
  gen0: 693,185 pos, gen1: 460,634 pos, 0 skipped.
- Python 3.13 ABI builds of minimax_cpp coexist with 3.10 in
  best/ + current/ (icpc, same flags) for the hexo venv.
- `evaluate.py --opp-dir <dir>` gates current/ vs any build dir
  (champion snapshotted in `champion_frozen/`).
- `train.py --loss mixdeep` — target = lam*score + (1-lam)*score_deep
  (both /1000); dataset cache now packs `score_deep`.

## Equal-time benchmark vs strix (150 games each, sims=64 vs 0.44s/turn)

| SealBot | W-L vs strix | Elo gap |
|---------|--------------|---------|
| original (best/) | **0-150** | ≈ −990 (est. floor) |
| NNUE champion    | **2-148** | ≈ −710 |

Strix is far stronger than both at equal time. Champion improvement vs
strix is visible but 0-vs-2 wins is not significant (p≈0.25) — handicap
ladder (reduced strix sims) queued for resolution.

## Strix single-forward value quality on OUR data (bridge-verified)

On gen0 (original self-play + random D2 openings, deep tl-0.12 labels):

| metric | strix fwd | champion hybrid |
|--------|-----------|-----------------|
| sign vs deep label (decided) | 0.630 | 0.828–0.857 |
| … on MATE-proved labels | 0.641 | 0.798 |
| … early positions (mc<12) | **0.535** | 0.937 |
| spearman vs deep (quiet) | 0.417 | 0.49–0.52 |
| sign vs GAME OUTCOME | **0.673** | 0.601 |

Strix's raw value looks weak against search-based labels — worst on
random-opening (off-distribution) positions, and it can't see tactics a
depth-3+ search proves. BUT it beats the champion at predicting the
actual game outcome — better long-horizon judgment. Lesson: offline
agreement with a *search* teacher is not the metric that decides play
strength when the consumer engine supplies its own tactics.

## Distillation run 1: pure strix targets (gen0 positions)

train: `--loss score --lam 1.0` on gen0_strix (target = strix_v*8,
Huber δ=4, 60 ep). Held-out metrics: **corr 0.939 / spearman 0.930 vs
strix target** (tiny net reproduces the GNN's judgment on this
distribution); inherits strix's profile vs deep labels (0.633/0.424);
sign vs outcome 0.647 (champ 0.601).

Gates (tl 0.1, 100 games, blend 0.15 + original CMA table):

| matchup | result |
|---------|--------|
| distill vs original (best/) | 77-18-5, **+235** (p=3.6e-09) |
| distill vs CHAMPION | **91-8-1, +413** (p=1.1e-16) |

Non-transitive triangle: distill > champion (91%) > original (87%) >
… distill "only" 79.5% vs original. Distill is 2-0 head-to-head — new
leader. Strix-vs-distill equal-time bench running.

## Open / queued

- Handicap ladder vs strix (sims 16/4) for original/champion/distill.
- mixdeep net (0.5 strix + 0.5 deep): does combining strix long-horizon
  with search-label tactics beat pure distill?
- Blend sweep for distill net (0 / 0.10 / 0.20).
- Stage 2: distill strix SEARCH values (batched_gumbel_mcts root Q,
  and/or `solve_forcing` VCF labels for exact tactics).
- gen1 strix-labeled data unexplored.

## Diagnosis battery (why 0.93 fidelity didn't convert to wins)

Value-fidelity offline metrics were the wrong comparison. Measured causes:
1. Fidelity collapses off-distribution: pearson 0.86 (gen0) -> 0.68 on
   strong-play (distill-vs-strix) positions; sign 0.97 -> 0.83. (DAgger data
   needed for the value side.)
2. ORDERING was the dominant leak: strix's chosen move is in our D2
   candidate set 94.6% of the time, but the old linear delta ranks it
   mean 9.2 / median 4 — 23% fall below the interior cap 15 (pruned
   unseen), 15% below the root cap 20.
3. Strix at FOUR sims still beats old/champion ~9:1 — knowledge >> search
   at these scales.

Basis battery (fit strix value, identical training, gen0 held-out corr):
codes (concat->linear) 0.910 < raw (3^11 lines, no interaction) 0.924 <
joint (current 8548 classes) 0.933 < joint+raw 0.940 < CELLNL 0.943
(per-cell nonlinearity over summed raw line embeddings — cross-line
interaction without enumeration; engine-viable at ~2x update cost with a
cached per-cell pre-activation). cellnl is the value-net architecture to
adopt next.

## Policy distillation -> move ordering (the big win)

PW[729]+PC[8548] tables trained with listwise KL on strix policy logits
over D2 candidates (401k gen0 positions; mover-relative, color-mirror
tables for opponent nodes). Ranking of strix's move on strong play:
mean 4.4 / median 2, top-15 95.0% (old: 9.2 / 4 / 76.7%).

Engine: _policy_score + _select_candidates with runtime knobs
(SEAL_POLICY_MODE, SEAL_CAND_CAP, SEAL_ROOT_CAP, SEAL_DELTA_KEEP).

Gates vs distill_frozen (same net, only ordering differs):
| config | result |
|--------|--------|
| policy everywhere (mode 3), caps 15/20 | 38.5%, -81 |
| mode 3, caps 25/30 | 48.5%, -10 |
| mode 3 + delta safety net 6 | 28.5%, -160 |
| policy INTERIOR only (mode 1) | 5%, -512 |
| **policy ROOT only (mode 2)** | **91.0% (n=300), +402** |

Lesson: the oracle policy is gold for CHOOSING the move at the root and
poison for interior tree ordering — interior alpha-beta needs refutation
ordering consistent with the ENGINE'S OWN eval (the linear delta is that
eval's derivative). Mode 2 is the new default.

Cumulative chain (head-to-head): original <- champion (+338) <- distill
(+413) <- policy-root (+402). Policy-root vs original: 80% (+241).
Strix benches for policy-root: running.

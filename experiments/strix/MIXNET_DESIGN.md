# Mixnet repro — Rapfi's network adapted to hex Connect6 (design)

Goal: reproduce, as faithfully as the game allows, the Rapfi Mixnet
(arXiv 2503.13178) — architecture AND standard-CE training recipe — and
see where it gets us vs strix. Reference summary: `../../gomoku-research.md`.
Branch: `mixnet-repro`. Trainer: `mixnet_train.py`.

## What Rapfi does (the parts we reproduce)

1. **Line codebook trunk.** Board → per-point line patterns (length 11,
   one per direction) → mapping CNN (5 Dir-Conv + 1x1 + skips) → baked
   into a pattern-indexed codebook. Per-point feature = sum of
   per-direction lookups → ReLU → depth-wise 3x3 conv over half the
   channels → feature map F'.
2. **Policy head = dynamic policy convolution.** Global mean pool of F'
   → 2-layer MLP → generates weights/bias of a point-wise conv applied
   to the first P channels of each cell → per-cell logit. π over the
   board plane.
3. **Value head.** Spatial pooling + star blocks (multiplicative
   nonlinearity, StarNet-style s(x) = (W_a x) ⊙ (W_b x)) + 3-layer MLP
   → categorical WDL (3 logits).
4. **Losses.** Plain CE on both heads. Targets mixed at the LOSS level:
   75% teacher soft outputs (distillation) + 25% true labels from the
   self-play data (outcome / recorded policy). Adam lr 1e-3, β=(0.9,
   0.999), eps 1e-8. Sizes (Small): mapping width M=64, feature C=32,
   policy P=16, value V=32.

## Game-forced adaptations (each is a deviation, keep the list honest)

| Rapfi (gomoku 15x15) | Ours (hex, win-6, Connect6, infinite) | Why |
|---|---|---|
| 4 directions, len-11 lines (2·5+1) | 3 axes, len-11 lines (2·6−1) — same length by coincidence | hex has 3 win axes |
| 4-state cell (edge/out-of-board) | 3-state (empty/own/opp), codebook 3^11 = 177,147 | no board edges |
| Dense H×W feature map | Sparse: universe U = active cells (any nonzero lp) ∪ hex-dilation(active) ∪ candidates; all other cells provably contribute 0 | infinite board |
| — | **Zero anchor**: mapping(all-empty pattern) ≡ 0 (subtract at bake) so inactive cells are exactly 0 and sparse pooling = infinite-board pooling; depthwise conv has no bias so 0-neighborhoods stay 0 | required for sparse/incremental equivalence |
| depth-wise 3x3 conv (9 cells) | depth-wise hex-neighborhood conv (center + 6 neighbors) on half the channels | grid geometry |
| π ∈ R^{H×W} | π over the D2 candidate set (softmax over candidates) | infinite board; matches engine movegen |
| no tempo input | g0 = move_count·0.02, g1 = moves_left·0.5 appended to pooled vectors (both heads) | Connect6 sub-turn state |
| value grouping (3x3 chunk pooling) | v1: global sum pool only (matches engine acc2 machinery). Value grouping parked as v2 | no fixed grid; keep engine port tractable |
| 3-class WDL | 3-class kept; soft target p_draw = 0 (strix value is a tanh scalar), true-label draws use the draw class | draws are rare but representable |

## Data (our analog of their teacher pipeline)

- **Distill stream (75%)**: `policy_targets/*.npz` (141 shards, ~460k
  positions: strix forward-pass policy logits over D2 candidates)
  joined per-shard with `../nnue/data/gen0_strix/*.pkl` strix values —
  exactly the "teacher soft outputs" analog (strix net = their ResNet
  teacher; we skip their intermediate ResNet and distill the GNN
  directly).
- **True stream (25%)**: bench game records (`bench_*.games.pkl`) →
  (position, strix's played move, final outcome). This is the honest
  analog of Katagomo's self-play tuples. Weaker than theirs: our visit
  distributions aren't stored, so true-policy = played-move one-hot;
  true-value = outcome one-hot.
- Known gap vs Rapfi: they had 30.8M positions; we start with ~0.5M.
  Scale-up (more strix self-play / DAgger harvests) is a separate lever.

## Losses (faithful)

```
value:  L_v = 0.75·CE(v̂, soft WDL from strix v) + 0.25·CE(v̂, outcome)   [when outcome known]
policy: L_p = 0.75·CE(π̂, softmax(strix logits)) + 0.25·CE(π̂, played move)[when move known & in cand set]
total:  L = L_v + L_p          (equal weight, per Rapfi)
```
Samples lacking a true label fall back to the soft term alone.
Optimizer Adam(1e-3, 0.9, 0.999, 1e-8). Rapfi's batch=128 × 600k iters
assumes 30.8M positions; at our scale we keep batch 512 and epochs
sized to avoid memorizing 460k positions — noted deviation.

## Known risks (from our own measurement laws)

- **WDL-CE saturation** (nnue-eval lesson): sigmoid-space targets on
  mate-heavy shallow labels collapse to sign-only. Mitigation: strix
  values are calibrated long-horizon, not shallow-search spikes — the
  exact regime where our pure distill (Huber, v*8) worked. Watch
  offline spearman on the mid-range, not just corr.
- **Soft-target flatness** (artifact_anatomy lesson): CE to a soft
  distribution reproduces the target's sharpness; our strix policy is
  64-sim-Gumbel-soft. The 25% one-hot true stream is the sharpener —
  this is the component our failed KL-only v1.1 lacked.
- **Law #2**: offline metrics don't transfer. The trainer's metrics
  only validate the code path; adoption is decided by gate.py vs strix.

## Engine port (later task, scoping notes)

Codebook 3^11 × C=32 int8 ≈ 5.7 MB (quantize like Rapfi's [−16,16]).
Incremental: a stone changes ≤3×11 cells' F; hex conv spreads each
delta to 7 cells' F'; pooled acc updated by deltas (same pattern as
today's acc2). Policy per candidate = one P×P dyn matmul (generated
once per position) + P-dot. Value = star+MLP per eval, recomputed per
node as today. 16-bit dyn-conv matmul per Rapfi A.3.

# NNUE Eval — Experiment Log

Branch: `nnue-eval`. Goal: replace the linear window-sum eval with an
NNUE-style net (window embeddings + 3-direction conjunction features),
trained by bootstrapping from the original SealBot (`best/`, frozen as the
fixed opponent). `pv[]`/`_eval_score` retained ONLY for move ordering.

## Architecture (v1)

- acc[K=32] float32, incrementally maintained in `_make`/`_undo`
- Features:
  - 729 window patterns (existing `_wp` indices) -> `NET_EW[729][32]`
  - conjunction classes: per-cell 11-cell line patterns (`_lp`, 3 dirs),
    codebook -> 6-level threat codes per (player, dir), sorted-triple canon
    -> 8548 classes -> `NET_EC[8548][32]`
- Head: clip(acc, 0, 8) ++ [move_count*0.02] -> 32 relu -> 1;
  eval = logit * 600 (logit space = win-prob space via sigmoid)
- Codebook: 3^11 patterns, exact one-stone-refutability for L3/L4 split;
  reversal + color-swap invariants verified.

## Verification status

- Feature parity engine vs numpy: EXACT on 60 random positions
- Eval parity: 3e-4 relative (float32 vs float64, icpc fast-fp) — OK
- Acc drift after live search (make/undo + rollback): < 3e-4 — OK
- Node rate: ~292k nps vs original 724k (2.5x/node cost, ~0.3 ply at EBF 17)

## Training pipeline

- `datagen.py`: self-play, best/ engine both sides, tl 0.02–0.05s,
  2–10 random opening stones, labels = root search score (mover POV) + outcome
- `train.py`: EmbeddingBag NNUE, BCE on t = 0.6*sigmoid(score/4000) + 0.4*outcome,
  alternating color-mirror batches
- `emit_net.py --ckpt` -> `current/net_data.h` -> rebuild -> `evaluate.py`

## Results

| Run | Eval | Gate vs best/ (orig, 100g @ 0.1s) | Notes |
|-----|------|------------------------------------|-------|
| random-init | net only | 0W/91L/9D (stale-build bug artifact) | gate.sh now force-cleans |
| v1 (wdl, scale 4000, 25 ep) | net only | 10W/90L, Elo −382 | flat/saturated logits |
| v2 (wdl, scale 15000, tempo, 60 ep) | net only | 7W/93L, Elo −449 | same wall |
| v2 + lin_blend 0.3 | **hybrid** | **Elo +67 (CI −2..+136, p=0.057)** | first win; under relabel load |

Blend sweep (v2 net, 0.1s, under relabel load):
- blend 0.30, 200g: 55.2% (+37, p=0.14); pooled w/ first 100g run ≈ +47
- blend 0.15, 100g: 61.0% (**+78, p=0.028**) — best so far
- blend 0.60, 100g: 50.0% (linear drowns the net)

Key diagnostic (11.8k non-mate deep-labeled positions):
- sign agreement w/ deep search (|score|>2000): net 0.84, old linear 0.57
- Spearman rank corr: net 0.18, old linear 0.43
- => net knows WHO is winning; linear knows WHICH move is locally better.
  Hybrid combines both. Deep-label retrain (tl 0.12) queued to fix the
  net's resolution directly; score-space Huber loss added as option.

## Deep-label round (tl 0.12 teacher, 693k pos)

| Config | Gate (100g) |
|--------|-------------|
| deep_score (Huber lam .85), blend 0.0 | +14 |
| **deep_score, blend 0.15** | **+363; confirmed 200g: 87.5% / +338 (p=2.8e-26)** |
| deep_score, blend 0.20 / 0.25 / 0.30 | +21 / −576 / −inf |
| deep_score, blend 0.05 / 0.10 | +14 / 0 |
| deep_wdl, blends 0/.15/.3 | +14 / −14 / −363 |

Blend is a NARROW band: old table's wrong signs (43% on decided
positions) outvote the net above ~0.2; below ~0.15 no ordering help.

## Linear refit on deep labels (linear_refit.py)

Quiet-subset (|score|<25k) ridge w/ intercept (intercept=+3038 = tempo
bias, dropped at emit). New vs old table on identical data:
spearman 0.518 vs 0.425, sign agreement 0.851 vs 0.576. Weight sanity:
all 4-stone shapes strongly positive (+530..+1087). -> output/pattern_deep.h
Gate matrix queued post-gen1: refit table changes BOTH ordering and blend
scale (weights ~25x smaller than old mate-weights; sweep blends 0.5/1/2).

## Gen1 self-improvement + variants (all flat)

- gen1 net (30k hybrid self-play games, hybrid teacher): +7 @ blend .15
- mix01 net (gen0_deep + gen1_deep): −7 @ blend .15
- refit linear table in engine (better offline metrics!): +28 @ blend 0,
  ≤0 at blends .3/1/3
- Interpretation: the champion net is maximally in-distribution for
  playing the ORIGINAL (trained on original-vs-original play, deep-labeled
  by the original). Hybrid-flavored data dilutes that. Self-improvement
  needs better data mixing / teacher depth to go further.

## FINAL champion (left built in current/)

**deep_score net (output/deep_score/net.pt) + original CMA pattern table
+ NET_LIN_BLEND 0.15**

| Run | Result |
|-----|--------|
| 100 games | 89W/11L, +363 |
| 200 games | 175W/25L, +338 (p=2.8e-26) |
| 300 games (final) | 259W/41L, +320 (p=2.6e-36) |
| **Pooled (600)** | **523W/77L = 87.2% ≈ +332 Elo** |

Wins while searching shallower (avg 2.7 vs 3.1) with avg game 49 moves.

## Caveats / next steps

1. Sharp optimum: neighboring blends (±0.05) and all net variants gate ~0.
   Part of the edge may be opponent-specific. Worth a round-robin among
   variants + games vs humans to assess generality.
2. Self-improvement loop (gen1) flatlined — fix before iterating: deeper
   fixed-depth teacher, data mixing ratios, opening diversity.
3. Speed: NNUE engine ~2.5x/node slower (~0.4 ply). int16+AVX quantization,
   lazy accumulator updates, K=16 all untried — stacks with everything.
4. Search upgrades from todo.md (LMR/PVS/aspiration) also stack.

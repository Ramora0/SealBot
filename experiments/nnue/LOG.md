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

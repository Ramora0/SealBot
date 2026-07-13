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

| Gen | Data | Train | Gate vs best/ (orig) | Notes |
|-----|------|-------|----------------------|-------|
| 0   | 50k games / ~700k pos (from original bot) | — | — | in progress |

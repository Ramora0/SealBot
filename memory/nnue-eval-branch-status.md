---
name: nnue-eval-branch-status
description: Final status of the overnight NNUE eval build on branch nnue-eval — champion config beats original 87% (+320 Elo, n=600), full pipeline in experiments/nnue/
metadata:
  type: project
---

Overnight autonomous build (2026-07-12→13), branch `nnue-eval`, goal achieved: **NNUE hybrid eval beats original SealBot 87.2% pooled over 600 games (~+330 Elo, final 300-game run: 86.3%, p=2.6e-36), while searching shallower (2.7 vs 3.1)**.

Champion config (left built in `current/`): deep_score net (`experiments/nnue/output/deep_score/net.pt`) + ORIGINAL CMA pattern table (ordering + leaf blend) + `NET_LIN_BLEND 0.15`. Recipe that produced the net: gen0 self-play by original bot (50k games) → relabel with original @ tl 0.12 → Huber score-space regression (λ=0.85 score, ±8 outcome anchor), 60 epochs.

Engine: acc[K=32] + window embeddings EW[729] + conjunction embeddings EC[8548] (11-cell `_lp` line patterns, 3^11 codebook, 6-level refutability alphabet), tempo input, `eval = net*1000 + 0.15*_eval_score`. Parity + drift tested (`test_parity.py`). ~2.5x/node slower than original — int16/lazy-update/K=16 optimizations untried.

Hard-won lessons:
- WDL-sigmoid targets saturate on mate-heavy shallow labels → net learns sign only, flat mid-range → LOSES (−400). Score-space Huber on deep labels fixed it.
- Decisive diagnostic: sign-agreement vs rank-correlation with deep search splits eval quality into "who wins" (net: 0.84) vs "local ordering" (linear: 0.43); blend combines. Blend is a SHARP optimum: 0.10→0, 0.15→+338, 0.20→+21, 0.25→−576 (old table's 43% wrong signs on decided positions outvote the net above ~0.2).
- All improvement attempts on top gated flat: gen1 self-play net (+7), mixed data (−7), deep-refit linear table (+28 despite better offline spearman 0.52 vs 0.43). Champion may be partly opponent-specific (in-distribution for original-vs-original). Self-improvement loop needs work (teacher depth, data mixing).
- gate.sh MUST force-clean builds (setuptools misses header deps — cost one false 0W/91L gate). Fit linear models on quiet positions with an intercept (tempo bias +3000); mate targets destroy l2 fits.

All scripts in `experiments/nnue/`: datagen, relabel, train (wdl|score), emit_net (--lin-blend), gate.sh, deep_pipeline.sh, gen1_pipeline.sh, linear_refit, test_parity, LOG.md (full history).

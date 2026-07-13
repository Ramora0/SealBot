---
name: sealbot-improvement-roadmap
description: Measured weaknesses and prioritized improvement plan for SealBot (hex Connect6 engine), from July 2026 analysis
metadata:
  type: project
---

Analysis of SealBot (as of 2026-07-12, commit c94749c) found these measured facts:

- Search is plain alpha-beta + TT + killers + delta ordering. No PVS, no LMR, no aspiration, no null-move. EBF ≈ 17 at inner branching ~56 (g_inner_pairs: 15 cands, i+j≤14 → 56 pairs); ~430k nodes/s; depth 4 ≈ 180 ms, so only depth 3–4 reached at the 0.1 s game control.
- `_history` table in search.h is written on beta cutoffs but never read (dead since "delta-only scoring" commit f53f926).
- Eval weights (CMA-ES-tuned, 364 free params) are mostly noise: 602/729 patterns are dead (both colors in window) yet carry median |w|≈345; 1–2-own-stone windows average negative; completed-6 pattern is −56 (harmless, win short-circuits). Real signal is only 4-stone (~+2.3k) and 5-stone (~+49k) weights. Reversal symmetry not enforced in full CMA mode (~0.7% violation).
- Eval is linear in pattern counts → fitting it is convex; supervised (Texel-style logistic regression on self-play outcomes) would beat CMA-ES massively and enables longer windows (3^7/3^8).
- Linear window-sum cannot represent "two independent threats = win" — root cause of the known "can't block colonies" todo item. Fix direction: non-linear threat-count terms from existing _wc/hot-set data.

Priority order agreed in analysis: (1) LMR+PVS+aspiration in search, (2) supervised eval retraining pipeline, (3) non-linear threat terms, (4) longer/openness-aware patterns, (5) VCF/VCT-style threat-space search.

User direction (2026-07-12): mainly interested in improving the eval function itself, explicitly curious about cheap incremental NNUE-like approaches that see beyond single lines. Key architectural fact: the existing incremental `_eval_score` + `_wp` window machinery is already a 1-neuron NNUE accumulator — generalizing pv[pattern] scalars to K-dim embeddings + tiny MLP is the natural NNUE analog. Main wrinkle: `_move_delta` ordering needs a cheap scalar path.

Env note: repo needs `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && ./build.sh` on fresh checkout; machine has 40 cores. `make evaluate N=100 T=0.1` is the ground-truth test; benchmark.py for fixed-depth speed.

---
name: nnue-eval-branch-status
description: Status of the overnight NNUE eval build on branch nnue-eval (2026-07-12/13) — what works, gate results, key findings
metadata:
  type: project
---

Overnight autonomous build (user asked to skip stages and go straight to full NNUE, bootstrap from original bot, keep iterating until it beats `best/`).

Built and verified on branch `nnue-eval`:
- Full NNUE engine in `current/`: acc[K=32] + EW[729] window embeddings + EC[8548] conjunction embeddings (11-cell `_lp` line patterns, 3^11 codebook w/ exact refutability); parity engine↔numpy EXACT on features; acc drift <1e-3 after live search. `pv[]` kept for ordering only.
- Pipeline: `datagen.py` (self-play, 50k games/693k pos), `relabel.py` (deep teacher tl=0.12), `train.py` (torch EmbeddingBag, wdl BCE or score Huber, mirror aug, tempo input g1), `emit_net.py` → `net_data.h` (+ sidecar npz), `gate.sh` (MUST force-clean build: setuptools misses header deps — this bug wasted one gate cycle), `test_parity.py`, `deep_pipeline.sh`.

Gate history vs frozen original (100g @ 0.1s): net-only v1 −382 Elo, v2 −449; **hybrid eval `net*scale + blend*_eval_score` at blend 0.15: +78 Elo p=0.028** (blend 0.3: +37..+67; 0.6: 0). [[sealbot-improvement-roadmap]]

THE key insight: net sign-agreement w/ deep search 0.84 vs old eval 0.57, but Spearman rank corr 0.18 vs 0.43 — net knows who's winning, linear window-sum knows local move gradients; hybrid combines. Net-only saturation traced to wdl targets squashing decided positions.

Perf: NNUE engine ~270-290k nps vs original 720k (2.5x/node, ~0.3-0.4 ply); diff-apply + line-code cache added, deeper opts (int16, K=16) untried.

Env gotchas: run evaluate/benchmark on an UNLOADED machine (30-worker relabel skews); background Bash needs explicit big timeout; icpc compiler (fast-fp → 3e-4 float noise in parity, fine).

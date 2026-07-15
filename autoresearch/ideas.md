# Ideas backlog — variable tweaks only this run

Ordered roughly by expected value. One experiment = one idea. Mark
attempts with the results.tsv name; move dead ideas to the graveyard at
the bottom with a one-line cause of death.

## Knob inventory (verified against cand/ source, 2026-07-14)

Env (no rebuild): SEAL_VCF_FK/FB (defense-filter probe; auto = max(4,k/2)=5,
clamp(budget/6,800,8000)=6666), SEAL_VCF_K=11, SEAL_VCF_BUDGET=40000,
SEAL_ROOT_CAP=20 (runtime-effective), SEAL_CAND_CAP=15 (runtime only
DOWN — interior pairs compile-time via g_inner_pairs), SEAL_TT_BITS
(auto 22 at T>=8; range 16–26), SEAL_DELTA_KEEP=0, SEAL_TRUNK_BLEND=0,
SEAL_POLICY_MODE=74.

Code constants (edit cand/ + rebuild): search/veto time split (0.82 tl
search, 0.16 tl veto — search.h:143,406); defense-filter clock cutoff
40% (search.h:221); veto tiers k+3 / max(6,k-2), chosen-move budget x2,
probe cap 5 (search.h:416–426,447–450); CANDIDATE_CAP=15,
ROOT_CANDIDATE_CAP=20, PAIR_SUM_CAP=14 (wedge — measured load-bearing,
don't touch), NEIGHBOR_DIST=2, DELTA_WEIGHT=15, MAX_QDEPTH=16
(constants.h); trunk time-check mask 255 (bot.h:557).

## Round 1 — env knobs (from EVAL_SCHEME.md, composable → factorial corners)

- [ ] **A: `SEAL_VCF_FK=8 SEAL_VCF_FB=20000`** — defense-filter probe
  strength (was k/2, budget/6 when sequential; now parallel). #1
  autopsy-aligned bet: 33/40 losses were proof-budget constrained.
- [ ] **B: `SEAL_ROOT_CAP=26`** — root width, newly ~free under
  root-split SMP.
- [ ] **C: `SEAL_VCF_K=13 SEAL_VCF_BUDGET=60000`** — proof depth, same
  parallel logic.
- [ ] Corners AB / AC / BC / ABC of whatever singles adopt.

## More knob territory (check engine getenv sites for exact names/ranges)

- [ ] `SEAL_TT_BITS` — TT sizing vs the 5800X3D's 96 MB V-cache; the
  auto-size (2^22 at T>=8) was tuned on cluster nodes, not this chip.
- [ ] `SEAL_SMP_MODE=3` — ABDADA re-test on this box: lost at T=20 on
  cluster (31 vs 35/100) but no cluster tuning has transferred yet;
  measured root-split occupancy here = 2.56 cores / 32% of physical.
- [ ] Tiered-veto knobs (chosen-move probe at k+3, doubled budget) —
  sweep the +3 and the multiplier now that probes run parallel.
- [ ] `SEAL_TRUNK_BLEND` micro-sweep (0 vs 0.05 vs 0.1) — champion is 0,
  but the old sharp optimum was measured pre-SMP.
- [ ] Time-management constants in `cand/` (iteration cutoff fractions,
  rollback margins) — cheap numeric edits, historically untouched.
- [ ] Move-ordering constants in `cand/` (killer weights, delta-scoring
  margins).

## Out of scope this run (structural — keep for a future run)

- Recursive PV-splitting / speculative depth d+1 (SMP next steps past
  root-split's ~2.5x Amdahl cap).
- LMR / PVS / aspiration windows (re-measure EBF on trunk champion
  first).
- int16 quantized accumulator, lazy accumulator update, K=16 trunk
  re-emit (+ parity test).
- Teacher-depth bump for the relabel/self-improvement loop.
- Code deletions (dead `_history` table, legacy lazy-SMP mode 1,
  pattern-eval fallback paths) — worthwhile, but not variable tweaks.

## Graveyard

(nothing yet on this machine)

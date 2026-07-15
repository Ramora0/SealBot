# Ideas backlog — variable tweaks only this run

Ordered roughly by expected value. One experiment = one idea. Mark
attempts with the results.tsv name; move dead ideas to the graveyard at
the bottom with a one-line cause of death.

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

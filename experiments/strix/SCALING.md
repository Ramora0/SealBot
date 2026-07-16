# Mixnet scaling battery — 2026-07-15

Six steps-matched runs, shared cache, identical val split (seed 0),
non-mirror (clean architecture science; shipping nets add --mirror,
measured tax ~ -0.013 corr / -0.023 top1). Metrics on the held val
split: corr/sp = value correlation/Spearman vs strix forward values,
top1/mrank = policy vs strix argmax. Driver: mixnet_battery.py.

| leg     | config       | data | corr   | sp     | top1  | mrank | v-loss | p-loss |
|---------|--------------|------|--------|--------|-------|-------|--------|--------|
| d25     | M64/C32      | 25%  | 0.9562 | 0.9421 | 0.633 | 1.90  | 0.5067 | 1.7276 |
| d50     | M64/C32      | 50%  | 0.9569 | 0.9431 | 0.631 | 1.95  | 0.5078 | 1.7382 |
| anchor  | M64/C32      | 100% | 0.9565 | 0.9426 | 0.634 | 1.92  | 0.5087 | 1.7390 |
| c16     | M64/C16      | 100% | 0.9497 | 0.9346 | 0.617 | 2.02  | 0.5118 | 1.7806 |
| m128    | M128/C32     | 100% | 0.9582 | 0.9433 | 0.637 | 1.90  | 0.5074 | 1.7246 |
| m128c64 | M128/C64     | 100% | 0.9636 | 0.9507 | 0.647 | 1.87  | 0.5056 | 1.7128 |

## Findings

1. **Data axis is flat at C32.** 4x data volume moves corr by
   +/-0.0004 (noise). No overfitting even at 48 epochs on the 25%
   split. More strix-forward relabeling of same-distribution states is
   worthless at this capacity — this is the noise band for everything
   else.
2. **Size axis is live and monotone.** C16 costs -0.0068 corr /
   -0.017 top1; C64 gains +0.0071 corr / +0.013 top1 over the anchor
   (~17x / ~4x noise). Still rising at C64 (epoch 11->12 gained
   +0.0018) — not saturated.
3. **M (mapping width, train-time only) is free capacity:** +0.0017
   corr at C32. The baked table depends on C alone, so M128 costs
   nothing at inference. Always train with M128.
4. **Revision of the "labels mined out" verdict:** the regime at
   M64/C32 is *model-limited*, not data-limited. The value head moved
   for the first time across all levers tried this epoch. Caveat: the
   data curve was measured at C32 only; at C64 data volume may matter
   again (re-check d50-at-C64 if C64 gates in).
5. **Offline != Elo** (mixnet1m: +0.17 top1 -> +2 gate). C64 doubles
   the codebook (23 -> 45 MB), _amem (2.5 -> 5 MB), and conv work; the
   NPS tax is unknown (5800X3D's 96 MB L3 can hold the C64 table).
   The gate is the arbiter. C128 (90 MB table) blows past L3 —
   only worth probing if C64 gates strongly.

## Actions

- Shipping candidate training: M128/C64 --mirror --epochs 24 (curve
  still rising at 12), shared battery cache -> output_ship_m128c64.
- Engine variant cand_mixnet64: MX_C=64, MX_C2=32; fixed latent
  policy-scorer bug (loop bound conflated MX_C2 with MX_P — buffer
  overflow for any C != 32).
- Pipeline: CPU bake -> parity -> pause datagen loop -> gate vs ctrl
  (clean clock) -> resume datagen.
- Gen2 cluster generation unaffected: label source remains the
  strategic lever; C64 is the tactical one. Both stack.

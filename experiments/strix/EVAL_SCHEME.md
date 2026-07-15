# SealBot Incremental-Change Evaluation Scheme (design doc)

Goal: a general, mostly-automated harness for gating **any** incremental
SealBot change — env knobs, engine code changes, retrained nets — against
strix, cheap enough to run every night, honest enough that adopted changes
compound instead of miscalibrating. This is the spec for another agent to
build. It is NOT just the one 2^3 env-knob sweep discussed on 2026-07-14;
that sweep is merely the first round the harness should run.

## The process model (fishtest-style additive adoption)

- There is always exactly one **baseline** = champion + all adopted changes.
- A **round** = a set of candidate configs, each fully specified, each gated
  against the baseline under identical conditions.
- Winners fold into the baseline; the winner's own gate numbers become the
  next round's baseline reference (a fresh control is re-run whenever
  conditions change: node type, thread count, harness edit).
- Rounds are serial; **within a round everything is parallel.** When a
  round's candidates are cheap/composable (env knobs), test the combination
  corners directly (factorial) instead of greedy one-at-a-time — it removes
  the serial dependency and catches interactions. Code/net candidates are
  usually tested singly.

## Two-tier gate (per candidate)

| Tier | Opponent | Games | Purpose | Cost (measured) |
|---|---|---|---|---|
| 1 | strix sims=16 @ tl 0.44 | 100 (dev openings) | screen + rough ordering | ~10 s/game, **~17 min** |
| 2 | strix sims=64 @ tl 0.44 | 100 (dev openings) | confirm at real strength | ~16 s/game, **~28 min** |

Adoption rule: candidate must beat baseline by **>= 10 points / 100 games at
Tier 1** AND show **no regression at Tier 2**. Borderline (+4..+9) extends to
200 games or gets parked. Parked list: retry the most plausible reject after
each adoption (order-dependence hedge). Every 2–3 adoptions, run the
**held-out check** (openings 50–74) — if dev gains stop transferring, we are
memorizing the dev set; stop and reassess. Winner's curse: the top pick of a
multi-candidate round is selected on noise; its adoption confirm (Tier 2 /
extended games) is the unbiased estimate, not its screening number.

## Protocol pins (violating any of these invalidates comparisons)

- `SEAL_THREADS=20`, `SEAL_SMP_MODE=2` (root-split YBW default), tl=0.44,
  `--bot-dir current`, checkpoint_00237000.pt, m_actions=16.
- Champion env: `SEAL_EVAL=trunk SEAL_TRUNK_POLICY=1 SEAL_TRUNK_BLEND=0
  SEAL_POLICY_MODE=74 SEAL_VCF=15 SEAL_VCF_K=11 SEAL_VCF_BUDGET=40000`
  (weights: output_trunk5 via default `TRK_ERAW_PATH`; `SEAL_TRUNK_BLOB`
  overrides).
- Openings: `openings_human.pkl` (75 openings, paired colors; bench plays
  opening i//2 for game i). `--games 100` = openings 0–49 = dev set;
  openings 50–74 = held-out; empty board = scoreboard only (~107 Elo
  memorization inflation, law #5).
- Do NOT shrink tl to save time: VCF budgets are node-count-based, so a
  shorter clock changes what proof knobs mean, not just wall time.
- Never gate on seal-vs-seal h2h (law #1: +160 h2h → 1/149 vs strix) or on
  offline metrics (law #2). Play vs strix is the only gate. Old pre-NNUE
  seal builds are NOT a valid opponent (same engine family + saturation).
- strix sims=1 is NOT a valid screen (no lookahead → cannot punish
  unsoundness, which is exactly what most candidates change). s16 is the
  cheap screen; it may under-detect s64-attack fixes, hence Tier 2.
- Thread count is pinned into results (law #6). Nets ship in co-trained
  pairs (law #3). Every ported net gets a python-parity test before benching
  (law #4). NPS is not a gate metric (law #7) — play + depth-at-fixed-clock.

## Infrastructure (what to build)

SLURM batch fan-out on OSC Pitzer, account `PAS2836`. Each gate = one job:
`--nodes=1 --ntasks=1 --cpus-per-task=20 --gpus-per-node=1 --time=1:30:00`.
Dual-V100 40-core nodes pack exactly 2 gates. GPU contention is harmless
(strix strength is sims-fixed, latency-immune); CPU contention is NOT
(seal's tl is wall-clock) — cgroup core isolation makes batch jobs safe.
Verified 2026-07-14: `minimax_cpp` imports and CUDA works on other nodes via
non-interactive shell (no module loads needed);
python = `/users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python`.

Components:
1. **Round manifest** (json/yaml): list of candidates, each
   `{name, bot_dir, env: {K: V, ...}, sims: [16, 64], games}`. `bot_dir`
   support is what makes the scheme general — code-change candidates are
   built into their own bot dirs (e.g. `cand_<name>/`) so parallel jobs
   never race on a shared `.so` and every gate is reproducible. Env-only
   candidates reuse `current`. Always include a `ctrl` entry (baseline).
2. **gate.sbatch**: takes (name, sims, bot_dir, env string); exports
   champion env then candidate overrides; runs
   `bench_vs_strix.py --bot-dir <d> --tl 0.44 --sims <s> --games <g>
   --openings openings_human.pkl --out overnight/<name>_s<s>.json`.
   Skip if the output json already exists and is non-empty (resumable,
   same trick as grid_row.sh).
3. **submit script**: reads manifest, submits all gates, then a collector
   job with `--dependency=afterany:<all ids>`.
4. **collector**: reads all jsons (seal wins = `strix_losses` field), emits
   SUMMARY.md — per-sims table (wins, Elo = 400*log10(p/(1-p)), binomial
   CI), deltas vs ctrl with two-proportion z, pooled s16+s64 ranking, and
   the adoption-rule verdict per candidate. Flag MISSING for failed jobs.
   numpy is available in the venv; scipy is NOT.
5. **Racing (optional v2)**: 50-game screen (openings 0–24), kill bottom
   half, extend leaders to 100. Saves ~35% of games; pure win when nodes
   are scarce, irrelevant when fully fanned out.

Build-hygiene notes for code-change candidates: `setup.py` does not track
header deps — `touch minimax_bot.cpp` before rebuild; compiler is icpc,
C++17. For search-code changes, check threads=1 bit-path identity vs the
previous build where the change claims to be SMP-only.

## Current round-1 candidates (first use of the harness, all env-only)

A: `SEAL_VCF_FK=8 SEAL_VCF_FB=20000` (defense-filter probe strength — was
   k/2, budget/6 when sequential; now parallel, the #1 autopsy-aligned bet)
B: `SEAL_ROOT_CAP=26` (root width, newly ~free under root-split SMP)
C: `SEAL_VCF_K=13 SEAL_VCF_BUDGET=60000` (proof depth, same parallel logic)
Plus combination corners AB/AC/BC/ABC and ctrl → 8 configs × {s16, s64}
× 100 games ≈ 7 node-hours, fully parallel.

Context docs: LOG.md (campaign history), memory files
`strix-campaign-lessons`, `strix-time-scaling-map`, `smp-parallel-search`
(the seven measurement laws). Autopsy corpus: `autopsy40.pkl` (40 recorded
losses) — user has ruled it out as a gate (play only), but it remains valid
for post-hoc analysis of why a candidate won or lost.

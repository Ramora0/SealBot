#!/usr/bin/env bash
# Post-grid verification: solo NPS probe (vs 59k contended) + solo re-run
# of the s64/tl0.44 cell to separate GPU/CPU-contention effect from
# node effect vs last night's anchor (29% on openings 0-49).
set -uo pipefail
cd "$(dirname "$0")"
PY=/users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python

export SEAL_EVAL=trunk SEAL_TRUNK_POLICY=1 SEAL_TRUNK_BLEND=0 \
       SEAL_POLICY_MODE=74 SEAL_VCF=15 SEAL_VCF_K=11 SEAL_VCF_BUDGET=40000

echo "--- solo NPS probe ($(date +%H:%M)) ---"
$PY nps_probe.py 2>&1 | tail -1

echo "--- solo re-bench s64/tl0.44 ($(date +%H:%M)) ---"
$PY bench_vs_strix.py --bot-dir current --games 100 --tl 0.44 --sims 64 \
    --openings openings_human.pkl --out solo_s64_tl0.44.json 2>&1 \
    | tee solo_s64_tl0.44.log | tail -2

rm -f grid_s256_tl0.11.json grid_s256_tl0.44.json grid_s256_tl1.76.json
echo "POST-VERIFY COMPLETE ($(date +%H:%M))"

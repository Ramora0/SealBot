#!/usr/bin/env bash
# One sims-row of the scaling grid, usage: grid_row.sh <sims> <tl>...
# Rows are parallel-safe: strix strength is sims-based, so GPU contention
# does not affect results, only the measured latency (use solo latencies
# for the time axis). Cells with an existing non-empty json are skipped.
set -uo pipefail
cd "$(dirname "$0")"
PY=/users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python
sims=$1; shift
GAMES=${GAMES:-100}

export SEAL_EVAL=trunk SEAL_TRUNK_POLICY=1 SEAL_TRUNK_BLEND=0 \
       SEAL_POLICY_MODE=74 SEAL_VCF=15 SEAL_VCF_K=11 SEAL_VCF_BUDGET=40000

for tl in "$@"; do
    tag="grid_s${sims}_tl${tl}"
    [ -s "$tag.json" ] && { echo "skip $tag"; continue; }
    echo "=== $tag ($(date +%H:%M)) ==="
    $PY bench_vs_strix.py --bot-dir current --games "$GAMES" \
        --tl "$tl" --sims "$sims" --openings openings_human.pkl \
        --out "$tag.json" 2>&1 | tee "$tag.log" | tail -2
done
echo "ROW s${sims} COMPLETE ($(date +%H:%M))"

#!/usr/bin/env bash
# Scaling grid: v1.5 champion (current/) vs strix across sealbot-tl x strix-sims.
# Paired human-openings protocol, 100 games/cell (openings 0-49, both colors).
# Champion env fixed; strix latency measured per cell by the bench itself.
# Resumable: cells with an existing non-empty json are skipped.
set -uo pipefail
cd "$(dirname "$0")"
PY=/users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python
GAMES=${GAMES:-100}

export SEAL_EVAL=trunk SEAL_TRUNK_POLICY=1 SEAL_TRUNK_BLEND=0 \
       SEAL_POLICY_MODE=74 SEAL_VCF=15 SEAL_VCF_K=11 SEAL_VCF_BUDGET=40000

for sims in 64 16 4 256; do
    for tl in 0.11 0.44 1.76; do
        tag="grid_s${sims}_tl${tl}"
        [ -s "$tag.json" ] && { echo "skip $tag"; continue; }
        echo "=== $tag ($(date +%H:%M)) ==="
        $PY bench_vs_strix.py --bot-dir current --games "$GAMES" \
            --tl "$tl" --sims "$sims" --openings openings_human.pkl \
            --out "$tag.json" 2>&1 | tee "$tag.log" | tail -2
    done
done
echo "GRID COMPLETE ($(date +%H:%M))"

#!/usr/bin/env bash
# Handicap ladder: strix at reduced sims vs each SealBot build, sealbot tl
# fixed at its equal-time budget (0.44 s/turn from the sims=64 calibration).
# Run with the CPU otherwise quiet — sealbot times are wall-clock.
set -uo pipefail
cd "$(dirname "$0")"
PY=/users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python
GAMES=${1:-100}

for sims in 16 4; do
    for bot in best champion_frozen distill_frozen; do
        tag="ladder_${bot}_s${sims}"
        echo "=== $tag ($(date +%H:%M)) ==="
        $PY bench_vs_strix.py --bot-dir "$bot" --games "$GAMES" --tl 0.44 \
            --sims "$sims" --out "$tag.json" 2>&1 | tail -3
    done
done
echo "LADDER COMPLETE"

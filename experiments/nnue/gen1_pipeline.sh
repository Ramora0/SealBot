#!/usr/bin/env bash
# Generation-1 self-improvement loop, run AFTER current/ holds the best
# hybrid bot: self-play data from current/, deep-relabel with current/,
# retrain, gate at the current best blend.
# Usage: ./gen1_pipeline.sh <loss> <lam> <blend>
set -uo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
LOSS=${1:-score}
LAM=${2:-0.85}
BLEND=${3:-0.15}

echo "[gen1] self-play from current/ (30k games)..."
$PY datagen.py --games 30000 --workers 32 --out data/gen1 \
    --bot-dir ../../current --tl-min 0.03 --tl-max 0.06 2>&1 | tail -2

echo "[gen1] deep relabel with current/ (tl 0.12)..."
$PY relabel.py --in data/gen1 --out data/gen1_deep \
    --bot-dir ../../current --tl 0.12 --workers 30 2>&1 | tail -2

echo "[gen1] training ($LOSS, lam $LAM)..."
$PY train.py --data data/gen1_deep --out output/gen1_$LOSS \
    --loss "$LOSS" --lam "$LAM" --epochs 60 --threads 30 2>&1 | tail -3

echo "=== GATE gen1_$LOSS blend $BLEND (200 games) ==="
./gate.sh "output/gen1_$LOSS/net.pt" 200 0.1 "$BLEND" 2>&1 \
    | grep -E "wins |win rate|Elo difference|p-value"
echo "[gen1] COMPLETE"

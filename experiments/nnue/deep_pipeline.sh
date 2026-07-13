#!/usr/bin/env bash
# Self-driving: wait for relabel, train score+wdl variants on deep labels,
# gate each at blends {0, 0.15, 0.3}.
set -uo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

echo "[pipeline] waiting for relabel to finish..."
while pgrep -f relabel.py > /dev/null; do sleep 30; done
echo "[pipeline] relabel done: $(ls data/gen0_deep/*.pkl | wc -l) shards"

echo "[pipeline] training deep_score (Huber, lam 0.85)..."
$PY train.py --data data/gen0_deep --out output/deep_score \
    --loss score --lam 0.85 --epochs 60 --threads 30 2>&1 | tail -3

echo "[pipeline] training deep_wdl (BCE, lam 0.7)..."
$PY train.py --data data/gen0_deep --out output/deep_wdl \
    --loss wdl --lam 0.7 --epochs 60 --threads 30 2>&1 | tail -3

for ck in deep_score deep_wdl; do
  for b in 0.0 0.15 0.3; do
    echo "=== GATE $ck blend $b ==="
    ./gate.sh "output/$ck/net.pt" 100 0.1 "$b" 2>&1 | grep -E "wins |win rate|Elo difference|p-value"
  done
done
echo "[pipeline] COMPLETE"

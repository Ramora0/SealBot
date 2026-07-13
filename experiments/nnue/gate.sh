#!/usr/bin/env bash
# Gate a trained checkpoint: emit header, rebuild current/, evaluate vs best/.
# Usage: ./gate.sh output/gen0/net.pt [num_games] [time_limit]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../..
CKPT=${1:?usage: gate.sh <ckpt> [n] [tl]}
N=${2:-100}
TL=${3:-0.1}

$ROOT/.venv/bin/python emit_net.py --ckpt "$CKPT"
# setuptools does not track header deps -- force full rebuild
(cd $ROOT/current && rm -rf build *.so *.egg-info \
    && ../.venv/bin/python setup.py build_ext --inplace >/dev/null 2>&1)
echo "=== gate: $CKPT vs best/ ($N games @ ${TL}s) ==="
$ROOT/.venv/bin/python $ROOT/evaluate.py -n "$N" -t "$TL" --no-tqdm

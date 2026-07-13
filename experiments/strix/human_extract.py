"""Human-game positions (KrakenBot distill_100k.parquet) -> strix labels.

Two stages (different venvs):
  prep  (SealBot venv, has pyarrow): parquet -> human_prep/prep_XXXX.pkl
        dedup by board, cap positions per game, midgame filter.
  label (hexo venv, GPU): prep shards -> human_targets/*.npz in the
        policy_targets format PLUS inline "val" (strix value, mover POV)
        — value and policy logits from ONE forward pass.

    .../SealBot/.venv/bin/python human_extract.py --stage prep
    .../hexo-strix/.venv/bin/python human_extract.py --stage label
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

PARQUET = "/users/PAS2836/leedavis/personal/KrakenBot/distill_100k.parquet"
PREP_DIR = SCRIPT_DIR / "human_prep"
OUT_DIR = SCRIPT_DIR / "human_targets"

_D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
       if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def candidate_cells(cells):
    occ = {(q, r) for q, r, _ in cells}
    out = set()
    for q, r, _ in cells:
        for dq, dr in _D2:
            c = (q + dq, r + dr)
            if c not in occ:
                out.add(c)
    return out


def stage_prep(max_per_game, shard_size, min_stones, max_stones):
    import pyarrow.parquet as pq
    os.makedirs(PREP_DIR, exist_ok=True)
    t = pq.read_table(PARQUET,
                      columns=["board", "current_player", "game_id"])
    boards = t["board"].to_pylist()
    movers = t["current_player"].to_pylist()
    gids = t["game_id"].to_pylist()
    rng = random.Random(11)
    idx = list(range(len(boards)))
    rng.shuffle(idx)
    seen, per_game = set(), {}
    recs = []
    for i in idx:
        b = boards[i]
        if b in seen:
            continue
        if per_game.get(gids[i], 0) >= max_per_game:
            continue
        d = json.loads(b)
        if not (min_stones <= len(d) <= max_stones):
            continue
        seen.add(b)
        per_game[gids[i]] = per_game.get(gids[i], 0) + 1
        cells = [(int(k.split(",")[0]), int(k.split(",")[1]), int(v))
                 for k, v in d.items()]
        recs.append((cells, int(movers[i]), 2, len(cells)))
    rng.shuffle(recs)
    print(f"prep: {len(recs)} positions from {len(per_game)} games")
    for s in range(0, len(recs), shard_size):
        with open(PREP_DIR / f"prep_{s // shard_size:04d}.pkl", "wb") as fh:
            pickle.dump(recs[s:s + shard_size], fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {(len(recs) + shard_size - 1) // shard_size} shards")


def stage_label(chunk):
    import torch
    from strix_bridge import load_strix, state_from_cells
    from hexo_a0.graph import axis_states_to_batch

    model, mc, gc = load_strix()
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(str(PREP_DIR / "*.pkl")))
    total = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        out_path = OUT_DIR / (os.path.basename(f).replace(".pkl", ".npz"))
        if out_path.exists():
            continue
        recs = pickle.load(open(f, "rb"))
        states, metas = [], []
        for cells, mover, ml, mcnt in recs:
            s = state_from_cells(cells, mover, ml, gc)
            if s is None or s.is_terminal():
                continue
            states.append(s)
            metas.append((cells, mover, ml, mcnt))
        pols, vals = [], []
        with torch.inference_mode():
            for i in range(0, len(states), chunk):
                sub = states[i:i + chunk]
                batch, aux = axis_states_to_batch(
                    sub, prune_empty_edges=mc.prune_empty_edges,
                    threat_features=mc.threat_features,
                    relative_stones=mc.relative_stone_encoding,
                    device="cuda:0")
                lg, cnt, vv = model._forward_batch_core(
                    batch, legal_idx=aux.legal_idx, stone_idx=aux.stone_idx,
                    stone_batch=aux.stone_batch)
                lg = lg.float().cpu().numpy()
                cnt = cnt.cpu().numpy()
                vals.extend(vv.float().cpu().tolist())
                off = 0
                for s, n in zip(sub, cnt):
                    lm = s.legal_moves()
                    pols.append(dict(zip(map(tuple, lm), lg[off:off + n])))
                    off += n
        cq, cr, lgf, lens, keep_meta, keep_val = [], [], [], [], [], []
        for (cells, mover, ml, mcnt), pol, v in zip(metas, pols, vals):
            owner = {(q, r): p for q, r, p in cells}
            if owner.get((0, 0)) != 1:
                a = [(q, r) for q, r, p in cells if p == 1][0]
            else:
                a = (0, 0)
            cands = candidate_cells(cells)
            got = [(c, pol.get((c[0] - a[0], c[1] - a[1])))
                   for c in sorted(cands)]
            got = [(c, x) for c, x in got if x is not None]
            if len(got) < 2:
                continue
            for (q, r), x in got:
                cq.append(q); cr.append(r); lgf.append(x)
            lens.append(len(got))
            keep_meta.append((cells, mover, ml, mcnt))
            keep_val.append(float(v))
        np.savez_compressed(
            out_path,
            cell_q=np.array(cq, dtype=np.int16),
            cell_r=np.array(cr, dtype=np.int16),
            logit=np.array(lgf, dtype=np.float32),
            lens=np.array(lens, dtype=np.int32),
            meta=np.array(keep_meta, dtype=object),
            val=np.array(keep_val, dtype=np.float32),
            n=len(lens),
        )
        total += len(lens)
        rate = total / max(time.time() - t0, 1e-9)
        print(f"[{fi+1}/{len(files)}] {total} pos, {rate:.0f} pos/s",
              flush=True)
    print(f"DONE {total} positions -> {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "label"])
    ap.add_argument("--max-per-game", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=4000)
    ap.add_argument("--min-stones", type=int, default=5)
    ap.add_argument("--max-stones", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep(args.max_per_game, args.shard_size, args.min_stones,
                   args.max_stones)
    else:
        stage_label(args.chunk)


if __name__ == "__main__":
    main()

"""Data builders for mixnet_train.py (see MIXNET_DESIGN.md).

Two products:

1. --vals: strix single-forward VALUE sidecars for the existing
   policy_targets/*.npz shards (their gen0 value pkls live only on the
   cluster). Writes val_targets/<shard>.npy, float32 aligned with the
   shard's meta order, NaN where the position is strix-unrepresentable.
   These complete the 75% teacher-soft stream locally.

2. --bench: the TRUE-label stream from recorded bench games
   (bench_*.games.pkl). For every stone strix placed: the position
   before it, the D2 candidate set (plus the played cell), strix
   forward policy logits + value over it (soft labels), the played
   cell index (true policy) and the game outcome from the mover's POV
   (true value). Writes bench_targets/<src>.npz in policy_targets
   format plus val/played/outcome arrays.

Run in the hexo venv (GPU):
    python mixnet_data.py --vals
    python mixnet_data.py --bench
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import strix_bridge

_LOCAL_CKPT = Path("C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")
if not strix_bridge.CKPT.exists() and _LOCAL_CKPT.exists():
    strix_bridge.CKPT = _LOCAL_CKPT

from policy_extract import candidate_cells
from strix_bridge import load_strix, state_from_cells, value_batch


def policy_value_batch(model, mc, states, device="cuda:0", chunk=256):
    """Full forward; per-state ({cell: logit} over HeXO-legal cells, value)."""
    import torch
    from hexo_a0.graph import axis_states_to_batch

    out = []
    with torch.inference_mode():
        for i in range(0, len(states), chunk):
            sub = states[i:i + chunk]
            batch, aux = axis_states_to_batch(
                sub,
                prune_empty_edges=mc.prune_empty_edges,
                threat_features=mc.threat_features,
                relative_stones=mc.relative_stone_encoding,
                device=device,
            )
            logits, counts, values = model._forward_batch_core(
                batch, legal_idx=aux.legal_idx, stone_idx=aux.stone_idx,
                stone_batch=aux.stone_batch)
            logits = logits.float().cpu().numpy()
            counts = counts.cpu().numpy()
            values = values.float().cpu().tolist()
            off = 0
            for s, n, v in zip(sub, counts, values):
                lm = s.legal_moves()          # sorted by coord = node order
                out.append((dict(zip(map(tuple, lm), logits[off:off + n])),
                            float(v)))
                off += n
    return out


# ── 1. value sidecars for policy_targets ────────────────────────────────

def build_val_sidecars(chunk):
    model, mc, gc = load_strix()
    out_dir = SCRIPT_DIR / "val_targets"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(str(SCRIPT_DIR / "policy_targets" / "*.npz")))
    total = skipped = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        out_path = out_dir / (os.path.basename(f).replace(".npz", ".npy"))
        if out_path.exists():
            continue
        metas = np.load(f, allow_pickle=True)["meta"]
        states, idx = [], []
        for i, (cells, mover, ml, mc_) in enumerate(metas):
            s = state_from_cells([tuple(c) for c in cells], int(mover),
                                 int(ml), gc)
            if s is None:
                skipped += 1
                continue
            states.append(s)
            idx.append(i)
        vals = value_batch(model, mc, states, chunk=chunk)
        arr = np.full(len(metas), np.nan, dtype=np.float32)
        arr[np.array(idx, dtype=np.int64)] = np.array(vals, dtype=np.float32)
        tmp = str(out_path) + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_path)
        total += len(states)
        if (fi + 1) % 20 == 0 or fi + 1 == len(files):
            rate = total / max(time.time() - t0, 1e-9)
            print(f"[{fi+1}/{len(files)}] {total} pos, {skipped} skipped, "
                  f"{rate:.0f} pos/s", flush=True)
    print(f"DONE vals: {total} positions, {skipped} skipped -> {out_dir}")


# ── 2. bench-game true+soft stream ──────────────────────────────────────

def bench_events(games):
    """(cells, mover, ml, move_count, played, outcome_pov) per strix stone."""
    for g in games:
        strix_player = 1 if g["hexo_is_a"] else 2
        winner = int(g.get("winner", 0))
        cells = []
        for k, (q, r, p) in enumerate(g["seq"]):
            ml = 1 if k == 0 else 2 - ((k - 1) % 2)
            if p == strix_player and cells:
                oc = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
                yield list(cells), int(p), int(ml), k, (int(q), int(r)), oc
            cells.append((int(q), int(r), int(p)))


def build_bench(chunk):
    model, mc, gc = load_strix()
    out_dir = SCRIPT_DIR / "bench_targets"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(str(SCRIPT_DIR / "bench_*.games.pkl"))
                   + glob.glob(str(SCRIPT_DIR / "data_runs" / "*.games.pkl")))
    total = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        name = os.path.basename(f).replace(".json.games.pkl", "")
        out_path = out_dir / f"{name}.npz"
        if out_path.exists():
            continue
        games = pickle.load(open(f, "rb"))
        # Openings 50-74 are the HELD-OUT set (transfer checks) — they must
        # never enter training data. Bench plays opening game_idx//2.
        games = [g for g in games if g.get("game_idx", 0) // 2 < 50]
        evs, states = [], []
        for ev in bench_events(games):
            s = state_from_cells(ev[0], ev[1], ev[2], gc)
            if s is None:
                continue
            evs.append(ev)
            states.append(s)
        pvs = policy_value_batch(model, mc, states, chunk=chunk)

        cq, cr, lg, lens = [], [], [], []
        keep_meta, vals, played, outc = [], [], [], []
        for (cells, mover, ml, mcnt, pl_cell, oc), (pol, v) in zip(evs, pvs):
            owner = {(q, r): pp for q, r, pp in cells}
            if owner.get((0, 0)) != 1:
                a = [(q, r) for q, r, pp in cells if pp == 1][0]
            else:
                a = (0, 0)
            cands = candidate_cells(cells)
            cands.add(pl_cell)
            got = [(c, pol.get((c[0] - a[0], c[1] - a[1])))
                   for c in sorted(cands)]
            got = [(c, x) for c, x in got if x is not None]
            if len(got) < 2:
                continue
            pi = next((j for j, (c, _) in enumerate(got) if c == pl_cell), -1)
            for (q, r), x in got:
                cq.append(q); cr.append(r); lg.append(x)
            lens.append(len(got))
            keep_meta.append((cells, mover, ml, mcnt))
            vals.append(v); played.append(pi); outc.append(oc)
        np.savez_compressed(
            out_path,
            cell_q=np.array(cq, dtype=np.int16),
            cell_r=np.array(cr, dtype=np.int16),
            logit=np.array(lg, dtype=np.float32),
            lens=np.array(lens, dtype=np.int32),
            meta=np.array(keep_meta, dtype=object),
            n=len(lens),
            val=np.array(vals, dtype=np.float32),
            played=np.array(played, dtype=np.int32),
            outcome=np.array(outc, dtype=np.float32),
        )
        total += len(lens)
        print(f"[{fi+1}/{len(files)}] {name}: +{len(lens)} pos "
              f"({total} total, {time.time()-t0:.0f}s)", flush=True)
    print(f"DONE bench: {total} positions -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    if args.vals:
        build_val_sidecars(args.chunk)
    if args.bench:
        build_bench(min(args.chunk, 256))
    if not (args.vals or args.bench):
        print("nothing to do: pass --vals and/or --bench")


if __name__ == "__main__":
    main()

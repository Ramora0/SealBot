"""Extract strix policy targets for SealBot positions.

For each gen0 position: run the full strix forward (policy + value), keep
the logits of cells in OUR candidate set (hex distance <= 2 from any stone,
intersected with HeXO-legal), the side to move's POV. Writes packed npz
shards: per position, candidate cell coords + strix logits.

Run in the hexo venv:
    python policy_extract.py --src gen0_deep --out policy_targets \
        --max-positions 400000
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from strix_bridge import load_strix, state_from_cells

SEAL = Path("/users/PAS2836/leedavis/personal/SealBot")
DATA = SEAL / "experiments/nnue/data"

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


def policy_batch(model, mc, states, device="cuda:0", chunk=256):
    """Full forward; returns per-state {cell: logit} over HeXO-legal cells."""
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
            logits, counts, _ = model._forward_batch_core(
                batch, legal_idx=aux.legal_idx, stone_idx=aux.stone_idx,
                stone_batch=aux.stone_batch)
            logits = logits.float().cpu().numpy()
            counts = counts.cpu().numpy()
            off = 0
            for s, n in zip(sub, counts):
                lm = s.legal_moves()          # sorted by coord = node order
                out.append(dict(zip(map(tuple, lm), logits[off:off + n])))
                off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gen0_deep")
    ap.add_argument("--out", default="policy_targets")
    ap.add_argument("--max-positions", type=int, default=400_000)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    model, mc, gc = load_strix()
    out_dir = Path(__file__).parent / args.out
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(str(DATA / args.src / "*.pkl")))
    total = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        if total >= args.max_positions:
            break
        out_path = out_dir / (os.path.basename(f).replace(".pkl", ".npz"))
        if out_path.exists():
            with np.load(out_path, allow_pickle=True) as d:
                total += int(d["n"])
            continue
        games = pickle.load(open(f, "rb"))
        states, metas = [], []
        for g in games:
            for p in g["positions"]:
                s = state_from_cells(p["cells"], p["mover"], p["moves_left"],
                                     gc)
                if s is None:
                    continue
                states.append(s)
                metas.append(p)
        pols = policy_batch(model, mc, states, chunk=args.chunk)

        # pack: flat arrays + per-position offsets
        cq, cr, lg, lens = [], [], [], []
        keep_meta = []
        for p, pol in zip(metas, pols):
            # translate policy dict back if the bridge translated the board
            owner = {(q, r): pl for q, r, pl in p["cells"]}
            if owner.get((0, 0)) != 1:
                a = [(q, r) for q, r, pl in p["cells"] if pl == 1][0]
            else:
                a = (0, 0)
            cands = candidate_cells(p["cells"])
            got = [(c, pol.get((c[0] - a[0], c[1] - a[1])))
                   for c in sorted(cands)]
            got = [(c, v) for c, v in got if v is not None]
            if len(got) < 2:
                continue
            for (q, r), v in got:
                cq.append(q); cr.append(r); lg.append(v)
            lens.append(len(got))
            keep_meta.append((p["cells"], p["mover"], p["moves_left"],
                              p["move_count"]))
        np.savez_compressed(
            out_path,
            cell_q=np.array(cq, dtype=np.int16),
            cell_r=np.array(cr, dtype=np.int16),
            logit=np.array(lg, dtype=np.float32),
            lens=np.array(lens, dtype=np.int32),
            meta=np.array(keep_meta, dtype=object),
            n=len(lens),
        )
        total += len(lens)
        rate = total / max(time.time() - t0, 1e-9)
        print(f"[{fi+1}/{len(files)}] {total} pos, {rate:.0f} pos/s",
              flush=True)
    print(f"DONE {total} positions -> {out_dir}")


if __name__ == "__main__":
    main()

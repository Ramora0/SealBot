"""Sibling child-value sets for contrastive value training.

For sampled parents from policy_targets/ (gen0) and human_targets/ (human):
pick --children candidates (top by strix parent logit + random tail), place
the stone, get strix VALUE of each child (one forward), store in PARENT-
mover POV. These sibling groups are exactly what search compares — the
battery showed pointwise training never learns to resolve them.

Writes sibling_targets/*.npz per source shard:
  pcells (object), pmover, pml, pmc  : parent meta   [nparent]
  glens                              : children per parent
  cq, cr                             : child stone coords (flat)
  val                                : strix value, PARENT-mover POV
  flip                               : +1 same mover / -1 opponent child

Run in the hexo venv:
    .../hexo-strix/.venv/bin/python sibling_extract.py \
        --parents-per-shard 400 --children 10
"""

import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from strix_bridge import load_strix, state_from_cells, value_batch

OUT_DIR = SCRIPT_DIR / "sibling_targets"

DIRS6 = ((1, 0), (0, 1), (1, -1))


def completes_six(occ, q, r, p):
    """True if the stone just placed at (q,r) by p makes a 6+ run.
    hexo_rs.GameState.from_state does NOT flag completed boards as terminal
    (audit finding 1), and strix values on won boards are garbage — so
    detect wins ourselves and never send them to strix."""
    for dq, dr in DIRS6:
        run = 1
        for s in (1, -1):
            k = 1
            while occ.get((q + s * k * dq, r + s * k * dr)) == p:
                run += 1
                k += 1
        if run >= 6:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents-per-shard", type=int, default=400)
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--top-frac", type=float, default=0.6,
                    help="fraction of children from top strix logits")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    model, mc_, gc = load_strix()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(23)

    shards = (sorted(glob.glob(str(SCRIPT_DIR / "policy_targets" / "*.npz")))
              + sorted(glob.glob(str(SCRIPT_DIR / "human_targets" / "*.npz"))))
    if args.max_shards:
        shards = shards[:args.max_shards]
    total_p = total_c = 0
    t0 = time.time()
    for fi, f in enumerate(shards):
        tag = ("g" if "policy_targets" in f else "h") + \
            os.path.basename(f).replace(".npz", "")
        out_path = OUT_DIR / f"{tag}.npz"
        if out_path.exists():
            with np.load(out_path, allow_pickle=True) as d:
                total_p += len(d["pmover"])
            continue
        d = np.load(f, allow_pickle=True)
        lens, cq, cr, lg = d["lens"], d["cell_q"], d["cell_r"], d["logit"]
        metas = d["meta"]
        offs = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offs[1:])
        ids = list(range(len(metas)))
        rng.shuffle(ids)
        ids = ids[:args.parents_per_shard]

        parents = []      # (cells, mover, ml, mc, [child cells coords])
        child_desc = []   # flat: (ccells, cm, cml, flip, parent_idx)
        n_top = max(1, int(round(args.children * args.top_frac)))
        for i in ids:
            cells, mover, ml, mcnt = metas[i]
            cells = [tuple(c) for c in cells]
            sl = slice(offs[i], offs[i + 1])
            cand = list(zip(cq[sl].tolist(), cr[sl].tolist()))
            lgs = lg[sl]
            if len(cand) < 4:
                continue
            order = np.argsort(-lgs)
            picks = list(order[:n_top])
            rest = [j for j in order[n_top:]]
            rng.shuffle(rest)
            picks += rest[:args.children - len(picks)]
            kept = []
            for j in picks:
                c = cand[j]
                ccells = cells + [(c[0], c[1], int(mover))]
                if int(ml) == 2:
                    cm, cml = int(mover), 1
                else:
                    cm, cml = 3 - int(mover), 2
                child_desc.append((ccells, cm, cml,
                                   1.0 if cm == int(mover) else -1.0,
                                   len(parents)))
                kept.append(c)
            parents.append((cells, int(mover), int(ml), int(mcnt), kept))

        states, term, drop = [], [], []
        for ccells, cm, cml, flip, pi in child_desc:
            q, r, pl = ccells[-1]
            occ = {(c[0], c[1]): c[2] for c in ccells}
            if completes_six(occ, q, r, pl):
                drop.append(False); term.append(True)
                continue
            s = state_from_cells(ccells, cm, cml, gc)
            if s is None:
                drop.append(True); term.append(False)
            else:
                drop.append(False); term.append(False)
                states.append(s)
        vals = value_batch(model, mc_, states, chunk=args.chunk)

        pcells, pmover, pml, pmc, glens = [], [], [], [], []
        fcq, fcr, fval, fflip = [], [], [], []
        vi = 0
        per_parent = {}
        for (ccells, cm, cml, flip, pi), dr, tm in zip(child_desc, drop,
                                                       term):
            if dr:
                continue
            if tm:
                v = 1.0          # parent mover completed 6
            else:
                v = flip * float(vals[vi])
            if not tm:
                vi += 1
            per_parent.setdefault(pi, []).append(
                (ccells[-1][0], ccells[-1][1], v, flip))
        for pi, (cells, mover, ml, mcnt, kept) in enumerate(parents):
            ch = per_parent.get(pi, [])
            if len(ch) < 2:
                continue
            pcells.append(cells); pmover.append(mover)
            pml.append(ml); pmc.append(mcnt)
            glens.append(len(ch))
            for q, r, v, fl in ch:
                fcq.append(q); fcr.append(r); fval.append(v); fflip.append(fl)
        np.savez_compressed(
            out_path,
            pcells=np.array(pcells, dtype=object),
            pmover=np.array(pmover, np.int8),
            pml=np.array(pml, np.int8),
            pmc=np.array(pmc, np.int16),
            glens=np.array(glens, np.int16),
            cq=np.array(fcq, np.int16), cr=np.array(fcr, np.int16),
            val=np.array(fval, np.float32),
            flip=np.array(fflip, np.float32),
        )
        total_p += len(pcells)
        total_c += len(fcq)
        if (fi + 1) % 10 == 0 or fi + 1 == len(shards):
            rate = total_c / max(time.time() - t0, 1e-9)
            print(f"[{fi+1}/{len(shards)}] {total_p} parents "
                  f"{total_c} children, {rate:.0f} ch/s", flush=True)
    print(f"DONE {total_p} parents -> {OUT_DIR}")


if __name__ == "__main__":
    main()

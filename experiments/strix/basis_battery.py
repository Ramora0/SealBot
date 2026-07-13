"""Feature-basis battery: which cell representation best fits strix's value?

All variants share window features EW[729] + globals + the same head; they
differ in the per-cell family:
  joint   — EC[8548] conjunction classes (current architecture)
  codes   — per-direction 6x6 (us,them) code embeddings, summed (this IS
            "concat the three lines then linear": no cross-line interaction,
            coarse alphabet)
  raw     — per-direction RAW 11-cell pattern embeddings E[3^11], summed
            (fine line detail, no cross-line interaction)
  joint+raw — interaction (coarse) + detail (fine)

Trained on gen0_strix labels (target = strix_v*8, Huber), identical
schedule; metric = held-out corr/spearman to the strix value.

Run in SealBot venv:  python basis_battery.py --variants joint codes raw joint_raw
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))

from features import CODEBOOK, CANON_EMPTY, CANON_OCC, PAD, DIRS, POW3_6, \
    POW3_11, NUM_CLASSES

DATA = os.path.join(SCRIPT_DIR, "..", "nnue", "data", "gen0_strix")
N_WIN, N_JOINT, N_RAW, N_CODES = 729, NUM_CLASSES, 3 ** 11, 36
OFF_JOINT = N_WIN
OFF_RAW = OFF_JOINT + N_JOINT
OFF_CODES = OFF_RAW + N_RAW
N_TOTAL = OFF_CODES + N_CODES
K, H, CLIP = 32, 32, 8.0


def extract_all(cells, mover):
    """Sparse counts over the combined feature space for one position."""
    qs = np.array([c[0] for c in cells]); rs = np.array([c[1] for c in cells])
    ps = np.array([c[2] for c in cells])
    q0, r0 = qs.min() - PAD, rs.min() - PAD
    Hh = int(qs.max() - q0 + PAD + 1); Ww = int(rs.max() - r0 + PAD + 1)
    grid = np.zeros((Hh, Ww), dtype=np.int64)
    grid[qs - q0, rs - r0] = np.where(ps == mover, 1, 2)

    idxs = []
    lps = []
    for dq, dr in DIRS:
        pat = np.zeros((Hh, Ww), dtype=np.int64)
        lp = np.zeros((Hh, Ww), dtype=np.int64)
        for j in range(6):
            sq, sr = j * dq, j * dr
            src = np.zeros((Hh, Ww), dtype=np.int64)
            src[max(0, -sq):min(Hh, Hh - sq), max(0, -sr):min(Ww, Ww - sr)] = \
                grid[max(0, -sq) + sq:min(Hh, Hh - sq) + sq,
                     max(0, -sr) + sr:min(Ww, Ww - sr) + sr]
            pat += src * POW3_6[j]
        for u in range(-5, 6):
            sq, sr = u * dq, u * dr
            src = np.zeros((Hh, Ww), dtype=np.int64)
            src[max(0, -sq):min(Hh, Hh - sq), max(0, -sr):min(Ww, Ww - sr)] = \
                grid[max(0, -sq) + sq:min(Hh, Hh - sq) + sq,
                     max(0, -sr) + sr:min(Ww, Ww - sr) + sr]
            lp += src * POW3_11[5 + u]
        nz = pat[pat > 0]
        if len(nz):
            idxs.append(np.unique(nz, return_counts=True))
        lps.append(lp)

    lp0, lp1, lp2 = lps
    active = (lp0 | lp1 | lp2) != 0

    # joint classes (identical to features.py)
    cb0, cb1, cb2 = CODEBOOK[lp0], CODEBOOK[lp1], CODEBOOK[lp2]
    p0 = (cb0 & 7).astype(np.int64) * 6 + (cb0 >> 3)
    p1 = (cb1 & 7).astype(np.int64) * 6 + (cb1 >> 3)
    p2 = (cb2 & 7).astype(np.int64) * 6 + (cb2 >> 3)
    cls_empty = CANON_EMPTY[(p0 * 36 + p1) * 36 + p2].astype(np.int64)
    own0 = np.where(grid == 1, cb0 & 7, cb0 >> 3).astype(np.int64)
    own1 = np.where(grid == 1, cb1 & 7, cb1 >> 3).astype(np.int64)
    own2 = np.where(grid == 1, cb2 & 7, cb2 >> 3).astype(np.int64)
    cls_occ = (8436 + (grid == 2) * 56
               + CANON_OCC[(own0 * 6 + own1) * 6 + own2])
    cls = np.where(grid == 0, cls_empty, cls_occ)[active]
    u, c = np.unique(cls, return_counts=True)
    idxs.append((u + OFF_JOINT, c))

    # raw per-direction patterns + code36 per direction (active cells)
    for lp, pp in ((lp0, p0), (lp1, p1), (lp2, p2)):
        u, c = np.unique(lp[active], return_counts=True)
        idxs.append((u + OFF_RAW, c))
        u, c = np.unique(pp[active], return_counts=True)
        idxs.append((u + OFF_CODES, c))

    fi = np.concatenate([a for a, _ in idxs]).astype(np.int64)
    fc = np.concatenate([b for _, b in idxs]).astype(np.int16)
    return fi, fc


def _shard(sp):
    out = []
    for g in pickle.load(open(sp, "rb")):
        for p in g["positions"]:
            fi, fc = extract_all(p["cells"], p["mover"])
            out.append((fi, fc, p["score"] / 1000.0, p["move_count"],
                        p.get("moves_left", 2)))
    return out


def build(cache, max_pos, workers):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(DATA, "*.pkl")))
    fis, fcs, lens, tgt, mc, ml = [], [], [], [], [], []
    n = 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_shard, shards)):
            for fi, fc, t, m, l in res:
                fis.append(fi); fcs.append(fc); lens.append(len(fi))
                tgt.append(t); mc.append(m); ml.append(l)
            n += len(res)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(shards)} shards {n} pos "
                      f"{time.time()-t0:.0f}s", flush=True)
            if n >= max_pos:
                pool.terminate()
                break
    offs = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    ds = {"fi": np.concatenate(fis), "fc": np.concatenate(fcs),
          "offs": offs, "tgt": np.array(tgt, np.float32),
          "mc": np.array(mc, np.int32), "ml": np.array(ml, np.int32)}
    np.savez(cache, **ds)
    print(f"battery dataset: {len(lens)} pos {time.time()-t0:.0f}s")
    return ds


VARIANTS = {
    "joint":     lambda i: (i < OFF_RAW),
    "codes":     lambda i: (i < OFF_JOINT) | (i >= OFF_CODES),
    "raw":       lambda i: (i < OFF_JOINT) | ((i >= OFF_RAW) & (i < OFF_CODES)),
    "joint_raw": lambda i: (i < OFF_CODES),
}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_TOTAL, K, mode="sum",
                                   include_last_offset=True)
        nn.init.normal_(self.emb.weight, 0.0, 0.03)
        self.w1 = nn.Linear(K + 2, H)
        self.w2 = nn.Linear(H, 1)

    def forward(self, idx, cnt, offsets, g0, g1):
        acc = self.emb(idx, offsets, per_sample_weights=cnt)
        x = torch.cat([torch.clamp(acc, 0.0, CLIP),
                       g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--max-pos", type=int, default=250_000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = os.path.join(SCRIPT_DIR, "output_battery")
    os.makedirs(out_dir, exist_ok=True)
    ds = build(os.path.join(out_dir, "battery_ds.npz"), args.max_pos,
               args.threads)
    n = len(ds["tgt"])
    offs = ds["offs"]
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    val_ids = np.sort(order[:8000])
    train_ids = order[8000:]
    g0 = torch.from_numpy((ds["mc"] * 0.02).astype(np.float32))
    g1 = torch.from_numpy((ds["ml"] * 0.5).astype(np.float32))
    tgt = torch.from_numpy(ds["tgt"])
    fi_all = torch.from_numpy(ds["fi"])
    fc_all = torch.from_numpy(ds["fc"].astype(np.float32))

    results = {}
    for name in args.variants:
        keep = VARIANTS[name]
        model = Net()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-4)
        loss_fn = nn.HuberLoss(delta=4.0)

        def gather(ids):
            g = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in ids])
            bi, bc = fi_all[g], fc_all[g]
            m = keep(bi)
            # rebuild offsets after masking
            lens = np.array([int(m[np.searchsorted(g, np.arange(offs[i], offs[i+1]))].sum())
                             for i in ids])  # slow path avoided below
            return bi, bc, m, g

        # precompute per-position masked lengths once per variant
        mask_full = keep(fi_all).numpy()
        mlens = np.add.reduceat(mask_full, offs[:-1])

        def batchify(ids):
            g = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in ids])
            m = mask_full[g]
            bi = fi_all[g][m]
            bc = fc_all[g][m]
            bo = np.zeros(len(ids) + 1, dtype=np.int64)
            np.cumsum(mlens[ids], out=bo[1:])
            return bi, bc, torch.from_numpy(bo), g0[ids], g1[ids], tgt[ids]

        t0 = time.time()
        for ep in range(args.epochs):
            rng.shuffle(train_ids)
            for s in range(0, len(train_ids), args.batch):
                ids = np.sort(train_ids[s:s + args.batch])
                bi, bc, bo, bg0, bg1, bt = batchify(ids)
                opt.zero_grad()
                loss = loss_fn(model(bi, bc, bo, bg0, bg1), bt)
                loss.backward()
                opt.step()
            sched.step()
        with torch.no_grad():
            preds = []
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                bi, bc, bo, bg0, bg1, _ = batchify(ids)
                preds.append(model(bi, bc, bo, bg0, bg1).numpy())
            pred = np.concatenate(preds)
        y = tgt[val_ids].numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        sp = spearman(pred, y)
        results[name] = (corr, sp)
        print(f"[{name:9s}] held-out corr {corr:.4f} spearman {sp:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(),
                   os.path.join(out_dir, f"battery_{name}.pt"))

    print("\n=== BASIS BATTERY (fit to strix value, gen0 held-out) ===")
    for name, (c, s) in results.items():
        print(f"  {name:9s}: corr {c:.4f}  spearman {s:.4f}")


if __name__ == "__main__":
    main()

"""Cell-level-nonlinearity architecture (the deep-sets NNUE variant):

  per cell:  s_c = E_raw[lp_dir0] + E_raw[lp_dir1] + E_raw[lp_dir2]
             cell_out = clamp(s_c, 0, CLIP)            (elementwise NL)
  global:    acc = sum_cells cell_out + EW-window sum
  head:      clamp(acc,0,CLIP) ++ globals -> 32 relu -> 1

Raw 3^11 line patterns, no codebook, cross-line interaction via the cell
nonlinearity. Engine-viable: updates touch the same ~31 cells the current
conjunction diff does, with a cached per-cell pre-activation.

Trains on gen0_strix labels; reports held-out corr/spearman vs strix value
for comparison with basis_battery variants.
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

from features import PAD, DIRS, POW3_6, POW3_11

DATA = os.path.join(SCRIPT_DIR, "..", "nnue", "data", "gen0_strix")
N_RAW = 3 ** 11
K, H, CLIP = 32, 32, 8.0


def extract(cells, mover):
    qs = np.array([c[0] for c in cells]); rs = np.array([c[1] for c in cells])
    ps = np.array([c[2] for c in cells])
    q0, r0 = qs.min() - PAD, rs.min() - PAD
    Hh = int(qs.max() - q0 + PAD + 1); Ww = int(rs.max() - r0 + PAD + 1)
    grid = np.zeros((Hh, Ww), dtype=np.int64)
    grid[qs - q0, rs - r0] = np.where(ps == mover, 1, 2)

    w_counts = np.zeros(729, dtype=np.int64)
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
            w_counts += np.bincount(nz, minlength=729)
        lps.append(lp)

    active = (lps[0] | lps[1] | lps[2]) != 0
    trip = np.stack([lp[active] for lp in lps], axis=1).astype(np.int32)
    wi = np.nonzero(w_counts)[0].astype(np.int32)
    return trip, wi, w_counts[wi].astype(np.int16)


def _shard(sp):
    out = []
    for g in pickle.load(open(sp, "rb")):
        for p in g["positions"]:
            trip, wi, wc = extract(p["cells"], p["mover"])
            out.append((trip, wi, wc, p["score"] / 1000.0, p["move_count"],
                        p.get("moves_left", 2)))
    return out


def build(cache, max_pos, workers):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(DATA, "*.pkl")))
    trips, wis, wcs, clens, wlens = [], [], [], [], []
    tgt, mc, ml = [], [], []
    n = 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_shard, shards)):
            for trip, wi, wc, t, m, l in res:
                trips.append(trip); wis.append(wi); wcs.append(wc)
                clens.append(len(trip)); wlens.append(len(wi))
                tgt.append(t); mc.append(m); ml.append(l)
            n += len(res)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(shards)} shards {n} pos "
                      f"{time.time()-t0:.0f}s", flush=True)
            if n >= max_pos:
                pool.terminate()
                break
    coffs = np.zeros(len(clens) + 1, dtype=np.int64)
    np.cumsum(clens, out=coffs[1:])
    woffs = np.zeros(len(wlens) + 1, dtype=np.int64)
    np.cumsum(wlens, out=woffs[1:])
    ds = {"trip": np.concatenate(trips), "coffs": coffs,
          "wi": np.concatenate(wis), "wc": np.concatenate(wcs),
          "woffs": woffs, "tgt": np.array(tgt, np.float32),
          "mc": np.array(mc, np.int32), "ml": np.array(ml, np.int32)}
    np.savez(cache, **ds)
    print(f"cellnl dataset: {len(clens)} pos, "
          f"{len(ds['trip'])/len(clens):.0f} cells/pos, {time.time()-t0:.0f}s")
    return ds


class CellNL(nn.Module):
    def __init__(self):
        super().__init__()
        self.eraw = nn.Embedding(N_RAW, K, sparse=True)
        nn.init.normal_(self.eraw.weight, 0.0, 0.03)
        self.ew = nn.EmbeddingBag(729, K, mode="sum",
                                  include_last_offset=True)
        nn.init.normal_(self.ew.weight, 0.0, 0.03)
        self.w1 = nn.Linear(K + 2, H)
        self.w2 = nn.Linear(H, 1)

    def forward(self, trip, seg, npos, wi, wc, woff, g0, g1):
        s = self.eraw(trip).sum(dim=1)                    # [ncell, K]
        cell = torch.clamp(s, 0.0, CLIP)
        acc = torch.zeros(npos, K).index_add_(0, seg, cell)
        acc = acc + self.ew(wi, woff, per_sample_weights=wc)
        x = torch.cat([torch.clamp(acc, 0.0, CLIP),
                       g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = os.path.join(SCRIPT_DIR, "output_battery")
    os.makedirs(out_dir, exist_ok=True)
    ds = build(os.path.join(out_dir, "cellnl_ds.npz"), args.max_pos,
               args.threads)
    n = len(ds["tgt"])
    coffs, woffs = ds["coffs"], ds["woffs"]
    trip = torch.from_numpy(ds["trip"].astype(np.int64))
    wi = torch.from_numpy(ds["wi"].astype(np.int64))
    wc = torch.from_numpy(ds["wc"].astype(np.float32))
    tgt = torch.from_numpy(ds["tgt"])
    g0 = torch.from_numpy((ds["mc"] * 0.02).astype(np.float32))
    g1 = torch.from_numpy((ds["ml"] * 0.5).astype(np.float32))

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    val_ids, train_ids = np.sort(order[:6000]), order[6000:]

    model = CellNL()
    dense = [p for nm, p in model.named_parameters() if "eraw" not in nm]
    opt_d = torch.optim.Adam(dense, lr=2e-3)
    opt_s = torch.optim.SparseAdam(model.eraw.parameters(), lr=2e-3)
    loss_fn = nn.HuberLoss(delta=4.0)

    def batchify(ids):
        cg = np.concatenate([np.arange(coffs[i], coffs[i + 1]) for i in ids])
        clens = (coffs[ids + 1] - coffs[ids])
        seg = torch.from_numpy(np.repeat(np.arange(len(ids)), clens))
        wg = np.concatenate([np.arange(woffs[i], woffs[i + 1]) for i in ids])
        wo = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(woffs[ids + 1] - woffs[ids], out=wo[1:])
        return (trip[cg], seg, len(ids), wi[wg], wc[wg],
                torch.from_numpy(wo), g0[ids], g1[ids], tgt[ids])

    print(f"training cellnl on {len(train_ids)} positions")
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        t0 = time.time()
        tot = nb = 0
        for s in range(0, len(train_ids), args.batch):
            ids = np.sort(train_ids[s:s + args.batch])
            *inp, bt = batchify(ids)
            opt_d.zero_grad(); opt_s.zero_grad()
            loss = loss_fn(model(*inp), bt)
            loss.backward()
            opt_d.step(); opt_s.step()
            tot += float(loss.detach()); nb += 1
        with torch.no_grad():
            preds = []
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                *inp, _ = batchify(ids)
                preds.append(model(*inp).numpy())
            pred = np.concatenate(preds)
        y = tgt[val_ids].numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        print(f"epoch {ep+1}/{args.epochs}: train {tot/nb:.4f} "
              f"val corr {corr:.4f} spearman {spearman(pred, y):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "cellnl.pt"))
    print("saved cellnl.pt")


if __name__ == "__main__":
    main()

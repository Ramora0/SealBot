"""Joint trunk training: ONE shared cellnl trunk, TWO heads (value + policy).

  per cell:  s_c = E_raw[lp_dir0] + E_raw[lp_dir1] + E_raw[lp_dir2]
             a_c = clamp(s_c, 0, CLIP)               (cell-level NL)
  value:     acc = sum_cells a_c + EW-window bag
             clamp(acc) ++ globals -> H relu -> scalar   (as cellnl battery)
  policy:    logit(cand cell) = P2 relu(P1 a_cand)       (readout of SAME a_c)

Trains jointly: Huber(value, strix_v*8) + lam * listwise KL(policy || strix
logits over D2 candidates). Data = per-shard join of gen0_strix (value
labels) x policy_targets/*.npz (policy labels + meta), so every position
carries both targets.

Baselines to beat (same data family):
  value  (cellnl solo, 100k pos):  corr 0.9428  spearman 0.9356
  policy (PW/PC tables, 400k pos): top1 0.460   mean-rank-of-best 4.05

Run in the hexo venv (GPU):
    .../hexo-strix/.venv/bin/python trunk_train.py --device cuda
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
TARGETS = os.path.join(SCRIPT_DIR, "policy_targets")
N_RAW = 3 ** 11
K, H, HP, CLIP = 32, 32, 32, 8.0


def extract(cells, mover, cand):
    """trip [ncell,3] (raw 3^11 codes of active cells), window bag (wi, wc),
    cand_trip [ncand,3] (codes at candidate cells; may be all-zero)."""
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
    cg = np.array([(q - q0, r - r0) for q, r in cand])
    cand_trip = np.stack([lp[cg[:, 0], cg[:, 1]] for lp in lps],
                         axis=1).astype(np.int32)
    return trip, wi, w_counts[wi].astype(np.int16), cand_trip


def _key(cells, mover, ml, mc):
    return (int(mover), int(ml), int(mc), tuple(sorted(map(tuple, cells))))


def _shard(paths):
    npz_path, pkl_path, flag_path = (paths if len(paths) == 3
                                     else (*paths, None))
    vcf_flags = (np.load(flag_path) if flag_path and os.path.exists(flag_path)
                 else None)
    vals = {}
    if pkl_path is not None:
        for g in pickle.load(open(pkl_path, "rb")):
            for p in g["positions"]:
                vals[_key(p["cells"], p["mover"], p["moves_left"],
                          p["move_count"])] = p["score"] / 1000.0
    d = np.load(npz_path, allow_pickle=True)
    inline_val = d["val"] if "val" in d.files else None
    lens, cq, cr, lg = d["lens"], d["cell_q"], d["cell_r"], d["logit"]
    metas = d["meta"]
    offs = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    T, WI, WC, CT, LG = [], [], [], [], []
    tgt, mcs, mls = [], [], []
    miss = 0
    for i, (cells, mover, ml, mc) in enumerate(metas):
        if inline_val is not None:
            t = float(inline_val[i]) * 8.0
        else:
            t = vals.get(_key(cells, mover, ml, mc))
        if t is None:
            miss += 1
            continue
        # Proven forced win for the mover: soft floor (preserve gradation
        # among wins — hard saturation to 8.0 regressed play, v1.4).
        if vcf_flags is not None and vcf_flags[i] and t < 6.5:
            t = 6.5
        sl = slice(offs[i], offs[i + 1])
        cand = list(zip(cq[sl].tolist(), cr[sl].tolist()))
        trip, wi, wc, ctrip = extract([tuple(c) for c in cells], int(mover),
                                      cand)
        T.append(trip); WI.append(wi); WC.append(wc); CT.append(ctrip)
        LG.append(lg[sl])
        tgt.append(t); mcs.append(int(mc)); mls.append(int(ml))
    return (np.concatenate(T), np.array([len(x) for x in T], np.int32),
            np.concatenate(WI), np.concatenate(WC),
            np.array([len(x) for x in WI], np.int32),
            np.concatenate(CT), np.concatenate(LG).astype(np.float32),
            np.array([len(x) for x in CT], np.int32),
            np.array(tgt, np.float32), np.array(mcs, np.int32),
            np.array(mls, np.int32), miss)


def build(cache, workers, max_shards=None, human=False,
          vcf_labels=False, dagger=False):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    vcf_dir = os.path.join(SCRIPT_DIR, "vcf_targets")

    def _flag(npz, pref):
        f = os.path.join(vcf_dir, pref +
                         os.path.basename(npz).replace(".npz", ".npy"))
        return f if vcf_labels else None

    pairs = []
    for npz in sorted(glob.glob(os.path.join(TARGETS, "*.npz"))):
        pkl = os.path.join(DATA, os.path.basename(npz).replace(".npz", ".pkl"))
        if os.path.exists(pkl):
            pairs.append((npz, pkl, _flag(npz, "g_")))
    if human:
        for npz in sorted(glob.glob(os.path.join(SCRIPT_DIR, "human_targets",
                                                 "*.npz"))):
            pairs.append((npz, None, _flag(npz, "h_")))
    if dagger:
        for npz in sorted(glob.glob(os.path.join(SCRIPT_DIR, "dagger_targets",
                                                 "*.npz"))):
            pairs.append((npz, None, _flag(npz, "d_")))
    if max_shards:
        pairs = pairs[:max_shards]
    print(f"building joint dataset from {len(pairs)} shards...", flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("trip", "clen", "wi", "wc", "wlen", "ctrip",
                           "logit", "plen", "tgt", "mc", "ml")}
    miss = 0
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_shard, pairs)):
            for k, v in zip(acc, res[:-1]):
                acc[k].append(v)
            miss += res[-1]
            if (i + 1) % 10 == 0:
                n = sum(len(x) for x in acc["tgt"])
                print(f"  {i+1}/{len(pairs)} shards, {n} pos, {miss} unjoined,"
                      f" {time.time()-t0:.0f}s", flush=True)
    ds = {k: np.concatenate(v) for k, v in acc.items()}
    for off_k, len_k in (("coffs", "clen"), ("woffs", "wlen"),
                         ("poffs", "plen")):
        o = np.zeros(len(ds[len_k]) + 1, dtype=np.int64)
        np.cumsum(ds[len_k], out=o[1:])
        ds[off_k] = o
    np.savez(cache, **ds)
    print(f"joint dataset: {len(ds['tgt'])} pos "
          f"({ds['clen'].mean():.0f} cells, {ds['plen'].mean():.0f} cands"
          f"/pos), {miss} unjoined, {time.time()-t0:.0f}s", flush=True)
    return ds


def multi_arange(starts, lens):
    ends = lens.cumsum()
    idx = np.ones(int(ends[-1]), dtype=np.int64)
    idx[0] = starts[0]
    idx[ends[:-1]] = starts[1:] - (starts[:-1] + lens[:-1] - 1)
    return idx.cumsum()


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.eraw = nn.Embedding(N_RAW, K)
        nn.init.normal_(self.eraw.weight, 0.0, 0.03)
        self.ew = nn.EmbeddingBag(729, K, mode="sum",
                                  include_last_offset=True)
        nn.init.normal_(self.ew.weight, 0.0, 0.03)
        self.w1 = nn.Linear(K + 2, H)
        self.w2 = nn.Linear(H, 1)
        self.p1 = nn.Linear(K, HP)
        self.p2 = nn.Linear(HP, 1)

    def cell_act(self, trip):
        return torch.clamp(self.eraw(trip).sum(dim=1), 0.0, CLIP)

    def value(self, trip, seg, npos, wi, wc, woff, g0, g1):
        acc = torch.zeros(npos, K, device=trip.device).index_add_(
            0, seg, self.cell_act(trip))
        acc = acc + self.ew(wi, woff, per_sample_weights=wc)
        x = torch.cat([torch.clamp(acc, 0.0, CLIP),
                       g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)

    def policy(self, cand_trip):
        return self.p2(torch.relu(self.p1(self.cell_act(cand_trip)))
                       ).squeeze(1)


def seg_ce(scores, tl, seg, npos, device):
    """listwise soft-target CE per position (KL up to a constant)."""
    z = torch.zeros(npos, device=device)
    t_max = z.clone().index_reduce_(0, seg, tl, "amax", include_self=False)
    t_exp = torch.exp(tl - t_max[seg])
    t_sum = z.clone().index_add_(0, seg, t_exp)
    t_soft = t_exp / t_sum[seg]
    s_max = z.clone().index_reduce_(0, seg, scores, "amax",
                                    include_self=False)
    s_exp = torch.exp(scores - s_max[seg])
    s_sum = z.clone().index_add_(0, seg, s_exp)
    s_logsoft = (scores - s_max[seg]) - torch.log(s_sum[seg])
    return -(t_soft * s_logsoft).sum() / npos


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", default="output_trunk")
    ap.add_argument("--human", action="store_true",
                    help="include human_targets/ shards (inline strix vals)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    ds = build(os.path.join(out_dir, "trunk_ds.npz"), args.threads,
               args.max_shards, human=args.human)

    n = len(ds["tgt"])
    coffs, woffs, poffs = ds["coffs"], ds["woffs"], ds["poffs"]
    trip = torch.from_numpy(ds["trip"].astype(np.int64))
    wi = torch.from_numpy(ds["wi"].astype(np.int64))
    wc = torch.from_numpy(ds["wc"].astype(np.float32))
    ctrip = torch.from_numpy(ds["ctrip"].astype(np.int64))
    logit = torch.from_numpy(ds["logit"])
    tgt = torch.from_numpy(ds["tgt"])
    g0 = torch.from_numpy((ds["mc"] * 0.02).astype(np.float32))
    g1 = torch.from_numpy((ds["ml"] * 0.5).astype(np.float32))

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = min(8000, n // 10)
    val_ids, train_ids = np.sort(order[:n_val]), order[n_val:]

    model = Trunk().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    huber = nn.HuberLoss(delta=4.0)

    def batchify(ids):
        clens = (coffs[ids + 1] - coffs[ids])
        cg = multi_arange(coffs[ids], clens)
        seg_c = torch.from_numpy(np.repeat(np.arange(len(ids)), clens)).to(dev)
        wlens = (woffs[ids + 1] - woffs[ids])
        wg = multi_arange(woffs[ids], wlens)
        wo = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(wlens, out=wo[1:])
        plens = (poffs[ids + 1] - poffs[ids])
        pg = multi_arange(poffs[ids], plens)
        seg_p = torch.from_numpy(np.repeat(np.arange(len(ids)), plens)).to(dev)
        return (trip[cg].to(dev), seg_c, len(ids), wi[wg].to(dev),
                wc[wg].to(dev), torch.from_numpy(wo).to(dev),
                g0[ids].to(dev), g1[ids].to(dev),
                ctrip[pg].to(dev), logit[pg].to(dev), seg_p,
                tgt[ids].to(dev))

    print(f"training trunk on {len(train_ids)} positions, dev={dev}, "
          f"lam={args.lam}", flush=True)
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        t0 = time.time()
        tv = tp = nb = 0
        for s in range(0, len(train_ids), args.batch):
            ids = np.sort(train_ids[s:s + args.batch])
            (bt, seg_c, npos, bwi, bwc, bwo, bg0, bg1, bct, btl, seg_p,
             by) = batchify(ids)
            opt.zero_grad()
            lv = huber(model.value(bt, seg_c, npos, bwi, bwc, bwo, bg0, bg1),
                       by)
            lp = seg_ce(model.policy(bct), btl, seg_p, npos, dev)
            (lv + args.lam * lp).backward()
            opt.step()
            tv += float(lv.detach()); tp += float(lp.detach()); nb += 1
        sched.step()
        with torch.no_grad():
            preds, top1, rank_sum, nv = [], 0, 0, 0
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                (bt, seg_c, npos, bwi, bwc, bwo, bg0, bg1, bct, btl, seg_p,
                 _) = batchify(ids)
                preds.append(model.value(bt, seg_c, npos, bwi, bwc, bwo, bg0,
                                         bg1).cpu().numpy())
                sc = model.policy(bct).cpu().numpy()
                tl = btl.cpu().numpy()
                off = 0
                for L in (poffs[ids + 1] - poffs[ids]):
                    t, sl = tl[off:off + L], sc[off:off + L]
                    best = int(np.argmax(t))
                    top1 += int(int(np.argmax(sl)) == best)
                    rank_sum += int((sl > sl[best]).sum()) + 1
                    nv += 1
                    off += L
            pred = np.concatenate(preds)
        y = tgt[val_ids].numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        print(f"epoch {ep+1}/{args.epochs}: train v {tv/nb:.4f} p {tp/nb:.4f}"
              f" | val corr {corr:.4f} spearman {spearman(pred, y):.4f}"
              f" | top1 {top1/nv:.3f} mrank {rank_sum/nv:.2f}"
              f" ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "K": K, "H": H, "HP": HP, "CLIP": CLIP},
               os.path.join(out_dir, "trunk.pt"))
    print(f"saved {out_dir}/trunk.pt")


if __name__ == "__main__":
    main()

"""Trunk v1.1: battery-driven fixes over trunk_train.py.

  1. Policy head gets GLOBAL context: logit = P2 relu(P1 [a_cand ;
     clamp(acc)]) — acc is the value accumulator the engine already
     maintains, so this is engine-free. (Battery: cell-local ptrunk ==
     tables despite far more capacity -> head is context-starved.)
  2. Sibling-contrastive value loss: pairwise logistic on strix-valued
     children of the same parent, weighted by |d oracle| — directly
     optimizes what alpha-beta consumes. (Battery: posval .79->.90 while
     sib_close stayed flat; pointwise Huber never trains contrasts.)
  3. Human data (--human) + sibling children double as tree-interior-like
     pointwise positions.

Run in the hexo venv (GPU), after sibling_extract.py:
    .../hexo-strix/.venv/bin/python trunk_train2.py --human
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import trunk_train
from trunk_train import (CLIP, H, HP, K, N_RAW, build, extract, multi_arange,
                         seg_ce, spearman)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))
from features import MIRROR729


def build_mir11():
    """LUT: swap digits 1<->2 of base-3 11-digit line codes (color mirror)."""
    codes = np.arange(3 ** 11, dtype=np.int64)
    c = codes.copy()
    m = np.zeros_like(codes)
    p = 1
    for _ in range(11):
        d = c % 3
        c //= 3
        m += np.where(d == 1, 2, np.where(d == 2, 1, 0)) * p
        p *= 3
    return m


# ── sibling dataset ─────────────────────────────────────────────────────

def _sib_shard(path):
    d = np.load(path, allow_pickle=True)
    pcells, pmover = d["pcells"], d["pmover"]
    pml, pmc, glens = d["pml"], d["pmc"], d["glens"]
    cq, cr, val, flip = d["cq"], d["cr"], d["val"], d["flip"]
    offs = np.zeros(len(glens) + 1, dtype=np.int64)
    np.cumsum(glens, out=offs[1:])
    T, WI, WC = [], [], []
    smc, sml, stgt, spval, keep_glens = [], [], [], [], []
    for i in range(len(pmover)):
        cells = [tuple(c) for c in pcells[i]]
        mover, ml, mcnt = int(pmover[i]), int(pml[i]), int(pmc[i])
        cm = mover if ml == 2 else 3 - mover
        cml = 1 if ml == 2 else 2
        for j in range(offs[i], offs[i + 1]):
            ccells = cells + [(int(cq[j]), int(cr[j]), mover)]
            trip, wi, wc, _ = extract(ccells, cm, [(0, 0)])
            T.append(trip); WI.append(wi); WC.append(wc)
            smc.append(mcnt + 1); sml.append(cml)
            stgt.append(float(val[j]) * float(flip[j]) * 8.0)  # child POV
            spval.append(float(val[j]))                        # parent POV
        keep_glens.append(int(glens[i]))
    return (np.concatenate(T), np.array([len(x) for x in T], np.int32),
            np.concatenate(WI), np.concatenate(WC),
            np.array([len(x) for x in WI], np.int32),
            np.array(smc, np.int32), np.array(sml, np.int32),
            np.array(stgt, np.float32), np.array(spval, np.float32),
            np.array(keep_glens, np.int32))


def build_siblings(cache, workers):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(SCRIPT_DIR, "sibling_targets",
                                           "*.npz")))
    print(f"building sibling dataset from {len(shards)} shards...",
          flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("trip", "clen", "wi", "wc", "wlen", "mc", "ml",
                           "tgt", "pval", "glen")}
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_sib_shard, shards)):
            for k, v in zip(acc, res):
                acc[k].append(v)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(shards)} shards "
                      f"{time.time()-t0:.0f}s", flush=True)
    ds = {k: np.concatenate(v) for k, v in acc.items()}
    for off_k, len_k in (("coffs", "clen"), ("woffs", "wlen"),
                         ("goffs", "glen")):
        o = np.zeros(len(ds[len_k]) + 1, dtype=np.int64)
        np.cumsum(ds[len_k], out=o[1:])
        ds[off_k] = o
    np.savez(cache, **ds)
    print(f"sibling dataset: {len(ds['glen'])} parents, "
          f"{len(ds['tgt'])} children, {time.time()-t0:.0f}s", flush=True)
    return ds


# ── model ───────────────────────────────────────────────────────────────

class Trunk2(nn.Module):
    def __init__(self):
        super().__init__()
        self.eraw = nn.Embedding(N_RAW, K)
        nn.init.normal_(self.eraw.weight, 0.0, 0.03)
        self.ew = nn.EmbeddingBag(729, K, mode="sum",
                                  include_last_offset=True)
        nn.init.normal_(self.ew.weight, 0.0, 0.03)
        self.w1 = nn.Linear(K + 2, H)
        self.w2 = nn.Linear(H, 1)
        self.p1 = nn.Linear(2 * K + 2, HP)
        self.p2 = nn.Linear(HP, 1)

    def cell_act(self, trip):
        return torch.clamp(self.eraw(trip).sum(dim=1), 0.0, CLIP)

    def accum(self, trip, seg, npos, wi, wc, woff):
        acc = torch.zeros(npos, K, device=trip.device).index_add_(
            0, seg, self.cell_act(trip))
        return acc + self.ew(wi, woff, per_sample_weights=wc)

    def value_from_acc(self, acc, g0, g1):
        x = torch.cat([torch.clamp(acc, 0.0, CLIP),
                       g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)

    def value(self, trip, seg, npos, wi, wc, woff, g0, g1):
        return self.value_from_acc(self.accum(trip, seg, npos, wi, wc, woff),
                                   g0, g1)

    def policy(self, cand_trip, acc, seg_p, g0, g1):
        """g0/g1 (move_count, moves_left scalars) per POSITION, gathered by
        seg_p — the cell's worth depends on stones-in-hand (a lone stone
        can't afford attack when two blocks are needed)."""
        a = self.cell_act(cand_trip)
        ctx = torch.clamp(acc, 0.0, CLIP)[seg_p]
        return self.p2(torch.relu(self.p1(torch.cat(
            [a, ctx, g0[seg_p].unsqueeze(1), g1[seg_p].unsqueeze(1)],
            dim=1)))).squeeze(1)


# ── training ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--sib-parents", type=int, default=192,
                    help="sibling groups per step")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1.0, help="policy KL wt")
    ap.add_argument("--mu", type=float, default=1.0, help="contrast wt")
    ap.add_argument("--eta", type=float, default=0.3,
                    help="child pointwise wt")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--mirror", action="store_true",
                    help="alternate color-mirrored batches (negated target "
                         "and tempo) so the engine can query root-relative")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", default="output_trunk2")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    ds = build(os.path.join(out_dir, "trunk_ds.npz"), args.threads,
               max_shards=args.max_shards, human=args.human)
    sib = build_siblings(os.path.join(out_dir, "sib_ds.npz"), args.threads)

    # joint stream tensors
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

    # sibling stream tensors
    ng = len(sib["glen"])
    scoffs, swoffs, goffs = sib["coffs"], sib["woffs"], sib["goffs"]
    strip = torch.from_numpy(sib["trip"].astype(np.int64))
    swi = torch.from_numpy(sib["wi"].astype(np.int64))
    swc = torch.from_numpy(sib["wc"].astype(np.float32))
    stgt = torch.from_numpy(sib["tgt"])
    spval = sib["pval"]
    sg0 = torch.from_numpy((sib["mc"] * 0.02).astype(np.float32))
    sg1 = torch.from_numpy((sib["ml"] * 0.5).astype(np.float32))
    sflip = torch.from_numpy(
        np.where(np.abs(sib["tgt"] - sib["pval"] * 8.0) < 1e-4, 1.0,
                 -1.0).astype(np.float32))

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = min(8000, n // 10)
    val_ids, train_ids = np.sort(order[:n_val]), order[n_val:]
    gorder = rng.permutation(ng)
    n_gval = min(3000, ng // 10)
    gval_ids, gtrain_ids = np.sort(gorder[:n_gval]), gorder[n_gval:]

    model = Trunk2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    huber = nn.HuberLoss(delta=4.0)

    mir11 = torch.from_numpy(build_mir11())
    mir729 = torch.from_numpy(MIRROR729.astype(np.int64))

    def joint_batch(ids, mirror=False):
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
        bt, bwi, bct = trip[cg], wi[wg], ctrip[pg]
        bg1, by = g1[ids], tgt[ids]
        if mirror:
            bt, bwi, bct = mir11[bt], mir729[bwi], mir11[bct]
            bg1, by = -bg1, -by
        return (bt.to(dev), seg_c, len(ids), bwi.to(dev),
                wc[wg].to(dev), torch.from_numpy(wo).to(dev),
                g0[ids].to(dev), bg1.to(dev),
                bct.to(dev), logit[pg].to(dev), seg_p, by.to(dev))

    def sib_batch(gids, mirror=False):
        """gather children of sibling groups; returns feats + pair arrays."""
        chlens = (goffs[gids + 1] - goffs[gids])
        ch = multi_arange(goffs[gids], chlens)     # child row ids
        clens = (scoffs[ch + 1] - scoffs[ch])
        cg = multi_arange(scoffs[ch], clens)
        seg_c = torch.from_numpy(np.repeat(np.arange(len(ch)), clens)).to(dev)
        wlens = (swoffs[ch + 1] - swoffs[ch])
        wg = multi_arange(swoffs[ch], wlens)
        wo = np.zeros(len(ch) + 1, dtype=np.int64)
        np.cumsum(wlens, out=wo[1:])
        # in-batch pair indices per group
        ii, jj, sg, wt = [], [], [], []
        base = 0
        for L in chlens:
            pv = spval[ch[base:base + L]]
            for a in range(int(L)):
                for b in range(a + 1, int(L)):
                    d = pv[a] - pv[b]
                    if abs(d) < 0.02:
                        continue
                    ii.append(base + a); jj.append(base + b)
                    sg.append(1.0 if d > 0 else -1.0)
                    wt.append(min(abs(d), 0.5) / 0.5)
            base += int(L)
        bt, bwi = strip[cg], swi[wg]
        bg1, bty, bfl = sg1[ch], stgt[ch], sflip[ch]
        if mirror:
            bt, bwi = mir11[bt], mir729[bwi]
            bg1, bty, bfl = -bg1, -bty, -bfl
        return ((bt.to(dev), seg_c, len(ch), bwi.to(dev),
                 swc[wg].to(dev), torch.from_numpy(wo).to(dev),
                 sg0[ch].to(dev), bg1.to(dev)),
                bty.to(dev), bfl.to(dev),
                torch.tensor(ii, device=dev), torch.tensor(jj, device=dev),
                torch.tensor(sg, device=dev), torch.tensor(wt, device=dev))

    def contrast_loss(vpov, ii, jj, sg, wt):
        if len(ii) == 0:
            return torch.tensor(0.0, device=dev)
        d = (vpov[ii] - vpov[jj]) * sg
        return (wt * F.softplus(-d)).sum() / wt.sum()

    steps = (len(train_ids) + args.batch - 1) // args.batch
    print(f"training trunk2 on {len(train_ids)} joint + {len(gtrain_ids)} "
          f"sibling groups, {steps} steps/epoch, dev={dev}", flush=True)
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        rng.shuffle(gtrain_ids)
        t0 = time.time()
        tv = tp = tc = nb = 0
        gpos = 0
        for s in range(0, len(train_ids), args.batch):
            ids = np.sort(train_ids[s:s + args.batch])
            if gpos + args.sib_parents > len(gtrain_ids):
                rng.shuffle(gtrain_ids)
                gpos = 0
            gids = np.sort(gtrain_ids[gpos:gpos + args.sib_parents])
            gpos += args.sib_parents

            mirror = args.mirror and (nb % 2 == 1)
            (bt, seg_c, npos, bwi, bwc, bwo, bg0, bg1, bct, btl, seg_p,
             by) = joint_batch(ids, mirror=mirror)
            sfeat, sby, sbflip, ii, jj, sgn, wt = sib_batch(gids,
                                                            mirror=mirror)

            opt.zero_grad()
            acc = model.accum(bt, seg_c, npos, bwi, bwc, bwo)
            lv = huber(model.value_from_acc(acc, bg0, bg1), by)
            lp = seg_ce(model.policy(bct, acc, seg_p, bg0, bg1), btl, seg_p,
                        npos, dev)
            sv = model.value(*sfeat)
            lc = contrast_loss(sv * sbflip, ii, jj, sgn, wt)
            le = huber(sv, sby)
            (lv + args.lam * lp + args.mu * lc + args.eta * le).backward()
            opt.step()
            tv += float(lv.detach()); tp += float(lp.detach())
            tc += float(lc.detach()); nb += 1
        sched.step()

        with torch.no_grad():
            preds, top1, rank_sum, nv = [], 0, 0, 0
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                (bt, seg_c, npos, bwi, bwc, bwo, bg0, bg1, bct, btl, seg_p,
                 _) = joint_batch(ids)
                acc = model.accum(bt, seg_c, npos, bwi, bwc, bwo)
                preds.append(model.value_from_acc(acc, bg0, bg1)
                             .cpu().numpy())
                sc = model.policy(bct, acc, seg_p, bg0, bg1).cpu().numpy()
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
            # sibling val: pairwise accuracy overall + close
            agree = tot = agree_cl = tot_cl = 0
            for s in range(0, len(gval_ids), 512):
                gids = gval_ids[s:s + 512]
                sfeat, _, sbflip, ii, jj, sgn, wt = sib_batch(gids)
                sv = (model.value(*sfeat) * sbflip)
                d = ((sv[ii] - sv[jj]) * sgn > 0)
                agree += int(d.sum()); tot += len(ii)
                cl = wt < 0.6001      # |d oracle| <= 0.3
                agree_cl += int(d[cl].sum()); tot_cl += int(cl.sum())
        y = tgt[val_ids].numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        print(f"epoch {ep+1}/{args.epochs}: v {tv/nb:.4f} p {tp/nb:.4f} "
              f"c {tc/nb:.4f} | corr {corr:.4f} sp {spearman(pred, y):.4f} "
              f"| top1 {top1/nv:.3f} mrank {rank_sum/nv:.2f} "
              f"| sib {agree/max(tot,1):.4f} close {agree_cl/max(tot_cl,1):.4f}"
              f" ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "K": K, "H": H, "HP": HP, "CLIP": CLIP, "arch": "v2"},
               os.path.join(out_dir, "trunk.pt"))
    print(f"saved {out_dir}/trunk.pt")


if __name__ == "__main__":
    main()

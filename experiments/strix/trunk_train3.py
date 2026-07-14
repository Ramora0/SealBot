"""Trunk v3 "LineNL": the unit of nonlinearity is the LINE, not the cell.

No 3^11 cell table. Only the 729-entry 6-window embedding EW remains.
Every active window instance (nonzero pattern, any anchor) is summed
into its LINE's accumulator (line = direction d plus invariant key
q*dr - r*dq); one clamp per line; the position vector A is the sum of
clamped line activations (lines with no active windows contribute
exactly 0 = clamp(0)).

  line l:  a_l = clamp(sum_{windows on l} EW[pattern], 0, CLIP)
  value:   A = sum_l a_l ; v = W2 relu(W1 [A; g0; g1])
  policy:  logit(c) = P2 relu(P1 [a_l0(c); a_l1(c); a_l2(c); A; g0; g1])
           (a candidate line with no active windows -> zero vector)

Losses/data/metrics mirror trunk_train2.py: Huber(value) + lam*listwise
KL(policy) + mu*sibling-contrast + eta*child pointwise; --mirror
alternates color-mirrored batches (MIRROR729 on wi, negated target and
tempo; line structure is color-independent). Sources: policy_targets x
gen0 pkls, --human (human_targets), --dagger (dagger_targets, same npz
format), --vcf-labels [DIR] soft-floors proven wins at 6.5 (sidecar
flags g_/h_/d_ per source; a missing flag file just skips the floor).

Run in the hexo venv (GPU), after sibling_extract.py:
    .../hexo-strix/.venv/bin/python trunk_train3.py --mirror --human \
        --dagger --vcf-labels
Self-test (CPU, seconds):
    .../SealBot/.venv/bin/python trunk_train3.py --smoke
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
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from trunk_train import (CLIP, DATA, H, HP, K, TARGETS, _key, extract,
                         multi_arange, seg_ce, spearman)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))
from features import DIRS, MIRROR729, PAD, POW3_6


# ── extraction ──────────────────────────────────────────────────────────

def extract3(cells, mover, cand):
    """Line-grouped window instances of one position.

    Returns (wi, wline, nlines, cline):
      wi     int16 [nw]      window pattern codes (1..728), all active
                             window instances (same multiset as
                             trunk_train.extract's w_counts bag)
      wline  int16 [nw]      line-local index per instance (lines in
                             first-encounter order, dir-major)
      nlines int32           lines with >=1 active window
      cline  int16 [nc, 3]   candidate's line per direction, -1 if that
                             line has no active windows
    """
    qs = np.array([c[0] for c in cells]); rs = np.array([c[1] for c in cells])
    ps = np.array([c[2] for c in cells])
    q0, r0 = qs.min() - PAD, rs.min() - PAD
    Hh = int(qs.max() - q0 + PAD + 1); Ww = int(rs.max() - r0 + PAD + 1)
    grid = np.zeros((Hh, Ww), dtype=np.int64)
    grid[qs - q0, rs - r0] = np.where(ps == mover, 1, 2)

    cg = (np.array([(q - q0, r - r0) for q, r in cand], dtype=np.int64)
          if len(cand) else np.zeros((0, 2), dtype=np.int64))
    WI, WL = [], []
    cline = np.full((len(cand), 3), -1, dtype=np.int16)
    base = 0
    for di, (dq, dr) in enumerate(DIRS):
        pat = np.zeros((Hh, Ww), dtype=np.int64)
        for j in range(6):
            sq, sr = j * dq, j * dr
            src = np.zeros((Hh, Ww), dtype=np.int64)
            src[max(0, -sq):min(Hh, Hh - sq), max(0, -sr):min(Ww, Ww - sr)] = \
                grid[max(0, -sq) + sq:min(Hh, Hh - sq) + sq,
                     max(0, -sr) + sr:min(Ww, Ww - sr) + sr]
            pat += src * POW3_6[j]
        wq, wr = np.nonzero(pat)
        if len(wq) == 0:
            continue
        keys = wq * dr - wr * dq            # constant along the line
        uk, fidx, inv = np.unique(keys, return_index=True,
                                  return_inverse=True)
        rank = np.empty(len(uk), dtype=np.int64)
        rank[np.argsort(fidx)] = np.arange(len(uk))
        WI.append(pat[wq, wr])
        WL.append(rank[inv] + base)
        if len(cand):
            ck = cg[:, 0] * dr - cg[:, 1] * dq
            pos = np.searchsorted(uk, ck)
            posc = np.minimum(pos, len(uk) - 1)
            ok = (pos < len(uk)) & (uk[posc] == ck)
            cline[:, di] = np.where(ok, rank[posc] + base, -1)
        base += len(uk)
    wi = np.concatenate(WI).astype(np.int16)
    wline = np.concatenate(WL).astype(np.int16)
    return wi, wline, np.int32(base), cline


# ── joint dataset ───────────────────────────────────────────────────────

def _shard3(paths):
    npz_path, pkl_path, flag_path = paths
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
    WI, WL, NL, CL, LG = [], [], [], [], []
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
        # Proven forced win for the mover: soft floor (v1.5 recipe).
        if vcf_flags is not None and vcf_flags[i] and t < 6.5:
            t = 6.5
        sl = slice(offs[i], offs[i + 1])
        cand = list(zip(cq[sl].tolist(), cr[sl].tolist()))
        wi, wline, nl, cl = extract3([tuple(c) for c in cells], int(mover),
                                     cand)
        WI.append(wi); WL.append(wline); NL.append(int(nl)); CL.append(cl)
        LG.append(lg[sl])
        tgt.append(t); mcs.append(int(mc)); mls.append(int(ml))
    return (np.concatenate(WI), np.concatenate(WL),
            np.array([len(x) for x in WI], np.int32),
            np.array(NL, np.int32), np.concatenate(CL),
            np.concatenate(LG).astype(np.float32),
            np.array([len(x) for x in CL], np.int32),
            np.array(tgt, np.float32), np.array(mcs, np.int32),
            np.array(mls, np.int32), miss)


def build3(cache, workers, max_shards=None, human=False, dagger=False,
           vcf_dir=None):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp

    def _flag(npz, pref):
        if vcf_dir is None:
            return None
        return os.path.join(vcf_dir, pref +
                            os.path.basename(npz).replace(".npz", ".npy"))

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
    print(f"building line dataset from {len(pairs)} shards...", flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("wi", "wline", "wlen", "nl", "cline", "logit",
                           "plen", "tgt", "mc", "ml")}
    miss = 0
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_shard3, pairs)):
            for k, v in zip(acc, res[:-1]):
                acc[k].append(v)
            miss += res[-1]
            if (i + 1) % 10 == 0:
                n = sum(len(x) for x in acc["tgt"])
                print(f"  {i+1}/{len(pairs)} shards, {n} pos, {miss} unjoined,"
                      f" {time.time()-t0:.0f}s", flush=True)
    ds = {k: np.concatenate(v) for k, v in acc.items()}
    for off_k, len_k in (("woffs", "wlen"), ("poffs", "plen")):
        o = np.zeros(len(ds[len_k]) + 1, dtype=np.int64)
        np.cumsum(ds[len_k], out=o[1:])
        ds[off_k] = o
    np.savez(cache, **ds)
    print(f"line dataset: {len(ds['tgt'])} pos ({ds['wlen'].mean():.0f} win, "
          f"{ds['nl'].mean():.0f} lines, {ds['plen'].mean():.0f} cands/pos), "
          f"{miss} unjoined, {time.time()-t0:.0f}s", flush=True)
    return ds


# ── sibling dataset ─────────────────────────────────────────────────────

def _sib_shard3(path):
    d = np.load(path, allow_pickle=True)
    pcells, pmover = d["pcells"], d["pmover"]
    pml, pmc, glens = d["pml"], d["pmc"], d["glens"]
    cq, cr, val, flip = d["cq"], d["cr"], d["val"], d["flip"]
    offs = np.zeros(len(glens) + 1, dtype=np.int64)
    np.cumsum(glens, out=offs[1:])
    WI, WL, NL = [], [], []
    smc, sml, stgt, spval, keep_glens = [], [], [], [], []
    for i in range(len(pmover)):
        cells = [tuple(c) for c in pcells[i]]
        mover, ml, mcnt = int(pmover[i]), int(pml[i]), int(pmc[i])
        cm = mover if ml == 2 else 3 - mover
        cml = 1 if ml == 2 else 2
        for j in range(offs[i], offs[i + 1]):
            ccells = cells + [(int(cq[j]), int(cr[j]), mover)]
            wi, wline, nl, _ = extract3(ccells, cm, [])
            WI.append(wi); WL.append(wline); NL.append(int(nl))
            smc.append(mcnt + 1); sml.append(cml)
            stgt.append(float(val[j]) * float(flip[j]) * 8.0)  # child POV
            spval.append(float(val[j]))                        # parent POV
        keep_glens.append(int(glens[i]))
    return (np.concatenate(WI), np.concatenate(WL),
            np.array([len(x) for x in WI], np.int32),
            np.array(NL, np.int32),
            np.array(smc, np.int32), np.array(sml, np.int32),
            np.array(stgt, np.float32), np.array(spval, np.float32),
            np.array(keep_glens, np.int32))


def build_siblings3(cache, workers):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(SCRIPT_DIR, "sibling_targets",
                                           "*.npz")))
    print(f"building sibling dataset from {len(shards)} shards...",
          flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("wi", "wline", "wlen", "nl", "mc", "ml", "tgt",
                           "pval", "glen")}
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_sib_shard3, shards)):
            for k, v in zip(acc, res):
                acc[k].append(v)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(shards)} shards "
                      f"{time.time()-t0:.0f}s", flush=True)
    ds = {k: np.concatenate(v) for k, v in acc.items()}
    for off_k, len_k in (("woffs", "wlen"), ("goffs", "glen")):
        o = np.zeros(len(ds[len_k]) + 1, dtype=np.int64)
        np.cumsum(ds[len_k], out=o[1:])
        ds[off_k] = o
    np.savez(cache, **ds)
    print(f"sibling dataset: {len(ds['glen'])} parents, "
          f"{len(ds['tgt'])} children, {time.time()-t0:.0f}s", flush=True)
    return ds


# ── model ───────────────────────────────────────────────────────────────

class Trunk3(nn.Module):
    def __init__(self, k=K, h=H, hp=HP):
        super().__init__()
        self.k, self.h, self.hp = k, h, hp
        self.ew = nn.Embedding(729, k)
        nn.init.normal_(self.ew.weight, 0.0, 0.03)
        self.w1 = nn.Linear(k + 2, h)
        self.w2 = nn.Linear(h, 1)
        self.p1 = nn.Linear(4 * k + 2, hp)
        self.p2 = nn.Linear(hp, 1)

    def line_acts(self, wi, wl, nl_tot):
        """clamped per-line accumulators; wl = global line idx per window."""
        acc = torch.zeros(nl_tot, self.k, device=wi.device).index_add_(
            0, wl, self.ew(wi))
        return torch.clamp(acc, 0.0, CLIP)

    def accum(self, a, lseg, npos):
        return torch.zeros(npos, self.k, device=a.device).index_add_(
            0, lseg, a)

    def value_from_acc(self, A, g0, g1):
        x = torch.cat([A, g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)

    def value(self, wi, wl, nl_tot, lseg, npos, g0, g1):
        a = self.line_acts(wi, wl, nl_tot)
        return self.value_from_acc(self.accum(a, lseg, npos), g0, g1)

    def policy(self, a, A, cg, seg_p, g0, g1):
        """cg [nc,3] global line rows into a (pad row nl_tot = zeros);
        g0/g1 per POSITION, gathered by seg_p."""
        a_pad = torch.cat([a, a.new_zeros(1, self.k)], dim=0)
        x = torch.cat([a_pad[cg].reshape(cg.shape[0], 3 * self.k),
                       A[seg_p], g0[seg_p].unsqueeze(1),
                       g1[seg_p].unsqueeze(1)], dim=1)
        return self.p2(torch.relu(self.p1(x))).squeeze(1)


# ── smoke test ──────────────────────────────────────────────────────────

def _smoke_batch(model, exs, mir=False):
    """exs: [(wi, wline, nl, cline, mc, ml)] -> (value, policy)."""
    lut = MIRROR729.astype(np.int64)
    NL = sum(int(e[2]) for e in exs)
    WIs, WLs, CGs, segs, nls, g0l, g1l = [], [], [], [], [], [], []
    lb = 0
    for i, (wi, wl, nl, cl, mc, ml) in enumerate(exs):
        w = wi.astype(np.int64)
        WIs.append(lut[w] if mir else w)
        WLs.append(wl.astype(np.int64) + lb)
        CGs.append(np.where(cl >= 0, cl.astype(np.int64) + lb, NL))
        segs.append(np.full(len(cl), i, dtype=np.int64))
        nls.append(int(nl)); g0l.append(mc * 0.02); g1l.append(ml * 0.5)
        lb += int(nl)
    bwi = torch.from_numpy(np.concatenate(WIs))
    bwl = torch.from_numpy(np.concatenate(WLs))
    lseg = torch.from_numpy(np.repeat(np.arange(len(exs)), nls))
    cg = torch.from_numpy(np.concatenate(CGs))
    seg_p = torch.from_numpy(np.concatenate(segs))
    g0, g1 = torch.tensor(g0l), torch.tensor(g1l)
    a = model.line_acts(bwi, bwl, NL)
    A = model.accum(a, lseg, len(exs))
    return (model.value_from_acc(A, g0, g1),
            model.policy(a, A, cg, seg_p, g0, g1))


def smoke():
    torch.manual_seed(0)
    tests = [
        ([(0, 0, 1)], 1, [(1, 1), (0, 3)]),
        ([(0, 0, 1), (1, 0, 2), (0, 1, 1), (2, 2, 1), (1, 2, 2)], 1,
         [(0, 2), (3, 0), (-1, 1), (0, 5)]),
        ([(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 2), (-1, 1, 2),
          (0, 2, 2), (1, 1, 1), (2, 1, 2)], 2,
         [(4, 0), (-1, 0), (1, 2), (5, 5)]),
    ]
    exs, exs_sw = [], []
    for pi, (cells, mover, cand) in enumerate(tests):
        wi, wl, nl, cl = extract3(cells, mover, cand)
        _, wi1, wc1, _ = extract(cells, mover, cand)
        bag = np.bincount(wi, minlength=729)
        ref = np.zeros(729, dtype=np.int64)
        ref[wi1] = wc1
        assert (bag == ref).all(), f"pos {pi}: window bag mismatch"
        assert wi.min() >= 1 and int(wl.max()) < int(nl)
        assert (cl < int(nl)).all() and (cl >= -1).all()
        # color-swapped board, same mover: line structure identical,
        # patterns digit-swapped
        sw = [(q, r, 3 - p) for q, r, p in cells]
        wis, wls, nls, cls = extract3(sw, mover, cand)
        assert int(nls) == int(nl) and np.array_equal(wls, wl)
        assert np.array_equal(cls, cl)
        assert np.array_equal(MIRROR729[wis.astype(np.int64)],
                              wi.astype(np.int64))
        mc, ml = 5 + pi, 1 + pi % 2
        exs.append((wi, wl, nl, cl, mc, ml))
        exs_sw.append((wis, wls, nls, cls, mc, ml))
    w0, l0, n0, c0, _, _ = exs[0]
    assert int(n0) == 3 and len(w0) == 18, "single stone: 3 lines, 18 wins"
    assert (c0[:, :] == -1).any(), "expected some off-line candidates"

    model = Trunk3()
    v, p = _smoke_batch(model, exs)
    assert v.shape == (len(exs),)
    assert p.shape == (sum(len(e[3]) for e in exs),)
    assert torch.isfinite(v).all() and torch.isfinite(p).all()
    # mirrored forward of the swapped boards == forward of the originals
    vm, pm = _smoke_batch(model, exs_sw, mir=True)
    dv = float((v - vm).abs().max().detach())
    dp = float((p - pm).abs().max().detach())
    assert dv == 0.0 and dp == 0.0, f"mirror identity: dv={dv} dp={dp}"
    print(f"smoke: {len(exs)} positions ok | bags match extract() | "
          f"value {v.detach().numpy().round(4)} | policy n={len(p)} finite | "
          f"mirror identity exact (dv={dv}, dp={dp})")
    print("SMOKE OK")


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
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--H", type=int, default=H)
    ap.add_argument("--HP", type=int, default=HP)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--dagger", action="store_true",
                    help="include dagger_targets/ shards (inline vals)")
    ap.add_argument("--vcf-labels", nargs="?", const="vcf_targets",
                    default=None, metavar="DIR",
                    help="soft-floor value targets on proven forced wins; "
                         "optional flags dir (default vcf_targets)")
    ap.add_argument("--mirror", action="store_true",
                    help="alternate color-mirrored batches (negated target "
                         "and tempo) so the engine can query root-relative")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", default="output_line1")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU self-test, no data needed")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    vcf_dir = (os.path.join(SCRIPT_DIR, args.vcf_labels)
               if args.vcf_labels else None)
    ds = build3(os.path.join(out_dir, "cache_trunk3.npz"), args.threads,
                max_shards=args.max_shards, human=args.human,
                dagger=args.dagger, vcf_dir=vcf_dir)
    sib = build_siblings3(os.path.join(out_dir, "sib_ds3.npz"), args.threads)

    # joint stream tensors
    n = len(ds["tgt"])
    woffs, poffs = ds["woffs"], ds["poffs"]
    nlarr = ds["nl"].astype(np.int64)
    wi = torch.from_numpy(ds["wi"].astype(np.int64))
    wline = ds["wline"].astype(np.int64)
    cline = ds["cline"].astype(np.int64)
    logit = torch.from_numpy(ds["logit"])
    tgt = torch.from_numpy(ds["tgt"])
    g0 = torch.from_numpy((ds["mc"] * 0.02).astype(np.float32))
    g1 = torch.from_numpy((ds["ml"] * 0.5).astype(np.float32))

    # sibling stream tensors
    ng = len(sib["glen"])
    swoffs, goffs = sib["woffs"], sib["goffs"]
    snl = sib["nl"].astype(np.int64)
    swi = torch.from_numpy(sib["wi"].astype(np.int64))
    swline = sib["wline"].astype(np.int64)
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

    model = Trunk3(k=args.K, h=args.H, hp=args.HP).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    huber = nn.HuberLoss(delta=4.0)

    mir729 = torch.from_numpy(MIRROR729.astype(np.int64))

    def line_gather(ids, offs, nl_all, wln):
        """global line index per window + line->pos seg for a batch."""
        wlens = offs[ids + 1] - offs[ids]
        wg = multi_arange(offs[ids], wlens)
        nl = nl_all[ids]
        lb = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(nl, out=lb[1:])
        wl = torch.from_numpy(wln[wg] + np.repeat(lb[:-1], wlens))
        lseg = torch.from_numpy(np.repeat(np.arange(len(ids)), nl)).to(dev)
        return wg, wl, int(lb[-1]), lseg, lb

    def joint_batch(ids, mirror=False):
        wg, wl, NL, lseg, lb = line_gather(ids, woffs, nlarr, wline)
        plens = poffs[ids + 1] - poffs[ids]
        pg = multi_arange(poffs[ids], plens)
        segp = np.repeat(np.arange(len(ids)), plens)
        seg_p = torch.from_numpy(segp).to(dev)
        cl = cline[pg]
        cg = torch.from_numpy(
            np.where(cl >= 0, cl + lb[:-1][segp][:, None], NL))
        bwi = wi[wg]
        bg1, by = g1[ids], tgt[ids]
        if mirror:
            bwi = mir729[bwi]
            bg1, by = -bg1, -by
        return (bwi.to(dev), wl.to(dev), NL, lseg, len(ids),
                g0[ids].to(dev), bg1.to(dev),
                cg.to(dev), logit[pg].to(dev), seg_p, by.to(dev))

    def sib_batch(gids, mirror=False):
        """gather children of sibling groups; returns feats + pair arrays."""
        chlens = goffs[gids + 1] - goffs[gids]
        ch = multi_arange(goffs[gids], chlens)     # child row ids
        wg, wl, NL, lseg, _ = line_gather(ch, swoffs, snl, swline)
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
        bwi = swi[wg]
        bg1, bty, bfl = sg1[ch], stgt[ch], sflip[ch]
        if mirror:
            bwi = mir729[bwi]
            bg1, bty, bfl = -bg1, -bty, -bfl
        return ((bwi.to(dev), wl.to(dev), NL, lseg, len(ch),
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
    print(f"training trunk3 (LineNL) on {len(train_ids)} joint + "
          f"{len(gtrain_ids)} sibling groups, {steps} steps/epoch, dev={dev}",
          flush=True)
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
            (bwi, bwl, NL, lseg, npos, bg0, bg1, cg, btl, seg_p,
             by) = joint_batch(ids, mirror=mirror)
            sfeat, sby, sbflip, ii, jj, sgn, wt = sib_batch(gids,
                                                            mirror=mirror)

            opt.zero_grad()
            a = model.line_acts(bwi, bwl, NL)
            A = model.accum(a, lseg, npos)
            lv = huber(model.value_from_acc(A, bg0, bg1), by)
            lp = seg_ce(model.policy(a, A, cg, seg_p, bg0, bg1), btl, seg_p,
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
                (bwi, bwl, NL, lseg, npos, bg0, bg1, cg, btl, seg_p,
                 _) = joint_batch(ids)
                a = model.line_acts(bwi, bwl, NL)
                A = model.accum(a, lseg, npos)
                preds.append(model.value_from_acc(A, bg0, bg1)
                             .cpu().numpy())
                sc = model.policy(a, A, cg, seg_p, bg0, bg1).cpu().numpy()
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
                "K": args.K, "H": args.H, "HP": args.HP, "CLIP": CLIP,
                "arch": "line1"},
               os.path.join(out_dir, "trunk.pt"))
    print(f"saved {out_dir}/trunk.pt")


if __name__ == "__main__":
    main()

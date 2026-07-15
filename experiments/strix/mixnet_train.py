"""Mixnet repro: Rapfi's network (arXiv 2503.13178) adapted to hex Connect6.

See MIXNET_DESIGN.md for the adaptation table. Shape of the model:

  mapping:  code(3^11 line pattern) -> 5-layer 1D conv net (M ch, skips)
            -> center column -> C ch, ZERO-ANCHORED (map(0) == 0)
  cell:     s_c = sum_axes map(lp_axis(c));  a_c = clamp(s_c, 0, CLIP)
  F':       [clamp(DW_hex7(a[:, :C/2]), 0, CLIP) ; a[:, C/2:]]
            (depthwise conv over center+6 hex neighbors, no bias)
  value:    A = sum_cells F' ; z = [A; g0; g1]
            star (W_a z)*(W_b z) -> V -> relu -> V -> relu -> 3 (WDL)
  policy:   gmean = A/ncells ; MLP([gmean; g0; g1]) -> {W in R^{PxP}, b}
            h = relu(W F'_cand[:P] + b) ; logit = w2 . h     (dyn conv)

Losses (Rapfi): CE on both heads, 75% teacher-soft (strix forward:
value tanh -> WDL probs, policy logits -> softmax over D2 candidates)
+ 25% true labels (game outcome / played move) where available.

Run (hexo venv, GPU):
    .../hexo-strix/.venv/Scripts/python.exe mixnet_train.py --device cuda
Smoke (CPU, seconds, no data):
    .../python.exe mixnet_train.py --smoke
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

from trunk_train import DATA, TARGETS, _key, multi_arange, spearman
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))
from features import DIRS, PAD, POW3_11

N_RAW = 3 ** 11
CLIP = 8.0
# center + 6 hex neighbors (axial); order is the DW kernel order
HEX7 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _mir3_11():
    """Color-mirror permutation of raw 3^11 codes (digits 1<->2)."""
    codes = np.arange(N_RAW, dtype=np.int64)
    out = np.zeros_like(codes)
    for i in range(11):
        d = (codes // 3 ** i) % 3
        out += np.where(d == 0, 0, 3 - d) * 3 ** i
    return out


MIR3_11 = _mir3_11()


# ── extraction ──────────────────────────────────────────────────────────

def extract_mix(cells, mover, cand):
    """Per-cell 3-axis raw 3^11 codes + coords for active-or-candidate
    cells of one position.

    Returns (codes [n,3] i32, coords [n,2] i16, cand_idx [nc] i32).
    coords are grid-relative (translation is irrelevant downstream).
    """
    qs = np.array([c[0] for c in cells]); rs = np.array([c[1] for c in cells])
    ps = np.array([c[2] for c in cells])
    aq = np.concatenate([qs, [q for q, r in cand]]) if cand else qs
    ar = np.concatenate([rs, [r for q, r in cand]]) if cand else rs
    q0, r0 = aq.min() - PAD, ar.min() - PAD
    Hh = int(aq.max() - q0 + PAD + 1); Ww = int(ar.max() - r0 + PAD + 1)
    grid = np.zeros((Hh, Ww), dtype=np.int64)
    grid[qs - q0, rs - r0] = np.where(ps == mover, 1, 2)

    lps = []
    for dq, dr in DIRS:
        lp = np.zeros((Hh, Ww), dtype=np.int64)
        for u in range(-5, 6):
            sq, sr = u * dq, u * dr
            src = np.zeros((Hh, Ww), dtype=np.int64)
            src[max(0, -sq):min(Hh, Hh - sq), max(0, -sr):min(Ww, Ww - sr)] = \
                grid[max(0, -sq) + sq:min(Hh, Hh - sq) + sq,
                     max(0, -sr) + sr:min(Ww, Ww - sr) + sr]
            lp += src * POW3_11[5 + u]
        lps.append(lp)

    keep = (lps[0] | lps[1] | lps[2]) != 0
    cg = (np.array([(q - q0, r - r0) for q, r in cand], dtype=np.int64)
          if len(cand) else np.zeros((0, 2), dtype=np.int64))
    keep[cg[:, 0], cg[:, 1]] = True
    rq, rr = np.nonzero(keep)
    rowid = np.full((Hh, Ww), -1, dtype=np.int64)
    rowid[rq, rr] = np.arange(len(rq))
    codes = np.stack([lp[rq, rr] for lp in lps], axis=1).astype(np.int32)
    coords = np.stack([rq, rr], axis=1).astype(np.int16)
    cand_idx = rowid[cg[:, 0], cg[:, 1]].astype(np.int32)
    assert (cand_idx >= 0).all()
    return codes, coords, cand_idx


# ── dataset ─────────────────────────────────────────────────────────────

def _shard_mix(paths):
    npz_path, pkl_path, sidecar_path = paths
    vals = {}
    if pkl_path is not None:
        for g in pickle.load(open(pkl_path, "rb")):
            for p in g["positions"]:
                vals[_key(p["cells"], p["mover"], p["moves_left"],
                          p["move_count"])] = p["score"] / 1000.0
    sidecar = (np.load(sidecar_path) if sidecar_path is not None else None)
    d = np.load(npz_path, allow_pickle=True)
    inline_val = d["val"] if "val" in d.files else None
    inline_played = d["played"] if "played" in d.files else None
    inline_outc = d["outcome"] if "outcome" in d.files else None
    lens, cq, cr, lg = d["lens"], d["cell_q"], d["cell_r"], d["logit"]
    metas = d["meta"]
    offs = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    CO, XY, CI, LG = [], [], [], []
    tgt, mcs, mls, plyd, outc = [], [], [], [], []
    miss = 0
    for i, (cells, mover, ml, mc) in enumerate(metas):
        if inline_val is not None:
            t = float(inline_val[i]) * 8.0
        elif sidecar is not None:
            t = (None if np.isnan(sidecar[i]) else float(sidecar[i]) * 8.0)
        else:
            t = vals.get(_key(cells, mover, ml, mc))
        if t is None:
            miss += 1
            continue
        sl = slice(offs[i], offs[i + 1])
        cand = list(zip(cq[sl].tolist(), cr[sl].tolist()))
        codes, coords, ci = extract_mix([tuple(c) for c in cells],
                                        int(mover), cand)
        CO.append(codes); XY.append(coords); CI.append(ci)
        LG.append(lg[sl])
        tgt.append(t); mcs.append(int(mc)); mls.append(int(ml))
        plyd.append(-1 if inline_played is None else int(inline_played[i]))
        outc.append(np.nan if inline_outc is None else float(inline_outc[i]))
    return (np.concatenate(CO), np.concatenate(XY),
            np.array([len(x) for x in CO], np.int32),
            np.concatenate(CI),
            np.concatenate(LG).astype(np.float32),
            np.array([len(x) for x in CI], np.int32),
            np.array(tgt, np.float32), np.array(mcs, np.int32),
            np.array(mls, np.int32),
            np.array(plyd, np.int32), np.array(outc, np.float32), miss)


VAL_SIDECARS = os.path.join(SCRIPT_DIR, "val_targets")
BENCH_TARGETS = os.path.join(SCRIPT_DIR, "bench_targets")


def build_mix(cache, workers, max_shards=None):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    pairs = []
    for npz in sorted(glob.glob(os.path.join(TARGETS, "*.npz"))):
        base = os.path.basename(npz)
        pkl = os.path.join(DATA, base.replace(".npz", ".pkl"))
        sc = os.path.join(VAL_SIDECARS, base.replace(".npz", ".npy"))
        if os.path.exists(pkl):
            pairs.append((npz, pkl, None))
        elif os.path.exists(sc):
            pairs.append((npz, None, sc))
    if max_shards:
        pairs = pairs[:max_shards]
    for npz in sorted(glob.glob(os.path.join(BENCH_TARGETS, "*.npz"))):
        pairs.append((npz, None, None))          # inline val/played/outcome
    print(f"building mixnet dataset from {len(pairs)} shards...", flush=True)
    t0 = time.time()
    acc = {k: [] for k in ("codes", "xy", "clen", "cand", "logit", "plen",
                           "tgt", "mc", "ml", "played", "outc")}
    miss = 0
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_shard_mix, pairs)):
            for k, v in zip(acc, res[:-1]):
                acc[k].append(v)
            miss += res[-1]
            if (i + 1) % 10 == 0:
                n = sum(len(x) for x in acc["tgt"])
                print(f"  {i+1}/{len(pairs)} shards, {n} pos, {miss} unjoined,"
                      f" {time.time()-t0:.0f}s", flush=True)
    ds = {k: np.concatenate(v) for k, v in acc.items()}
    for off_k, len_k in (("coffs", "clen"), ("poffs", "plen")):
        o = np.zeros(len(ds[len_k]) + 1, dtype=np.int64)
        np.cumsum(ds[len_k], out=o[1:])
        ds[off_k] = o
    np.savez(cache, **ds)
    print(f"mixnet dataset: {len(ds['tgt'])} pos "
          f"({ds['clen'].mean():.0f} cells, {ds['plen'].mean():.0f} cands), "
          f"{miss} unjoined, {time.time()-t0:.0f}s", flush=True)
    return ds


# ── batch geometry (dilation + hex neighbors), pure numpy ───────────────

OFF = 512          # coord shift; |relative coords| stay far below this
FIELD = 11         # bits per coordinate


def batch_geometry(xy, seg, ncand_rows):
    """From stored cells (coords + position seg), build the conv universe
    U = cells âˆª hex-dilation(cells) and the [|U|, 7] neighbor index table.

    Returns (src [n_stored] rows into U, nbr [|U|,7] with -1 pad,
    useg [|U|] position ids). ncand_rows unused here (cands are stored).
    """
    q = xy[:, 0].astype(np.int64) + OFF
    r = xy[:, 1].astype(np.int64) + OFF
    base = (seg.astype(np.int64) << (2 * FIELD)) | (q << FIELD) | r
    parts = [base]
    for dq, dr in HEX7[1:]:
        parts.append(base + (dq << FIELD) + dr)
    ukeys = np.unique(np.concatenate(parts))
    src = np.searchsorted(ukeys, base)            # stored -> U row
    nbr = np.empty((len(ukeys), 7), dtype=np.int64)
    order = np.argsort(base, kind="stable")
    stored_sorted = base[order]
    for k, (dq, dr) in enumerate(HEX7):
        tkeys = ukeys + (dq << FIELD) + dr
        pos = np.searchsorted(stored_sorted, tkeys)
        posc = np.minimum(pos, len(stored_sorted) - 1)
        hit = stored_sorted[posc] == tkeys
        nbr[:, k] = np.where(hit, order[posc], -1)
    useg = (ukeys >> (2 * FIELD)).astype(np.int64)
    return src, nbr, useg


# ── model ───────────────────────────────────────────────────────────────

class Mapping(nn.Module):
    """Line-pattern mapping net: one-hot [n,3,11] -> C channels (center).
    5 receptive-field-growing convs + 1x1s + skips, Rapfi Dir-Conv style.
    Bakeable to a 3^11 codebook; zero-anchored by the caller."""

    def __init__(self, m=64, c=32):
        super().__init__()
        self.inp = nn.Conv1d(3, m, 3, padding=1)
        self.conv = nn.ModuleList(nn.Conv1d(m, m, 3, padding=1)
                                  for _ in range(4))
        self.mix = nn.ModuleList(nn.Conv1d(m, m, 1) for _ in range(4))
        self.head = nn.Linear(m, c)

    def forward(self, onehot):
        h = torch.relu(self.inp(onehot))
        for cv, mx in zip(self.conv, self.mix):
            h = h + mx(torch.relu(cv(h)))
        return self.head(h[:, :, 5])              # center column


def decode_onehot(codes, device):
    """int64 raw codes [n] -> one-hot [n, 3, 11] (digit 0/1/2 per cell)."""
    d = codes.unsqueeze(1) // torch.tensor(
        [3 ** i for i in range(11)], device=device, dtype=torch.int64) % 3
    return F.one_hot(d, 3).permute(0, 2, 1).float()


class Mixnet(nn.Module):
    def __init__(self, m=64, c=32, p=16, v=32):
        super().__init__()
        assert c % 2 == 0 and p <= c
        self.c, self.p = c, p
        self.mapping = Mapping(m, c)
        self.dw = nn.Parameter(torch.zeros(7, c // 2))
        nn.init.normal_(self.dw, 0.0, 0.2)
        with torch.no_grad():
            self.dw[0] += 1.0                     # near-identity start
        zin = c + 2
        self.star_a = nn.Linear(zin, v)
        self.star_b = nn.Linear(zin, v)
        self.v1 = nn.Linear(v, v)
        self.v2 = nn.Linear(v, 3)
        self.pg1 = nn.Linear(zin, 64)
        self.pg2 = nn.Linear(64, p * p + p)
        with torch.no_grad():
            self.pg2.weight *= 0.1
        self.pout = nn.Linear(p, 1)

    def cell_feats(self, codes, device):
        """codes [n,3] (stored cells) -> clamped pre-conv features a [n, C]."""
        uc, inv = torch.unique(codes.reshape(-1), return_inverse=True)
        if uc[0] != 0:                             # ensure zero anchor row
            uc = torch.cat([uc.new_zeros(1), uc])
            inv = inv + 1
        e = self.mapping(decode_onehot(uc, device))
        e = e - e[0:1]                             # zero-anchor: map(0) == 0
        s = e[inv].reshape(-1, 3, self.c).sum(dim=1)
        return torch.clamp(s, 0.0, CLIP)

    def conv_pool(self, a_stored, src, nbr, useg, npos, act):
        """a_stored [n_stored, C], act [n_stored] bool (cell has any nonzero
        line pattern) -> (F' [nU, C], A [npos, C], count).

        count matches the engine's |U| = #{cells with an active cell in
        their 7-neighborhood}; rows that exist only as candidate-readout
        halos carry F' == 0 and must NOT be counted (the engine cannot
        know the candidate list at eval time)."""
        nU = nbr.shape[0]
        c2 = self.c // 2
        a_pad = torch.cat([a_stored, a_stored.new_zeros(1, self.c)], dim=0)
        idx = torch.where(nbr >= 0, nbr, a_stored.shape[0])
        gath = a_pad[idx.reshape(-1), :c2].reshape(nU, 7, c2)
        dwout = torch.clamp((gath * self.dw.unsqueeze(0)).sum(dim=1),
                            0.0, CLIP)
        rest = a_pad[idx[:, 0], c2:]               # stored value or 0
        fp = torch.cat([dwout, rest], dim=1)
        A = torch.zeros(npos, self.c, device=fp.device).index_add_(
            0, useg, fp)
        act_pad = torch.cat([act, act.new_zeros(1)], dim=0)
        counted = act_pad[idx.reshape(-1)].reshape(nU, 7).any(dim=1)
        cnt = torch.zeros(npos, device=fp.device).index_add_(
            0, useg, counted.float())
        return fp, A, cnt

    def value(self, A, g0, g1):
        z = torch.cat([A, g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        s = self.star_a(z) * self.star_b(z)
        return self.v2(torch.relu(self.v1(torch.relu(s))))

    def policy(self, fp, A, cnt, cand_u, seg_p, g0, g1):
        gm = A / cnt.unsqueeze(1).clamp(min=1.0)
        z = torch.cat([gm, g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        wb = self.pg2(torch.relu(self.pg1(z)))
        W = wb[:, :self.p * self.p].reshape(-1, self.p, self.p)
        b = wb[:, self.p * self.p:]
        x = fp[cand_u, :self.p]
        h = torch.relu(torch.einsum("np,npq->nq", x, W[seg_p]) + b[seg_p])
        return self.pout(h).squeeze(1)


# ── losses / targets ────────────────────────────────────────────────────

def wdl_soft(v8):
    """strix value scale (v*8) -> soft (win, loss, draw) probs, mover POV."""
    v = torch.clamp(v8 / 8.0, -0.999, 0.999)
    pw = (1.0 + v) / 2.0
    return torch.stack([pw, 1.0 - pw, torch.zeros_like(pw)], dim=1)


def wdl_true(outc):
    """game outcome (+1 win / -1 loss / 0 draw, mover POV) -> one-hot WDL.
    NaN (no label) rows return zeros (they get weight 0 anyway)."""
    oc = torch.nan_to_num(outc, nan=2.0)
    return torch.stack([(oc == 1.0).float(), (oc == -1.0).float(),
                        (oc == 0.0).float()], dim=1)


def value_ce_mixed(logits, v8, outc, true_w):
    """Rapfi mixed loss: (1-w)·CE(soft strix WDL) + w·CE(outcome one-hot),
    the true term only where an outcome label exists."""
    lsm = F.log_softmax(logits, dim=1)
    ce_soft = -(wdl_soft(v8) * lsm).sum(dim=1)
    has = (~torch.isnan(outc)).float()
    ce_true = -(wdl_true(outc) * lsm).sum(dim=1)
    return ((1.0 - true_w * has) * ce_soft + true_w * has * ce_true).mean()


def policy_ce_padded(logits, tlogits, seg_p, npos):
    """Per-position CE(softmax(tlogits) || softmax(logits)) via padding.
    Returns (ce [npos], pl [npos,mx], tl [npos,mx])."""
    dev = logits.device
    lens = torch.bincount(seg_p, minlength=npos)
    mx = int(lens.max())
    col = torch.arange(len(seg_p), device=dev) - torch.cumsum(
        F.pad(lens, (1, 0)), 0)[seg_p]
    pl = torch.full((npos, mx), -1e9, device=dev)
    tl = torch.full((npos, mx), -1e9, device=dev)
    pl[seg_p, col] = logits
    tl[seg_p, col] = tlogits
    t = F.softmax(tl, dim=1)
    return -(t * F.log_softmax(pl, dim=1)).sum(dim=1), pl, tl


def policy_ce_mixed(logits, tlogits, seg_p, npos, played, true_w):
    """(1-w)·CE(soft strix policy) + w·CE(played-move one-hot) where the
    played move is recorded (played = column in the cand slice, -1 = none)."""
    ce_soft, pl, tl = policy_ce_padded(logits, tlogits, seg_p, npos)
    lsm = F.log_softmax(pl, dim=1)
    has = (played >= 0).float()
    idx = played.clamp(min=0).unsqueeze(1)
    ce_true = -lsm.gather(1, idx).squeeze(1)
    loss = ((1.0 - true_w * has) * ce_soft + true_w * has * ce_true).mean()
    return loss, pl, tl


# ── smoke test ──────────────────────────────────────────────────────────

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
    model = Mixnet(m=16, c=8, p=4, v=8)

    def forward(exs):
        CO, XY, CI, segs, g0l, g1l = [], [], [], [], [], []
        cb = 0
        for i, (cells, mover, cand) in enumerate(exs):
            codes, coords, ci = extract_mix(cells, mover, cand)
            CO.append(codes); XY.append(coords)
            CI.append(ci.astype(np.int64) + cb)
            segs.append(np.full(len(ci), i, dtype=np.int64))
            g0l.append(0.1 * (i + 1)); g1l.append(0.5 + 0.5 * (i % 2))
            cb += len(codes)
        seg_c = np.concatenate([np.full(len(c), i, dtype=np.int64)
                                for i, c in enumerate(CO)])
        src, nbr, useg = batch_geometry(np.concatenate(XY), seg_c, None)
        codes_t = torch.from_numpy(np.concatenate(CO).astype(np.int64))
        a = model.cell_feats(codes_t, torch.device("cpu"))
        fp, A, cnt = model.conv_pool(
            a, torch.from_numpy(src), torch.from_numpy(nbr),
            torch.from_numpy(useg), len(exs), (codes_t != 0).any(dim=1))
        cand_u = torch.from_numpy(src[np.concatenate(CI)])
        seg_p = torch.from_numpy(np.concatenate(segs))
        g0 = torch.tensor(g0l); g1 = torch.tensor(g1l)
        v = model.value(A, g0, g1)
        p = model.policy(fp, A, cnt, cand_u, seg_p, g0, g1)
        return v, p

    v, p = forward(tests)
    assert v.shape == (3, 3) and torch.isfinite(v).all()
    assert p.shape == (10,) and torch.isfinite(p).all()

    # translation invariance: shift every stone/cand by (7, -3)
    shifted = [([(q + 7, r - 3, pl) for q, r, pl in cells], mv,
                [(q + 7, r - 3) for q, r in cand])
               for cells, mv, cand in tests]
    v2, p2 = forward(shifted)
    dv = float((v - v2).abs().max()); dp = float((p - p2).abs().max())
    assert dv < 1e-4 and dp < 1e-4, f"translation: dv={dv} dp={dp}"

    # zero anchor: an isolated far candidate has zero pre-conv feature,
    # and a position's value must not change if we add such a candidate
    base = tests[1]
    aug = (base[0], base[1], base[2] + [(40, 40)])
    v3, _ = forward([base])
    v4, _ = forward([aug])
    assert float((v4[0] - v3[0]).abs().max()) < 1e-4, \
        "far candidate changed value pooling"

    # WDL soft target sanity
    probs = wdl_soft(torch.tensor([8.0, -8.0, 0.0]))
    assert abs(float(probs[0, 0]) - 0.99950) < 1e-3
    assert abs(float(probs[1, 1]) - 0.99950) < 1e-3
    assert abs(float(probs[2, 0]) - 0.5) < 1e-6

    # mirror LUT: involution, zero fixed, all-ones <-> all-twos
    assert (MIR3_11[MIR3_11] == np.arange(N_RAW)).all()
    assert MIR3_11[0] == 0
    ones = sum(1 * 3 ** i for i in range(11))
    assert MIR3_11[ones] == 2 * ones

    # policy CE: identical logits -> CE == entropy of target
    lg = torch.tensor([1.0, 2.0, 0.5, -1.0])
    seg = torch.tensor([0, 0, 0, 0])
    ce, _, _ = policy_ce_padded(lg, lg, seg, 1)
    t = F.softmax(lg, 0)
    ent = float(-(t * torch.log(t)).sum())
    assert abs(float(ce.mean()) - ent) < 1e-5

    # mixed losses: true_w=1 with played=2 -> pure -logsoftmax at col 2
    played = torch.tensor([2])
    lm, _, _ = policy_ce_mixed(lg, lg, seg, 1, played, 1.0)
    ref = float(-F.log_softmax(lg, 0)[2])
    assert abs(float(lm) - ref) < 1e-5
    vlog = torch.tensor([[0.3, -0.2, 0.1]])
    lvm = value_ce_mixed(vlog, torch.tensor([4.0]), torch.tensor([1.0]), 1.0)
    assert abs(float(lvm) - float(-F.log_softmax(vlog, 1)[0, 0])) < 1e-5
    lvn = value_ce_mixed(vlog, torch.tensor([4.0]),
                         torch.tensor([float("nan")]), 0.25)
    assert torch.isfinite(lvn)

    print(f"smoke: v {tuple(v.shape)} p {tuple(p.shape)} finite | "
          f"translation exact (dv={dv:.1e}, dp={dp:.1e}) | zero-anchor ok | "
          f"wdl + ce ok")
    print("SMOKE OK")


# ── training ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256, help="positions/step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="train on this fraction of the (fixed-seed) train "
                         "pool; the val split is identical across fractions")
    ap.add_argument("--cache", default=None,
                    help="shared dataset cache path (default <out>/cache)")
    ap.add_argument("--true-w", type=float, default=0.25,
                    help="Rapfi mixed-loss weight on true labels (outcome / "
                         "played move); soft-only samples are unaffected")
    ap.add_argument("--mirror", action="store_true",
                    help="alternate color-mirrored batches (digit-swapped "
                         "codes, negated value/outcome/tempo, same policy "
                         "targets) so the engine can query root-relative")
    ap.add_argument("--out", default="output_mixnet1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    cache = args.cache or os.path.join(out_dir, "cache_mixnet.npz")
    ds = build_mix(cache, args.threads, max_shards=args.max_shards)

    n = len(ds["tgt"])
    coffs, poffs = ds["coffs"], ds["poffs"]
    codes = torch.from_numpy(ds["codes"].astype(np.int64))
    xy = ds["xy"]
    candi = ds["cand"].astype(np.int64)
    logit = torch.from_numpy(ds["logit"])
    tgt = torch.from_numpy(ds["tgt"])
    g0 = torch.from_numpy((ds["mc"] * 0.02).astype(np.float32))
    g1 = torch.from_numpy((ds["ml"] * 0.5).astype(np.float32))
    played = torch.from_numpy(ds["played"].astype(np.int64))
    outc = torch.from_numpy(ds["outc"])
    n_true = int((ds["played"] >= 0).sum())
    print(f"true-label positions: {n_true}/{n} (w={args.true_w})", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = min(8000, n // 10)
    val_ids, train_ids = np.sort(order[:n_val]), order[n_val:]
    if args.data_frac < 1.0:      # nested subsets, identical val split
        train_ids = train_ids[:int(round(len(train_ids) * args.data_frac))]
        print(f"data-frac {args.data_frac}: {len(train_ids)} train pos",
              flush=True)

    model = Mixnet(m=args.M, c=args.C, p=args.P, v=args.V).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    mir = torch.from_numpy(MIR3_11).to(dev)

    def batch(ids, mirror=False):
        clens = coffs[ids + 1] - coffs[ids]
        cg = multi_arange(coffs[ids], clens)
        seg_c = np.repeat(np.arange(len(ids)), clens)
        src, nbr, useg = batch_geometry(xy[cg], seg_c, None)
        plens = poffs[ids + 1] - poffs[ids]
        pg = multi_arange(poffs[ids], plens)
        seg_p = np.repeat(np.arange(len(ids)), plens)
        cbase = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(clens, out=cbase[1:])
        cand_rows = candi[pg] + cbase[:-1][seg_p]
        bco = codes[torch.from_numpy(cg)].to(dev)
        bg1, by, boc = g1[ids].to(dev), tgt[ids].to(dev), outc[ids].to(dev)
        if mirror:
            bco = mir[bco]
            bg1, by, boc = -bg1, -by, -boc
        return (bco,
                torch.from_numpy(src).to(dev),
                torch.from_numpy(nbr).to(dev),
                torch.from_numpy(useg).to(dev), len(ids),
                torch.from_numpy(src[cand_rows]).to(dev),
                torch.from_numpy(seg_p).to(dev),
                g0[ids].to(dev), bg1,
                logit[pg].to(dev), by,
                played[ids].to(dev), boc)

    steps = (len(train_ids) + args.batch - 1) // args.batch
    print(f"training mixnet (M={args.M} C={args.C} P={args.P} V={args.V}) "
          f"on {len(train_ids)} pos, {steps} steps/epoch, dev={dev}",
          flush=True)
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        t0 = time.time()
        tv = tp = nb = 0
        for s in range(0, len(train_ids), args.batch):
            ids = np.sort(train_ids[s:s + args.batch])
            (bco, src, nbr, useg, npos, cand_u, seg_p, bg0, bg1, btl,
             by, bpl, boc) = batch(ids, mirror=args.mirror and nb % 2 == 1)
            opt.zero_grad()
            a = model.cell_feats(bco, dev)
            fp, A, cnt = model.conv_pool(a, src, nbr, useg, npos,
                                         (bco != 0).any(dim=1))
            lv = value_ce_mixed(model.value(A, bg0, bg1), by, boc,
                                args.true_w)
            lp, _, _ = policy_ce_mixed(
                model.policy(fp, A, cnt, cand_u, seg_p, bg0, bg1),
                btl, seg_p, npos, bpl, args.true_w)
            (lv + lp).backward()
            opt.step()
            tv += float(lv.detach()); tp += float(lp.detach()); nb += 1
        sched.step()

        with torch.no_grad():
            preds, top1, rank_sum, nv = [], 0, 0, 0
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                (bco, src, nbr, useg, npos, cand_u, seg_p, bg0, bg1, btl,
                 _, _, _) = batch(ids)
                a = model.cell_feats(bco, dev)
                fp, A, cnt = model.conv_pool(a, src, nbr, useg, npos,
                                             (bco != 0).any(dim=1))
                vl = model.value(A, bg0, bg1)
                pr = F.softmax(vl, dim=1)
                preds.append((pr[:, 0] - pr[:, 1]).cpu().numpy())
                po = model.policy(fp, A, cnt, cand_u, seg_p, bg0, bg1)
                _, pl, tl = policy_ce_padded(po, btl, seg_p, npos)
                best = tl.argmax(dim=1)
                top1 += int((pl.argmax(dim=1) == best).sum())
                bl = pl.gather(1, best.unsqueeze(1))
                rank_sum += int((pl > bl).sum()) + npos
                nv += npos
            pred = np.concatenate(preds)
        y = (tgt[val_ids] / 8.0).numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        print(f"epoch {ep+1}/{args.epochs}: v {tv/nb:.4f} p {tp/nb:.4f} "
              f"| corr {corr:.4f} sp {spearman(pred, y):.4f} "
              f"| top1 {top1/nv:.3f} mrank {rank_sum/nv:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "M": args.M, "C": args.C, "P": args.P, "V": args.V,
                "CLIP": CLIP, "arch": "mixnet1"},
               os.path.join(out_dir, "mixnet.pt"))
    print(f"saved {out_dir}/mixnet.pt")


if __name__ == "__main__":
    main()

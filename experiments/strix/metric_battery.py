"""Metric battery: which offline metric actually discriminates play strength?

Ground truth = strix value of EVERY child (one stone placed on a D2
candidate) of each base position, in base-mover POV tanh units. This is
what search consumes: a ranking over siblings.

Value scorers (evaluate children, like leaf eval): linear (original
pattern eval), champion (frozen hybrid), distill (frozen hybrid, current
engine), trunk (new joint net). Known Elo order: linear < champion <
distill; a useful metric must reproduce it AND show the residual gap to
the oracle.

Policy scorers (score candidate cells directly at the base position):
delta (linear |dEval|), tables (PW/PC), ptrunk (trunk policy head),
strixpol (strix's own policy logits = imitation ceiling).

Metrics per scorer x {REAL, PERTURBED}:
  posval    spearman of base-position value vs oracle (the old, suspected-
            useless metric, for contrast)
  sib-all   pairwise sibling ordering accuracy, all pairs
  sib-close pairs with 0.05 < |d oracle| <= 0.3   (the hard calls)
  sib-dec   pairs with |d oracle| > 0.5           (blunder avoidance)
  top1      argmax matches oracle argmax
  regret    oracle(best child) - oracle(chosen child), mean / p90
  fb<=3     forced-block cells ranked in top 3 (policy scorers)

Run in the hexo venv (GPU):
    .../hexo-strix/.venv/bin/python metric_battery.py --n 300
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
SEAL = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / ".." / "nnue"))

import torch

from features import extract_features, net_forward
from strix_bridge import load_strix, state_from_cells, value_batch
from refutation_recall import (cand_cells, load_tables, rankers,
                               forced_blocks, perturb)
from sibling_extract import completes_six
from policy_extract import policy_batch
import trunk_train
from trunk_train import Trunk


# ── scorer loading ──────────────────────────────────────────────────────

def load_pv(path):
    text = open(path).read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    return np.array([float(x) for x in nums])


def load_frozen(dirname):
    d = dict(np.load(SEAL / dirname / "net_data.h.npz"))
    net = {k: d[k].astype(np.float64) for k in ("ew", "ec", "w1", "b1", "w2")}
    net.update(b2=float(d["b2"]), out_scale=float(d["out_scale"]),
               clip=float(d["clip"]))
    return net, float(d["lin_blend"]), load_pv(SEAL / dirname /
                                               "pattern_data.h")


class TrunkScorer:
    def __init__(self, path, device="cuda"):
        ck = torch.load(path, map_location="cpu")
        self.arch = ck.get("arch", "v1")
        if self.arch == "v2":
            from trunk_train2 import Trunk2
            self.m = Trunk2()
        else:
            self.m = Trunk()
        self.m.load_state_dict(ck["state"])
        self.m.to(device).eval()
        self.dev = device

    def value_many(self, feats, mcs, mls):
        """feats: list of (trip, wi, wc). Returns np array of values."""
        out = np.empty(len(feats))
        B = 4096
        with torch.no_grad():
            for s in range(0, len(feats), B):
                sub = feats[s:s + B]
                trip = torch.from_numpy(
                    np.concatenate([f[0] for f in sub]).astype(np.int64))
                seg = torch.repeat_interleave(
                    torch.arange(len(sub)),
                    torch.tensor([len(f[0]) for f in sub]))
                wi = torch.from_numpy(
                    np.concatenate([f[1] for f in sub]).astype(np.int64))
                wc = torch.from_numpy(
                    np.concatenate([f[2] for f in sub]).astype(np.float32))
                wo = torch.tensor(
                    np.cumsum([0] + [len(f[1]) for f in sub]))
                g0 = torch.tensor((np.array(mcs[s:s + B]) * 0.02),
                                  dtype=torch.float32)
                g1 = torch.tensor((np.array(mls[s:s + B]) * 0.5),
                                  dtype=torch.float32)
                v = self.m.value(trip.to(self.dev), seg.to(self.dev),
                                 len(sub), wi.to(self.dev), wc.to(self.dev),
                                 wo.to(self.dev), g0.to(self.dev),
                                 g1.to(self.dev))
                out[s:s + B] = v.cpu().numpy()
        return out

    def policy_many(self, cand_trip, parent=None):
        with torch.no_grad():
            t = torch.from_numpy(cand_trip.astype(np.int64)).to(self.dev)
            if self.arch == "v2":
                trip, wi, wc = parent
                acc = self.m.accum(
                    torch.from_numpy(trip.astype(np.int64)).to(self.dev),
                    torch.zeros(len(trip), dtype=torch.int64,
                                device=self.dev),
                    1,
                    torch.from_numpy(wi.astype(np.int64)).to(self.dev),
                    torch.from_numpy(wc.astype(np.float32)).to(self.dev),
                    torch.tensor([0, len(wi)], device=self.dev))
                seg_p = torch.zeros(len(cand_trip), dtype=torch.int64,
                                    device=self.dev)
                return self.m.policy(t, acc, seg_p).cpu().numpy()
            return self.m.policy(t).cpu().numpy()


# ── feature extraction over children (parallel) ────────────────────────

def _child_feats(args_):
    cells, mover, ml, mc = args_
    w_idx, w_cnt, c_idx, c_cnt = extract_features(cells, mover)
    trip, wi, wc, _ = trunk_train.extract(cells, mover, [(0, 0)])
    return (w_idx, w_cnt, c_idx, c_cnt, trip, wi, wc, ml, mc)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--trunk", default=str(SCRIPT_DIR / "output_trunk"
                                           / "trunk.pt"))
    ap.add_argument("--out", default=str(SCRIPT_DIR / "metric_battery.json"))
    args = ap.parse_args()

    model, mc_, gc = load_strix()
    pw, pc, pv_cur = load_tables()
    net_ch, blend_ch, pv_ch = load_frozen("champion_frozen")
    net_di, blend_di, pv_di = load_frozen("distill_frozen")
    trunk = (TrunkScorer(args.trunk) if os.path.exists(args.trunk) else None)
    if trunk is None:
        print("NOTE: no trunk checkpoint, running old nets only")

    rng = random.Random(31)
    with open(SCRIPT_DIR / "strong_play_recs.pkl", "rb") as fh:
        recs = [r[:4] for r in pickle.load(fh) if r[3] >= 10]
    seen_b = set()
    ded = []
    for r in recs:
        k = tuple(sorted(map(tuple, r[0])))
        if k not in seen_b:
            seen_b.add(k)
            ded.append(r)
    recs = ded
    rng.shuffle(recs)

    def make_perturbed(base_recs):
        out = []
        for cells, mover, ml, mc in base_recs:
            r = perturb(cells, mover, rng, pv_cur, pw, pc,
                        rng.choice([1, 2]))
            if r:
                out.append((r[0], r[1], 2, mc + 2))
        return out

    sets = {"REAL": recs[:args.n],
            "PERT": make_perturbed(recs[args.n:args.n * 2])}
    human_pkl = SCRIPT_DIR / "human_recs.pkl"
    if human_pkl.exists():
        with open(human_pkl, "rb") as fh:
            sets["HUMAN"] = pickle.load(fh)[:args.n]

    results = {}
    for set_name, base_set in sets.items():
        t0 = time.time()
        # 1. build children for every base position
        rows = []          # (bi, cand list, child descriptors)
        child_meta = []    # flat: (cells, mover, ml, mc, base_idx, flip)
        base_meta = []
        for bi, (cells, mover, ml, mc) in enumerate(base_set):
            s = state_from_cells(cells, mover, ml, gc)
            if s is None or s.is_terminal():
                continue
            cand = sorted(cand_cells(cells))
            kept, kidx = [], []
            for c in cand:
                ccells = cells + [(c[0], c[1], mover)]
                if ml == 2:
                    cm, cml = mover, 1
                else:
                    cm, cml = (3 - mover), 2
                kept.append(c)
                kidx.append(len(child_meta))
                child_meta.append((ccells, cm, cml, mc + 1, bi,
                                   -1.0 if cm != mover else 1.0))
            base_meta.append((bi, cells, mover, ml, mc, kept, kidx))
        print(f"[{set_name}] {len(base_meta)} base positions, "
              f"{len(child_meta)} children", flush=True)

        # 2. oracle: strix value of every child (+ base), base-mover POV
        states, term, drop = [], np.zeros(len(child_meta), bool), \
            np.zeros(len(child_meta), bool)
        for i, (ccells, cm, cml, cmc, bi, flip) in enumerate(child_meta):
            q, r, pl = ccells[-1]
            occ = {(c[0], c[1]): c[2] for c in ccells}
            if completes_six(occ, q, r, pl):
                term[i] = True
                continue
            s = state_from_cells(ccells, cm, cml, gc)
            if s is None:
                drop[i] = True
            else:
                states.append(s)
        vals = value_batch(model, mc_, states, chunk=512)
        oracle = np.zeros(len(child_meta))
        vi = 0
        for i, (ccells, cm, cml, cmc, bi, flip) in enumerate(child_meta):
            if drop[i]:
                continue
            if term[i]:
                oracle[i] = 1.0     # base mover just completed 6
            else:
                oracle[i] = flip * float(vals[vi])
                vi += 1
        base_states = [state_from_cells(c, m, l, gc)
                       for _, c, m, l, _, _, _ in base_meta]
        base_oracle = np.array(value_batch(model, mc_, base_states,
                                           chunk=512))
        # strix policy logits at base positions
        strix_pols = policy_batch(model, mc_, base_states, chunk=256)
        print(f"[{set_name}] oracle done {time.time()-t0:.0f}s", flush=True)

        # 3. child features (parallel)
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            feats = pool.map(_child_feats,
                             [(cm[0], cm[1], cm[2], cm[3])
                              for cm in child_meta], chunksize=256)
        print(f"[{set_name}] features done {time.time()-t0:.0f}s",
              flush=True)

        # 4. score children with every value net
        def nnue_vals(net, blend, pv):
            out = np.empty(len(feats))
            for i, f in enumerate(feats):
                w_idx, w_cnt, c_idx, c_cnt, trip, wi, wc, ml2, mc2 = f
                v = net_forward(w_idx, w_cnt, c_idx, c_cnt, mc2,
                                ml2 * 0.5, net)
                if blend:
                    v += blend * float((pv[w_idx] * w_cnt).sum())
                out[i] = v
            return out

        val_scores = {
            "linear": np.array([float((pv_cur[f[0]] * f[1]).sum())
                                for f in feats]),
            "champion": nnue_vals(net_ch, blend_ch, pv_ch),
            "distill": nnue_vals(net_di, blend_di, pv_di),
        }
        if trunk:
            val_scores["trunk"] = trunk.value_many(
                [(f[4], f[5], f[6]) for f in feats],
                [f[8] for f in feats], [f[7] for f in feats])
        # to base-mover POV
        flips = np.array([cm[5] for cm in child_meta])
        for k in val_scores:
            val_scores[k] = val_scores[k] * flips
        print(f"[{set_name}] value nets done {time.time()-t0:.0f}s",
              flush=True)

        # 5. policy scores at base positions
        pol_scores = {k: np.zeros(len(child_meta))
                      for k in ("delta", "tables", "ptrunk", "strixpol")}
        fb_mask = np.zeros(len(child_meta), bool)
        for (bi, cells, mover, ml, mc, kept, kidx), spol in zip(
                base_meta, strix_pols):
            pol, delta = rankers(cells, mover, kept, pw, pc, pv_cur)
            pol_scores["tables"][kidx] = pol
            pol_scores["delta"][kidx] = delta
            if trunk:
                ptrip, pwi, pwc, ctrip = trunk_train.extract(
                    [tuple(c) for c in cells], mover, kept)
                pol_scores["ptrunk"][kidx] = trunk.policy_many(
                    ctrip, parent=(ptrip, pwi, pwc))
            # strix logits: translate coords if the bridge translated
            owner = {(q, r): p for q, r, p in cells}
            if owner.get((0, 0)) != 1:
                a = [(q, r) for q, r, p in cells if p == 1][0]
            else:
                a = (0, 0)
            lo = min(spol.values()) - 10.0 if spol else -10.0
            pol_scores["strixpol"][kidx] = [
                spol.get((c[0] - a[0], c[1] - a[1]), lo) for c in kept]
            for b in forced_blocks(cells, mover, kept):
                fb_mask[kidx[kept.index(b)]] = True
        if not trunk:
            del pol_scores["ptrunk"]

        # 6. metrics
        seg = np.array([cm[4] for cm in child_meta])
        keep = ~drop

        def per_position_metrics(scores, ml_filter=None):
            sib = {"all": [0, 0], "close": [0, 0], "dec": [0, 0]}
            top1 = 0
            regs = []
            npos = 0
            for bi, cells, mover, ml, mc, kept, kidx in base_meta:
                if ml_filter is not None and ml != ml_filter:
                    continue
                m = keep[kidx]
                if m.sum() < 2:
                    continue
                o = oracle[np.array(kidx)[m]]
                sc = scores[np.array(kidx)[m]]
                npos += 1
                do = o[:, None] - o[None, :]
                ds = sc[:, None] - sc[None, :]
                iu = np.triu_indices(len(o), 1)
                do, ds = do[iu], ds[iu]
                nz = do != 0
                agree = (np.sign(do[nz]) == np.sign(ds[nz]))
                ado = np.abs(do[nz])
                for k, mk in (("all", np.ones_like(ado, bool)),
                              ("close", (ado > 0.05) & (ado <= 0.3)),
                              ("dec", ado > 0.5)):
                    sib[k][0] += int(agree[mk].sum())
                    sib[k][1] += int(mk.sum())
                pick = int(np.argmax(sc))
                top1 += int(o[pick] == o.max())
                regs.append(float(o.max() - o[pick]))
            regs = np.array(regs) if regs else np.zeros(1)
            return {
                "sib_all": sib["all"][0] / max(sib["all"][1], 1),
                "sib_close": sib["close"][0] / max(sib["close"][1], 1),
                "sib_dec": sib["dec"][0] / max(sib["dec"][1], 1),
                "top1": top1 / max(npos, 1),
                "regret_mean": float(regs.mean()),
                "regret_p90": float(np.quantile(regs, 0.9)),
            }

        def fb_rank_metrics(scores):
            hit = tot = 0
            for bi, cells, mover, ml, mc, kept, kidx in base_meta:
                ki = np.array(kidx)
                m = keep[ki]
                sc = scores[ki[m]]
                fb = fb_mask[ki[m]]
                order = np.argsort(-sc)
                r = np.empty(len(sc), int)
                r[order] = np.arange(len(sc))
                for j in np.nonzero(fb)[0]:
                    tot += 1
                    hit += int(r[j] < 3)
            return hit / max(tot, 1)

        def add_ml_split(met, sc):
            for mlv in (1, 2):
                sub = per_position_metrics(sc, ml_filter=mlv)
                met[f"regret_ml{mlv}"] = sub["regret_mean"]
                met[f"top1_ml{mlv}"] = sub["top1"]
            return met

        res = {}
        for name, sc in val_scores.items():
            met = add_ml_split(per_position_metrics(sc), sc)
            # old-style: base position value vs oracle (value nets score
            # the base position itself = child of the "pre-position")
            bo, bs = [], []
            for (bi, cells, mover, ml, mc, kept, kidx), ov in zip(
                    base_meta, base_oracle):
                w_idx, w_cnt, c_idx, c_cnt = extract_features(cells, mover)
                if name == "linear":
                    v = float((pv_cur[w_idx] * w_cnt).sum())
                elif name == "champion":
                    v = net_forward(w_idx, w_cnt, c_idx, c_cnt, mc, ml * 0.5,
                                    net_ch) + blend_ch * float(
                        (pv_ch[w_idx] * w_cnt).sum())
                elif name == "distill":
                    v = net_forward(w_idx, w_cnt, c_idx, c_cnt, mc, ml * 0.5,
                                    net_di) + blend_di * float(
                        (pv_di[w_idx] * w_cnt).sum())
                else:
                    trip, wi, wc, _ = trunk_train.extract(
                        [tuple(c) for c in cells], mover, [(0, 0)])
                    v = trunk.value_many([(trip, wi, wc)], [mc], [ml])[0]
                bs.append(v); bo.append(float(ov))
            met["posval_spearman"] = spearman(np.array(bs), np.array(bo))
            res[f"value/{name}"] = met
        for name, sc in pol_scores.items():
            met = add_ml_split(per_position_metrics(sc), sc)
            met["fb_top3"] = fb_rank_metrics(sc)
            res[f"policy/{name}"] = met
        results[set_name] = res
        print(f"[{set_name}] metrics done {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)

    cols = ["posval_spearman", "sib_all", "sib_close", "sib_dec", "top1",
            "regret_mean", "regret_p90", "regret_ml1", "regret_ml2",
            "fb_top3"]
    for set_name, res in results.items():
        print(f"\n=== {set_name} ===")
        print(f"{'scorer':<18}" + "".join(f"{c:>12}" for c in cols))
        for name, met in res.items():
            row = f"{name:<18}"
            for c in cols:
                row += (f"{met[c]:>12.3f}" if c in met else f"{'-':>12}")
            print(row)


if __name__ == "__main__":
    main()

"""Train the policy-ordering tables PW[729] + PC[8548] against strix policy.

score(cell) = sum_18 PW[window_pattern] + PC[conj_class]   (mover-relative)
loss = KL(softmax(strix logits over candidates) || softmax(scores))
     = soft-target cross-entropy per position.

Reads policy_targets/*.npz (from policy_extract.py), extracts per-cell
features in parallel, trains, writes policy.pt + reports ranking metrics
(top-1 agreement with strix, mean rank of strix's best move — compared
against the old linear _move_delta ordering).

Run in the SealBot venv:
    ../../.venv/bin/python policy_train.py --epochs 8 --threads 14
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "nnue"))

NUM_CLASSES = 8548


def _process_shard(path):
    from policy_features import extract_cell_features
    d = np.load(path, allow_pickle=True)
    lens = d["lens"]
    cq, cr, lg = d["cell_q"], d["cell_r"], d["logit"]
    metas = d["meta"]
    offs = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    W, C, L, LN = [], [], [], []
    for i, (cells, mover, ml, mc) in enumerate(metas):
        sl = slice(offs[i], offs[i + 1])
        cand = list(zip(cq[sl].tolist(), cr[sl].tolist()))
        win, cls = extract_cell_features(list(map(tuple, cells)), int(mover),
                                         cand)
        W.append(win)
        C.append(cls)
        L.append(lg[sl])
        LN.append(len(cand))
    return (np.concatenate(W), np.concatenate(C),
            np.concatenate(L).astype(np.float32),
            np.array(LN, dtype=np.int32))


def build_dataset(target_dir, cache, workers, max_shards=None):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(target_dir, "*.npz")))
    if max_shards:
        shards = shards[:max_shards]
    print(f"building policy dataset from {len(shards)} shards...")
    t0 = time.time()
    Ws, Cs, Ls, LNs = [], [], [], []
    with mp.Pool(workers) as pool:
        for i, (w, c, l, ln) in enumerate(pool.imap(_process_shard, shards)):
            Ws.append(w); Cs.append(c); Ls.append(l); LNs.append(ln)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(shards)} shards {time.time()-t0:.0f}s",
                      flush=True)
    lens = np.concatenate(LNs)
    ds = {
        "win": np.concatenate(Ws),          # [ncand, 18]
        "cls": np.concatenate(Cs),          # [ncand]
        "logit": np.concatenate(Ls),        # [ncand]
        "lens": lens,
    }
    np.savez(cache, **ds)
    print(f"policy dataset: {len(lens)} positions, {len(ds['cls'])} candidates,"
          f" {time.time()-t0:.0f}s")
    return ds


class PolicyTables(nn.Module):
    def __init__(self):
        super().__init__()
        self.pw = nn.Embedding(729, 1)
        self.pc = nn.Embedding(NUM_CLASSES + 1, 1)  # last row = cls -1
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pc.weight)

    def forward(self, win, cls):
        return (self.pw(win).squeeze(-1).sum(dim=1)
                + self.pc(cls).squeeze(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="policy_targets")
    ap.add_argument("--out", default="output_policy")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-pos", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset(os.path.join(SCRIPT_DIR, args.targets),
                       os.path.join(out_dir, "policy_ds.npz"), args.threads)

    lens = ds["lens"]
    n = len(lens)
    offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    win = torch.from_numpy(ds["win"].astype(np.int64))
    cls_np = ds["cls"].astype(np.int64)
    cls_np[cls_np < 0] = NUM_CLASSES
    cls = torch.from_numpy(cls_np)
    logit = torch.from_numpy(ds["logit"])

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_ids, train_ids = order[:n_val], order[n_val:]

    model = PolicyTables()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    def batch_loss(ids, eval_metrics=False):
        gather = np.concatenate(
            [np.arange(offs[i], offs[i + 1]) for i in ids])
        bl = torch.from_numpy(lens[ids].astype(np.int64))
        bw, bc = win[gather], cls[gather]
        tl = logit[gather]
        scores = model(bw, bc)
        # per-position soft-target CE via segment softmax
        seg = torch.repeat_interleave(torch.arange(len(ids)), bl)
        t_max = torch.zeros(len(ids)).index_reduce_(0, seg, tl, "amax")
        t_exp = torch.exp(tl - t_max[seg])
        t_sum = torch.zeros(len(ids)).index_add_(0, seg, t_exp)
        t_soft = t_exp / t_sum[seg]
        s_max = torch.zeros(len(ids)).index_reduce_(0, seg, scores, "amax",
                                                    include_self=False)
        s_exp = torch.exp(scores - s_max[seg])
        s_sum = torch.zeros(len(ids)).index_add_(0, seg, s_exp)
        s_logsoft = (scores - s_max[seg]) - torch.log(s_sum[seg])
        loss = -(t_soft * s_logsoft).sum() / len(ids)
        if not eval_metrics:
            return loss
        # top-1 agreement + rank of strix best
        top1 = rank_sum = 0
        off = 0
        for L in bl.tolist():
            t = tl[off:off + L]
            s = scores[off:off + L]
            best = int(torch.argmax(t))
            top1 += int(int(torch.argmax(s)) == best)
            rank_sum += int((s > s[best]).sum()) + 1
            off += L
        return loss, top1 / len(ids), rank_sum / len(ids)

    print(f"training on {len(train_ids)} positions "
          f"({len(ds['cls'])} candidates)")
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        t0 = time.time()
        tot = nb = 0
        for s in range(0, len(train_ids), args.batch_pos):
            ids = train_ids[s:s + args.batch_pos]
            opt.zero_grad()
            loss = batch_loss(ids)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        with torch.no_grad():
            vl, top1, mrank = batch_loss(val_ids[:4000], eval_metrics=True)
        print(f"epoch {ep+1}/{args.epochs}: train {tot/nb:.4f} "
              f"val {float(vl):.4f} top1 {top1:.3f} "
              f"mean-rank-of-best {mrank:.2f} ({time.time()-t0:.0f}s)",
              flush=True)

    torch.save({"pw": model.pw.weight.detach().squeeze(-1).clone(),
                "pc": model.pc.weight.detach().squeeze(-1).clone()},
               os.path.join(out_dir, "policy.pt"))
    print(f"saved {out_dir}/policy.pt")


if __name__ == "__main__":
    main()

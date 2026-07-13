"""Train the NNUE eval on self-play data.

Architecture (must match current/engine/bot.h::_leaf_eval):
    acc  = sum EW[window pattern] + sum EC[conjunction class]   (K dims)
    h    = clip(acc, 0, CLIP)
    out  = W2 @ relu(W1 @ [h; move_count*0.02] + b1) + b2
    eval = out * OUT_SCALE   (engine side)

Training target: t = lam * sigmoid(score / SCORE_SCALE) + (1-lam) * outcome,
with the net's sigmoid(out) matched by BCE. Mirror augmentation (color swap
= negated logit via feature permutations) is applied on the fly.

Usage:
    python train.py --data data/gen0 --out output/gen0
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
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from features import extract_features, MIRROR729, CLASS_MIRROR, NUM_CLASSES

K, H = 32, 32
CLIP = 8.0
OUT_SCALE = 600.0
SCORE_CLAMP = 30000.0    # tame mate scores
WIN_SCORE_MIN = 1e6      # anything above this is a mate score


# ── Dataset build: shards -> packed sparse feature arrays ──────────────────

def _clamp_score(sc):
    if abs(sc) > WIN_SCORE_MIN:
        sc = np.sign(sc) * SCORE_CLAMP
    return float(np.clip(sc, -SCORE_CLAMP, SCORE_CLAMP))


def _process_shard(sp):
    from features import extract_features as ef
    feat_idx, feat_cnt, lens = [], [], []
    scores, sdeeps, outs, mcs, mls = [], [], [], [], []
    with open(sp, "rb") as f:
        games = pickle.load(f)
    for g in games:
        winner = g["winner"]
        for pos in g["positions"]:
            mover = pos["mover"]
            w_idx, w_cnt, c_idx, c_cnt = ef(pos["cells"], mover)
            fi = np.concatenate([w_idx, c_idx + 729])
            fc = np.concatenate([w_cnt, c_cnt])
            feat_idx.append(fi.astype(np.int32))
            feat_cnt.append(fc.astype(np.int16))
            lens.append(len(fi))
            scores.append(_clamp_score(pos["score"]))
            sdeeps.append(_clamp_score(pos.get("score_deep", pos["score"])))
            outs.append(0.5 if winner == 0 else
                        (1.0 if winner == mover else 0.0))
            mcs.append(pos["move_count"])
            mls.append(pos.get("moves_left", 2))
    if not lens:
        return None
    return (np.concatenate(feat_idx), np.concatenate(feat_cnt),
            np.array(lens, dtype=np.int64),
            np.array(scores, dtype=np.float32),
            np.array(sdeeps, dtype=np.float32),
            np.array(outs, dtype=np.float32),
            np.array(mcs, dtype=np.int32),
            np.array(mls, dtype=np.int32))


def build_dataset(data_dir, cache_path, max_positions=None, workers=16):
    if os.path.exists(cache_path):
        print(f"loading cached dataset {cache_path}")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    print(f"building dataset from {len(shards)} shards ({workers} workers)...")
    t0 = time.time()

    fis, fcs, lens_all = [], [], []
    scores, sdeeps, outs, mcs, mls = [], [], [], [], []
    n_pos = 0
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_process_shard, shards)):
            if res is None:
                continue
            fi, fc, lens, sc, sd, ou, mc, ml = res
            fis.append(fi); fcs.append(fc); lens_all.append(lens)
            scores.append(sc); sdeeps.append(sd); outs.append(ou)
            mcs.append(mc); mls.append(ml)
            n_pos += len(lens)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(shards)} shards, {n_pos} positions, "
                      f"{time.time()-t0:.0f}s", flush=True)
            if max_positions and n_pos >= max_positions:
                pool.terminate()
                break

    lens = np.concatenate(lens_all)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    ds = {
        "feat_idx": np.concatenate(fis),
        "feat_cnt": np.concatenate(fcs),
        "offsets": offsets,
        "score": np.concatenate(scores),
        "score_deep": np.concatenate(sdeeps),
        "outcome": np.concatenate(outs),
        "move_count": np.concatenate(mcs),
        "moves_left": np.concatenate(mls),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **ds)
    print(f"dataset: {n_pos} positions, "
          f"{len(ds['feat_idx'])/n_pos:.0f} feats/pos, {time.time()-t0:.0f}s")
    return ds


# ── Model ───────────────────────────────────────────────────────────────────

class SealNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(729 + NUM_CLASSES, K, mode="sum",
                                   include_last_offset=True)
        nn.init.normal_(self.emb.weight, 0.0, 0.05)
        with torch.no_grad():
            self.emb.weight[0].zero_()  # window pattern 0 never occurs; keep 0
        self.w1 = nn.Linear(K + 2, H)   # + [move_count*0.02, tempo]
        self.w2 = nn.Linear(H, 1)

    def forward(self, idx, cnt, offsets, g0, g1):
        acc = self.emb(idx, offsets, per_sample_weights=cnt)
        h = torch.clamp(acc, 0.0, CLIP)
        x = torch.cat([h, g0.unsqueeze(1), g1.unsqueeze(1)], dim=1)
        return self.w2(torch.relu(self.w1(x))).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/gen0")
    ap.add_argument("--out", type=str, default="output/gen0")
    ap.add_argument("--loss", choices=["wdl", "score", "mixdeep"], default="wdl",
                    help="wdl: BCE on blended win-prob; score: Huber "
                         "regression on search score (keeps resolution); "
                         "mixdeep: lam*score + (1-lam)*score_deep, both /1000")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data_dir = os.path.join(SCRIPT_DIR, args.data)
    out_dir = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    ds = build_dataset(data_dir, os.path.join(out_dir, "dataset_v2.npz"),
                       args.max_positions)
    n = len(ds["score"])

    if args.loss == "wdl":
        # score->prob squash scaled so ~75% of nonzero scores stay unsaturated
        score_scale = float(np.quantile(np.abs(ds["score"]), 0.75)) / 2.0
        score_scale = max(score_scale, 500.0)
        print(f"SCORE_SCALE = {score_scale:.0f} (data-driven)")
        t_score = 1.0 / (1.0 + np.exp(-ds["score"] / score_scale))
        target = (args.lam * t_score
                  + (1.0 - args.lam) * ds["outcome"]).astype(np.float32)
        out_scale = 600.0
    elif args.loss == "mixdeep":
        if "score_deep" not in ds:
            raise SystemExit("mixdeep needs a dataset built with score_deep "
                             "(rebuild cache from strix-relabeled shards)")
        target = (args.lam * ds["score"] / 1000.0
                  + (1.0 - args.lam) * ds["score_deep"] / 1000.0
                  ).astype(np.float32)
        out_scale = 1000.0
        print(f"mixdeep targets: mean {target.mean():.2f} "
              f"std {target.std():.2f}")
    else:
        # regression target: score in units of 1000, outcome as +-8 anchor
        t_out = (ds["outcome"] * 2.0 - 1.0) * 8.0
        target = (args.lam * ds["score"] / 1000.0
                  + (1.0 - args.lam) * t_out).astype(np.float32)
        out_scale = 1000.0   # engine eval = model output * 1000
        print(f"score-regression targets: mean {target.mean():.2f} "
              f"std {target.std():.2f}")
    g0 = (ds["move_count"] * 0.02).astype(np.float32)
    g1 = (ds["moves_left"] * 0.5).astype(np.float32)  # root always to move

    # mirror-permutation for augmentation over the merged feature space
    perm = np.concatenate([MIRROR729, CLASS_MIRROR + 729]).astype(np.int64)
    perm_t = torch.from_numpy(perm)

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_ids, train_ids = order[:n_val], order[n_val:]

    offsets = ds["offsets"]
    fidx = torch.from_numpy(ds["feat_idx"].astype(np.int64))
    fcnt = torch.from_numpy(ds["feat_cnt"].astype(np.float32))
    tgt = torch.from_numpy(target)
    g0_t = torch.from_numpy(g0)
    g1_t = torch.from_numpy(g1)

    def gather_batch(ids, mirror=False):
        lens = offsets[ids + 1] - offsets[ids]
        bo = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(lens, out=bo[1:])
        gather = np.concatenate(
            [np.arange(offsets[i], offsets[i + 1]) for i in ids])
        bi = fidx[gather]
        bc = fcnt[gather]
        bt = tgt[ids]
        bg0 = g0_t[ids]
        bg1 = g1_t[ids]
        if mirror:
            # color swap: root becomes the non-mover -> tempo negates
            bi = perm_t[bi]
            bt = (1.0 - bt) if args.loss == "wdl" else -bt
            bg1 = -bg1
        return bi, bc, torch.from_numpy(bo), bg0, bg1, bt

    model = SealNNUE()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    if args.loss == "wdl":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.HuberLoss(delta=4.0)

    print(f"training: {len(train_ids)} train / {n_val} val, "
          f"lam={args.lam}, {args.epochs} epochs")
    best_val = float("inf")

    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train_ids)
        t0 = time.time()
        tot_loss = n_batches = 0
        for s in range(0, len(train_ids), args.batch):
            ids = train_ids[s:s + args.batch]
            mirror = (n_batches % 2 == 1)   # alternate color-mirrored batches
            bi, bc, bo, bg0, bg1, bt = gather_batch(ids, mirror=mirror)
            opt.zero_grad()
            logit = model(bi, bc, bo, bg0, bg1)
            loss = loss_fn(logit, bt)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = 0.0
            nb = 0
            for s in range(0, len(val_ids), args.batch):
                ids = val_ids[s:s + args.batch]
                bi, bc, bo, bg0, bg1, bt = gather_batch(ids)
                vl += float(loss_fn(model(bi, bc, bo, bg0, bg1), bt))
                nb += 1
            vl /= max(nb, 1)

        marker = ""
        if vl < best_val:
            best_val = vl
            sd = {
                "ew.weight": model.emb.weight[:729].detach().clone(),
                "ec.weight": model.emb.weight[729:].detach().clone(),
                "w1.weight": model.w1.weight.detach().clone(),
                "w1.bias": model.w1.bias.detach().clone(),
                "w2.weight": model.w2.weight.detach().clone(),
                "w2.bias": model.w2.bias.detach().clone(),
                "out_scale": torch.tensor(out_scale),
                "clip": torch.tensor(CLIP),
            }
            torch.save(sd, os.path.join(out_dir, "net.pt"))
            marker = "  *saved*"
        print(f"epoch {ep+1}/{args.epochs}: train {tot_loss/n_batches:.4f} "
              f"val {vl:.4f} ({time.time()-t0:.0f}s){marker}", flush=True)

    print(f"best val loss {best_val:.4f}; checkpoint {out_dir}/net.pt")


if __name__ == "__main__":
    main()

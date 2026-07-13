"""Refit the linear window table (pattern_data.h) on deep search labels.

Solves ridge regression in score space for the ~35 free single-color
reversal classes (mirror antisymmetry + reversal symmetry enforced,
dead/mixed patterns pinned to 0). Writes output/pattern_deep.h; does NOT
touch current/ (copy manually / via gate flow).

Rationale: the CMA-era table gets the *sign* of decided positions wrong
~43% of the time, which poisons the hybrid blend above ~0.2 and degrades
move ordering. A deep-label refit fixes both consumers.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "cma"))

from symmetry import (mirror, reverse_pattern, save_pattern_data_h)

SCORE_CLAMP = 30000.0


def build_classes():
    """Single-color(1) patterns grouped by reversal symmetry."""
    classes = {}  # representative -> list of pattern indices
    for i in range(1, 729):
        digs, x = [], i
        ok = True
        for _ in range(6):
            d = x % 3
            if d == 2:
                ok = False
                break
            digs.append(d)
            x //= 3
        if not ok:
            continue
        rep = min(i, reverse_pattern(i))
        classes.setdefault(rep, set()).add(i)
        classes[rep].add(reverse_pattern(i))
    reps = sorted(classes)
    return reps, [sorted(classes[r]) for r in reps]


def main():
    ds_path = os.path.join(SCRIPT_DIR, "output", "deep_score", "dataset_v2.npz")
    d = np.load(ds_path)
    n = len(d["score"])
    print(f"{n} positions from {ds_path}")

    reps, members = build_classes()
    n_cls = len(reps)
    print(f"{n_cls} free linear classes")

    # map each window pattern index -> (class, sign) or unused
    cls_of = np.full(729, -1, dtype=np.int32)
    sign_of = np.zeros(729, dtype=np.float64)
    for c, mem in enumerate(members):
        for i in mem:
            cls_of[i] = c
            sign_of[i] = 1.0
            j = mirror(i)
            cls_of[j] = c
            sign_of[j] = -1.0

    # design matrix Z (n x n_cls) accumulated sparsely
    Z = np.zeros((n, n_cls), dtype=np.float32)
    offs, fidx, fcnt = d["offsets"], d["feat_idx"], d["feat_cnt"]
    for i in range(n):
        sl = slice(offs[i], offs[i + 1])
        idx = fidx[sl]
        cnt = fcnt[sl]
        wmask = idx < 729
        wi = idx[wmask]
        wc = cnt[wmask].astype(np.float64)
        c = cls_of[wi]
        s = sign_of[wi]
        use = c >= 0
        np.add.at(Z[i], c[use], (s[use] * wc[use]).astype(np.float32))
        if (i + 1) % 100000 == 0:
            print(f"  {i+1}/{n}")

    raw = d["score"].astype(np.float64)
    # fit ONLY on non-mate positions: mates dominate l2 and a 35-param
    # linear model chasing them destroys ordering in the quiet range
    quiet = np.abs(raw) < 25000.0
    y = np.clip(raw, -8000.0, 8000.0)
    Zq, yq = Z[quiet].astype(np.float64), y[quiet]
    print(f"fitting on {quiet.sum()} quiet positions "
          f"({100*quiet.mean():.0f}% of data)")

    # intercept absorbs the mover-tempo bias (y mean ~ +2000); the
    # antisymmetric mirror-difference features cannot express a constant.
    Zi = np.hstack([Zq, np.ones((len(yq), 1))])
    ZtZ = Zi.T @ Zi
    Zty = Zi.T @ yq
    alpha = 1e-3 * np.trace(ZtZ) / (n_cls + 1)
    reg = alpha * np.eye(n_cls + 1)
    reg[-1, -1] = 0.0   # do not shrink the intercept
    wi_ = np.linalg.solve(ZtZ + reg, Zty)
    w, intercept = wi_[:-1], wi_[-1]

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    pred = Zq @ w + intercept
    ss = 1 - ((yq - pred) ** 2).sum() / ((yq - yq.mean()) ** 2).sum()
    print(f"quiet-set R^2 = {ss:.3f}, spearman = {spearman(pred, yq):.3f}, "
          f"intercept = {intercept:+.0f} (dropped at emit)")
    m2 = np.abs(yq) > 2000
    sign_ok = np.mean(np.sign(pred[m2]) == np.sign(yq[m2]))
    print(f"sign agreement |y|>2000 (quiet): {sign_ok:.3f}")

    # baseline: how does the OLD (current/) table rank the same subset?
    import re
    text = open(os.path.join(ROOT_DIR, "current", "pattern_data.h")).read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      re.search(r"PATTERN_VALUES\[\]\s*=\s*\{([^}]+)\}",
                                text).group(1))
    pv_old = np.array([float(x) for x in nums])
    # old-table prediction per position over window features
    pred_old = np.zeros(quiet.sum())
    qidx = np.where(quiet)[0]
    for k, i in enumerate(qidx):
        sl = slice(offs[i], offs[i + 1])
        idx = fidx[sl]
        cnt = fcnt[sl]
        m = idx < 729
        pred_old[k] = float((pv_old[idx[m]] * cnt[m]).sum())
    print(f"OLD table on same subset: spearman = "
          f"{spearman(pred_old, yq):.3f}, sign = "
          f"{np.mean(np.sign(pred_old[m2]) == np.sign(yq[m2])):.3f}")

    full = [0.0] * 729
    for c, mem in enumerate(members):
        for i in mem:
            full[i] = float(w[c])
            full[mirror(i)] = -float(w[c])
    out = os.path.join(SCRIPT_DIR, "output", "pattern_deep.h")
    save_pattern_data_h(full, out)
    print(f"wrote {out}")
    top = np.argsort(-np.abs(w))[:8]
    for c in top:
        pat = "".join(str((reps[c] // 3**k) % 3) for k in range(6))
        print(f"  class {pat}: {w[c]:+.0f}")


if __name__ == "__main__":
    main()

"""Parity: engine mixnet path (SEAL_EVAL=mixnet) vs the python model.

Checks, per random position (laws #3/#4 — every ported net gets this
before benching):
  1. eval_position == (p_w - p_l) * 8000 from the torch model (mover POV)
  2. policy_debug(cells, for_root=True/False) == model policy with g1 = ±ml/2
  3. acc_drift: |_acc3 drift| after a real search (make/undo + rollback)

Run (hexo venv; GPU not needed):
    python mixnet_parity.py --ckpt output_mixnet_tiny/mixnet.pt \
        --blob <baked blob> --bot-dir ../../cand_mixnet
"""

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from mixnet_train import CLIP, Mixnet, batch_geometry, extract_mix

MX_OUT_SCALE = 8000.0


def oracle(model, cells, mover, cand, mc, ml, for_root=None):
    """(value_engine_units, policy list) — mover POV, fresh-eval convention
    (digit 1 = mover = root). for_root=False flips the tempo sign only."""
    codes, coords, ci = extract_mix(cells, mover, cand)
    seg_c = np.zeros(len(codes), dtype=np.int64)
    src, nbr, useg = batch_geometry(coords, seg_c, None)
    with torch.no_grad():
        a = model.cell_feats(torch.from_numpy(codes.astype(np.int64)),
                             torch.device("cpu"))
        fp, A, cnt = model.conv_pool(a, torch.from_numpy(src),
                                     torch.from_numpy(nbr),
                                     torch.from_numpy(useg), 1,
                                     torch.from_numpy(codes != 0).any(dim=1))
        g0 = torch.tensor([mc * 0.02], dtype=torch.float32)
        g1v = torch.tensor([ml * 0.5], dtype=torch.float32)
        pr = torch.softmax(model.value(A, g0, g1v), dim=1)[0]
        val = float(pr[0] - pr[1]) * MX_OUT_SCALE
        sign = 1.0 if (for_root is None or for_root) else -1.0
        g1p = torch.tensor([sign * ml * 0.5], dtype=torch.float32)
        cand_u = torch.from_numpy(src[ci.astype(np.int64)])
        seg_p = torch.zeros(len(ci), dtype=torch.int64)
        pol = model.policy(fp, A, cnt, cand_u, seg_p, g0, g1p)
    return val, [float(x) for x in pol]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_mixnet_tiny/mixnet.pt")
    ap.add_argument("--blob", required=True)
    ap.add_argument("--bot-dir", default=os.path.join(REPO, "cand_mixnet"))
    ap.add_argument("--positions", type=int, default=12)
    args = ap.parse_args()

    ck = torch.load(os.path.join(SCRIPT_DIR, args.ckpt), map_location="cpu",
                    weights_only=False)
    model = Mixnet(m=ck["M"], c=ck["C"], p=ck["P"], v=ck["V"])
    model.load_state_dict(ck["state"])
    model.eval()

    os.environ["SEAL_EVAL"] = "mixnet"
    os.environ["SEAL_MIXNET_BLOB"] = os.path.abspath(args.blob)
    sys.path.insert(0, os.path.abspath(args.bot_dir))
    sys.path.insert(0, REPO)                     # game.py for the wrapper
    import game as game_mod
    import minimax_cpp

    class Stub:
        def __init__(self, cells, mover, ml, mc):
            self.board = {(q, r): (game_mod.Player.A if p == 1
                                   else game_mod.Player.B)
                          for q, r, p in cells}
            self.current_player = (game_mod.Player.A if mover == 1
                                   else game_mod.Player.B)
            self.moves_left_in_turn = ml
            self.move_count = mc

    bot = minimax_cpp.MinimaxBot(0.05)
    rng = np.random.default_rng(11)
    worst_v = worst_p = 0.0
    for t in range(args.positions):
        n = int(rng.integers(4, 22))
        pts = set()
        while len(pts) < n + 4:
            pts.add((int(rng.integers(-7, 8)), int(rng.integers(-7, 8))))
        pts = list(pts)
        cells = [(q, r, int(rng.integers(1, 3))) for q, r in pts[:n]]
        occ = {(q, r) for q, r, _ in cells}
        cand = [c for c in pts[n:] if c not in occ][:3] + [(30, 30)]
        mover = int(rng.integers(1, 3))
        ml = int(rng.integers(1, 3))
        mc = n
        stub = Stub(cells, mover, ml, mc)

        ev = bot.eval_position(stub)
        ov, _ = oracle(model, cells, mover, cand, mc, ml)
        worst_v = max(worst_v, abs(ev - ov))

        for fr in (True, False):
            ep = list(bot.policy_debug(stub, cand, fr))
            _, op = oracle(model, cells, mover, cand, mc, ml, for_root=fr)
            mag = max(1.0, max(abs(x) for x in op))
            worst_p = max(worst_p,
                          max(abs(a - b) for a, b in zip(ep, op)) / mag)

    drift = bot.acc_drift(Stub([(0, 0, 1), (1, 0, 2), (0, 1, 1),
                                (2, 1, 2), (1, 2, 1)], 2, 2, 5), 0.1)

    print(f"parity over {args.positions} positions: "
          f"|dv| {worst_v:.4f} (of ~8000 scale), |dp|/mag {worst_p:.6f}, "
          f"acc_drift {drift:.2e}")
    ok = worst_v < 1.0 and worst_p < 1e-3 and drift < 1e-3
    print("PARITY OK" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

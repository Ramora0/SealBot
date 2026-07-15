"""Bake a trained mixnet (mixnet_train.py checkpoint) into an engine blob.

The mapping net is enumerated over all 3^11 line patterns into a codebook
(zero-anchored: row 0 is exactly zero), Rapfi-style. Heads are exported
raw float32. Blob layout (little-endian, no padding):

  magic   int32   0x4D584E31 ('MXN1')
  dims    int32×4 C, P, V, reserved(0)
  clip    float32
  codebook float32[3^11][C]
  dw      float32[7][C/2]
  star_a  float32[V][C+2] + float32[V]
  star_b  float32[V][C+2] + float32[V]
  v1      float32[V][V]   + float32[V]
  v2      float32[3][V]   + float32[3]
  pg1     float32[64][C+2] + float32[64]
  pg2     float32[P*P+P][64] + float32[P*P+P]
  pout    float32[1][P]   + float32[1]

Also verifies: table-based forward == module forward on random synthetic
positions (the parity oracle for the C++ port).

Run (hexo venv):
    python mixnet_bake.py --ckpt output_mixnet1/mixnet.pt --out current/mixnet.bin
"""

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from mixnet_train import (CLIP, HEX7, Mixnet, N_RAW, batch_geometry,
                          decode_onehot, extract_mix)

MAGIC = 0x4D584E31


def bake_codebook(model, device, chunk=16384):
    rows = []
    with torch.no_grad():
        for i in range(0, N_RAW, chunk):
            codes = torch.arange(i, min(i + chunk, N_RAW), dtype=torch.int64,
                                 device=device)
            rows.append(model.mapping(decode_onehot(codes, device)).cpu())
    tab = torch.cat(rows)
    tab = tab - tab[0:1]                     # zero anchor
    assert float(tab[0].abs().max()) == 0.0
    return tab.numpy().astype(np.float32)


def table_forward(tab, model, cells, mover, cand):
    """Reference forward using the baked table (no mapping net)."""
    codes, coords, ci = extract_mix(cells, mover, cand)
    seg_c = np.zeros(len(codes), dtype=np.int64)
    src, nbr, useg = batch_geometry(coords, seg_c, None)
    s = tab[codes.astype(np.int64)].sum(axis=1)          # [n, C]
    a = torch.from_numpy(np.clip(s, 0.0, CLIP))
    fp, A, cnt = model.conv_pool(a, torch.from_numpy(src),
                                 torch.from_numpy(nbr),
                                 torch.from_numpy(useg), 1)
    g0 = torch.tensor([0.3]); g1 = torch.tensor([0.5])
    cand_u = torch.from_numpy(src[ci.astype(np.int64)])
    seg_p = torch.zeros(len(ci), dtype=torch.int64)
    v = model.value(A, g0, g1)
    p = model.policy(fp, A, cnt, cand_u, seg_p, g0, g1)
    return v, p


def module_forward(model, cells, mover, cand):
    codes, coords, ci = extract_mix(cells, mover, cand)
    seg_c = np.zeros(len(codes), dtype=np.int64)
    src, nbr, useg = batch_geometry(coords, seg_c, None)
    with torch.no_grad():
        a = model.cell_feats(torch.from_numpy(codes.astype(np.int64)),
                             torch.device("cpu"))
        fp, A, cnt = model.conv_pool(a, torch.from_numpy(src),
                                     torch.from_numpy(nbr),
                                     torch.from_numpy(useg), 1)
        g0 = torch.tensor([0.3]); g1 = torch.tensor([0.5])
        cand_u = torch.from_numpy(src[ci.astype(np.int64)])
        seg_p = torch.zeros(len(ci), dtype=torch.int64)
        return (model.value(A, g0, g1),
                model.policy(fp, A, cnt, cand_u, seg_p, g0, g1))


def _lin(fh, lin):
    fh.write(lin.weight.detach().numpy().astype(np.float32).tobytes())
    fh.write(lin.bias.detach().numpy().astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output_mixnet1/mixnet.pt")
    ap.add_argument("--out", default=None,
                    help="default: <ckpt dir>/mixnet.bin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    ck = torch.load(os.path.join(SCRIPT_DIR, args.ckpt), map_location="cpu",
                    weights_only=False)
    model = Mixnet(m=ck["M"], c=ck["C"], p=ck["P"], v=ck["V"])
    model.load_state_dict(ck["state"])
    model.eval()
    C, P, V = ck["C"], ck["P"], ck["V"]

    dev = torch.device(args.device)
    model.to(dev)
    tab = bake_codebook(model, dev)
    model.cpu()
    print(f"codebook baked: {tab.shape}, {tab.nbytes/1e6:.1f} MB, "
          f"|max| {np.abs(tab).max():.3f}")

    # parity: table forward == module forward on synthetic positions
    rng = np.random.default_rng(7)
    worst_v = worst_p = 0.0
    for _ in range(8):
        ncell = int(rng.integers(3, 24))
        pts = set()
        while len(pts) < ncell:
            pts.add((int(rng.integers(-8, 9)), int(rng.integers(-8, 9))))
        pts = list(pts)
        cells = [(q, r, int(rng.integers(1, 3))) for q, r in pts[:-2]]
        cand = pts[-2:]
        mover = int(rng.integers(1, 3))
        v1, p1 = module_forward(model, cells, mover, cand)
        v2, p2 = table_forward(tab, model, cells, mover, cand)
        worst_v = max(worst_v, float((v1 - v2).abs().max()))
        worst_p = max(worst_p, float((p1 - p2).abs().max()))
    assert worst_v < 2e-4 and worst_p < 2e-4, (worst_v, worst_p)
    print(f"parity: table vs module | dv {worst_v:.2e} dp {worst_p:.2e}")

    out = args.out or os.path.join(os.path.dirname(
        os.path.join(SCRIPT_DIR, args.ckpt)), "mixnet.bin")
    with open(out, "wb") as fh:
        fh.write(np.array([MAGIC, C, P, V, 0], dtype=np.int32)[:1].tobytes())
        fh.write(np.array([C, P, V, 0], dtype=np.int32).tobytes())
        fh.write(np.array([CLIP], dtype=np.float32).tobytes())
        fh.write(tab.tobytes())
        fh.write(model.dw.detach().numpy().astype(np.float32).tobytes())
        for lin in (model.star_a, model.star_b, model.v1, model.v2,
                    model.pg1, model.pg2, model.pout):
            _lin(fh, lin)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Attribute the int16 quantization value error: tab-quant vs dw-quant vs
int conv rounding, measured at z (value-head input) and at the value
output (engine units). Emulates cand_mixnet64q's exact int pipeline."""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from mixnet_train import CLIP, Mixnet, batch_geometry, extract_mix

QS_TAB, QS_DW, DSH = 256, 4096, 12
FMAX = 8 << 20


def load_float_blob(path, C, P, V):
    with open(path, "rb") as f:
        f.seek(5 * 4 + 4)
        n = 3 ** 11
        tab = np.frombuffer(f.read(n * C * 4), dtype=np.float32).reshape(n, C)
        dw = np.frombuffer(f.read(7 * (C // 2) * 4),
                           dtype=np.float32).reshape(7, C // 2)
    return tab, dw


def main():
    ck = torch.load("output_ship_m128c64/mixnet.pt", map_location="cpu",
                    weights_only=False)
    model = Mixnet(m=ck["M"], c=ck["C"], p=ck["P"], v=ck["V"])
    model.load_state_dict(ck["state"])
    model.eval()
    C = ck["C"]
    C2 = C // 2

    tab_f, dw_f = load_float_blob("../../cand_mixnet64/mixnet.bin",
                                  C, ck["P"], ck["V"])
    tab_q = np.round(tab_f * QS_TAB).astype(np.int32)   # int16 range verified
    dw_q = np.round(dw_f * QS_DW).astype(np.int32)

    def z_of(tab, dw, integer):
        """Universe pooling -> z (unnormalized sum), matching the engine."""
        def run(codes, coords):
            seg = np.zeros(len(codes), dtype=np.int64)
            src, nbr, useg = batch_geometry(coords, seg, None)
            nU = nbr.shape[0]
            if integer:
                amem = tab[codes[:, 0]] + tab[codes[:, 1]] + tab[codes[:, 2]]
                amem = np.clip(amem, 0, 8 * QS_TAB)          # int16 amem
                ap = np.vstack([amem, np.zeros((1, C), dtype=amem.dtype)])
                idx = np.where(nbr >= 0, nbr, amem.shape[0])
                g = ap[idx][:, :, :C2]                       # [nU,7,C2]
                s = (dw[None, :, :] * g).sum(axis=1)         # int32
                s = np.clip(s, 0, FMAX)
                fpc = (s + (1 << (DSH - 1))) >> DSH          # scale 256
                z = np.zeros(C, dtype=np.int64)
                z[:C2] = fpc.sum(axis=0)
                # passthrough: each universe cell adds its own amem row
                # (halo rows are zero-anchored -> contribute nothing)
                own = np.full(nU, amem.shape[0], dtype=np.int64)
                own[src] = np.arange(len(src))
                z[C2:] = ap[own][:, C2:].sum(axis=0)
                return z.astype(np.float64) / QS_TAB
            amem = np.clip(tab[codes[:, 0]] + tab[codes[:, 1]]
                           + tab[codes[:, 2]], 0.0, CLIP)
            ap = np.vstack([amem, np.zeros((1, C), dtype=amem.dtype)])
            idx = np.where(nbr >= 0, nbr, amem.shape[0])
            g = ap[idx][:, :, :C2]
            s = (dw[None, :, :] * g).sum(axis=1)
            fpc = np.clip(s, 0.0, CLIP)
            z = np.zeros(C)
            z[:C2] = fpc.sum(axis=0)
            own = np.full(nbr.shape[0], amem.shape[0], dtype=np.int64)
            own[src] = np.arange(len(src))
            z[C2:] = ap[own][:, C2:].sum(axis=0)
            return z
        return run

    def value(z, mc, ml_signed):
        zz = torch.tensor(np.concatenate([z, [mc * 0.02, ml_signed * 0.5]]),
                          dtype=torch.float32)
        with torch.no_grad():
            sa = model.star_a(zz)
            sb = model.star_b(zz)
            r = torch.relu(sa * sb)
            h = torch.relu(model.v1(r))
            lg = model.v2(h)
            p = torch.softmax(lg, dim=0)
        return float((p[0] - p[1]) * 8000.0)

    rng = np.random.default_rng(23)
    rows = []
    for _ in range(12):
        ncell = int(rng.integers(6, 30))
        pts = set()
        while len(pts) < ncell:
            pts.add((int(rng.integers(-9, 10)), int(rng.integers(-9, 10))))
        pts = list(pts)
        cells = [(q, r, int(rng.integers(1, 3))) for q, r in pts]
        codes, coords, _ = extract_mix(cells, 1, [(30, 30)])
        codes = codes.astype(np.int64)

        z_ff = z_of(tab_f, dw_f, False)(codes, coords)
        z_qf = z_of(tab_q.astype(np.float64) / QS_TAB, dw_f, False)(codes, coords)
        z_qq = z_of(tab_q.astype(np.float64) / QS_TAB,
                    dw_q.astype(np.float64) / QS_DW, False)(codes, coords)
        z_ii = z_of(tab_q, dw_q, True)(codes, coords)

        v_ff = value(z_ff, ncell, 0.5)
        rows.append((np.abs(z_qf - z_ff).max(), np.abs(z_qq - z_ff).max(),
                     np.abs(z_ii - z_ff).max(),
                     abs(value(z_qf, ncell, 0.5) - v_ff),
                     abs(value(z_qq, ncell, 0.5) - v_ff),
                     abs(value(z_ii, ncell, 0.5) - v_ff),
                     np.abs(z_ff).max()))

    rows = np.array(rows)
    print("            tab-only   +dw       full-int   (|z|max ref)")
    print(f"max |dz|:   {rows[:,0].max():9.4f} {rows[:,1].max():9.4f} "
          f"{rows[:,2].max():9.4f}  {rows[:,6].max():9.1f}")
    print(f"max |dv|:   {rows[:,3].max():9.3f} {rows[:,4].max():9.3f} "
          f"{rows[:,5].max():9.3f}")


if __name__ == "__main__":
    main()

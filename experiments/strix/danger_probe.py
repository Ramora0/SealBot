"""Teacher-vs-student danger timing at provably-lost entry points.

For every recorded SealBot loss, find the earliest strix-to-move position
with a VCF-proven strix win (entry). Then, at turn-pair offsets 0..5
BEFORE entry, evaluate the same positions with (a) strix's raw net value
(single forward, no search) and (b) our v1.5 trunk value. Both mover-POV,
sign-aligned to strix POV and rescaled to [-1, 1]. If the teacher dives
negative-for-us turns before the student does, the distillation residual
holds the missing Elo; if both are flat until entry, the value channel
does not carry this signal at all.

CPU only (training owns the GPU). Run in the hexo venv:
    .../hexo-strix/.venv/bin/python danger_probe.py out.pkl in1.pkl [...]
"""

import pickle
import sys

import numpy as np
import torch

SEAL = "/users/PAS2836/leedavis/personal/SealBot"
sys.path.insert(0, SEAL)
sys.path.insert(0, SEAL + "/current")
sys.path.insert(0, SEAL + "/experiments/strix")

from game import HexGame, Player
import minimax_cpp
from strix_bridge import load_strix, state_from_cells, value_batch
from trunk_train2 import Trunk2
from trunk_train import extract

MAXK = 16


def replay(seq):
    game = HexGame(win_length=6)
    out = []
    for (q, r, pl) in seq:
        cells = [(qq, rr, p.value) for (qq, rr), p in game.board.items()]
        out.append((cells, game.current_player.value,
                    game.moves_left_in_turn, game.move_count))
        if game.game_over:
            break
        game.make_move(q, r)
    return out


def main():
    out_path = sys.argv[1]
    games = []
    for p in sys.argv[2:]:
        games += pickle.load(open(p, "rb"))
    losses = [g for g in games if not g["sealbot_won"]]
    print(f"{len(losses)} losses across {len(sys.argv) - 2} pkls")

    bot = minimax_cpp.MinimaxBot(0.1)
    bot.vcf_node_budget = 60000

    ck = torch.load(SEAL + "/experiments/strix/output_trunk5/trunk.pt",
                    map_location="cpu")
    student = Trunk2(k=ck.get("K", 32), h=ck.get("H", 32),
                     hp=ck.get("HP", 32))
    student.load_state_dict(ck["state"])
    student.eval()

    model, mc, gc = load_strix(device="cpu")

    def student_val(cells, mover, ml, mcnt):
        trip, wi, wc, _ = extract([tuple(c) for c in cells], mover,
                                  [tuple(cells[0][:2])])
        t = torch.from_numpy(trip.astype(np.int64))
        w = torch.from_numpy(wi.astype(np.int64))
        c = torch.from_numpy(wc.astype(np.float32))
        off = torch.tensor([0, len(w)])
        seg = torch.zeros(len(t), dtype=torch.int64)
        g0 = torch.tensor([mcnt * 0.02])
        g1 = torch.tensor([ml * 0.5])
        with torch.no_grad():
            return float(student.value(t, seg, 1, w, c, off, g0, g1)) / 8.0

    # entry detection + probe positions
    probes = {k: {"teacher": [], "student": []} for k in range(6)}
    n_entry = 0
    for li, g in enumerate(losses):
        seq = g["seq"]
        strix_val = 1 if g["hexo_is_a"] else 2
        pos = replay(seq)
        entry_i = None
        for i, (cells, mover, ml, mcnt) in enumerate(pos):
            if mover != strix_val or mcnt < 8 or ml != 2:
                continue
            game = HexGame(win_length=6)
            for q, r, p in cells:
                game.board[(q, r)] = Player(p)
            game.current_player = Player(mover)
            game.moves_left_in_turn = ml
            game.move_count = mcnt
            res, _ = bot.forced_win(game, MAXK)
            if res == 1:
                entry_i = i
                break
        if entry_i is None:
            continue
        n_entry += 1
        # strix-to-move turn starts at entry and 1..5 turn-pairs earlier
        strix_turns = [i for i, (c, m, ml, mc) in enumerate(pos)
                       if m == strix_val and ml == 2]
        at = strix_turns.index(entry_i)
        for off in range(6):
            j = at - off
            if j < 0:
                continue
            cells, mover, ml, mcnt = pos[strix_turns[j]]
            s = state_from_cells(cells, mover, ml, gc)
            if s is None:
                continue
            tv = value_batch(model, mc, [s], device="cpu")[0]
            sv = student_val(cells, mover, ml, mcnt)
            # mover is strix here; mover-POV == strix-POV
            probes[off]["teacher"].append(float(tv))
            probes[off]["student"].append(float(sv))
        if (li + 1) % 20 == 0:
            print(f"  {li+1}/{len(losses)} losses, {n_entry} entries",
                  flush=True)

    print(f"\nentries found: {n_entry}")
    print("turns-before-entry  teacher(strix POV)  student  n")
    for off in range(6):
        t = probes[off]["teacher"]; s = probes[off]["student"]
        if t:
            print(f"  -{off}: {np.mean(t):+.3f} (sd {np.std(t):.2f})   "
                  f"{np.mean(s):+.3f} (sd {np.std(s):.2f})   {len(t)}")
    with open(out_path, "wb") as fh:
        pickle.dump(probes, fh)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

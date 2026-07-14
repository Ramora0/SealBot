"""Decompose CONSTRAINT blunders: which generous parameter binds?

For each CONSTRAINT loss from blunder_autopsy, re-derive the last
decision point P and ask four engine configs for a move; a config
"saves" if its chosen pair leaves strix without a proven win (k=16/100k).

  A  tl 0.44  k=11  40k    (in-game repro — expect blunder)
  B  tl 0.44  k=16  200k   (proof depth only)
  C  tl 2.00  k=11  40k    (clock only)
  D  tl 2.00  k=16  200k   (both — autopsy's generous, expect save)

Run in SealBot venv:
    ../../.venv/bin/python constraint_decomp.py autopsy40.pkl \
        bench_t5_open150.json.games.pkl bench_t5_150.json.games.pkl
"""

import os
import pickle
import sys

os.environ.setdefault("SEAL_EVAL", "trunk")
os.environ.setdefault("SEAL_TRUNK_POLICY", "1")
os.environ.setdefault("SEAL_TRUNK_BLEND", "0")
os.environ.setdefault("SEAL_POLICY_MODE", "74")
os.environ["SEAL_VCF"] = "15"

SEAL = "/users/PAS2836/leedavis/personal/SealBot"
sys.path.insert(0, SEAL)
sys.path.insert(0, SEAL + "/current")

import minimax_cpp
from blunder_autopsy import mk_game, replay, pair_saves

MAXK = 16


def bot(tl, k, budget):
    os.environ["SEAL_VCF_K"] = str(k)
    os.environ["SEAL_VCF_BUDGET"] = str(budget)
    return minimax_cpp.MinimaxBot(tl)


def main():
    autopsy = pickle.load(open(sys.argv[1], "rb"))
    games = []
    for p in sys.argv[2:]:
        games += pickle.load(open(p, "rb"))
    losses = [g for g in games if not g["sealbot_won"]][:len(autopsy)]

    probe = minimax_cpp.MinimaxBot(0.1)
    probe.vcf_node_budget = 100000
    cfgs = {"A_ingame": bot(0.44, 11, 40000),
            "B_deepk": bot(0.44, 16, 200000),
            "C_clock": bot(2.0, 11, 40000),
            "D_both": bot(2.0, 16, 200000)}

    saves = {k: 0 for k in cfgs}
    n = 0
    for rec, g in zip(autopsy, losses):
        if rec.get("class") != "CONSTRAINT":
            continue
        seq = g["seq"]
        strix_val = 1 if g["hexo_is_a"] else 2
        seal_val = 2 if g["hexo_is_a"] else 1
        pos = replay(seq)
        entry_i = None
        for i, (cells, mover, ml, mc) in enumerate(pos):
            if mover != strix_val or mc < 8 or ml != 2:
                continue
            res, _ = probe.forced_win(mk_game(cells, mover, ml, mc), MAXK)
            if res == 1:
                entry_i = i
                break
        if entry_i is None:
            continue
        seal_turns = [i for i, (c, m, ml_, mc_) in enumerate(pos)
                      if m == seal_val and ml_ == 2 and i < entry_i]
        pi = seal_turns[-1 - rec["back"]]
        cells, mover, ml, mc = pos[pi]
        n += 1
        for name, b in cfgs.items():
            gm = b.get_move(mk_game(cells, mover, ml, mc))
            pair = tuple(map(tuple, gm)) if b.pair_moves else None
            if pair and pair_saves(probe, cells, mover, ml, mc, pair):
                saves[name] += 1
        if n % 5 == 0:
            print(f"[{n}] " + "  ".join(f"{k}:{v}" for k, v in saves.items()),
                  flush=True)
    print(f"\nFINAL over {n} CONSTRAINT positions:")
    for k, v in saves.items():
        print(f"  {k}: {v}/{n}")


if __name__ == "__main__":
    main()

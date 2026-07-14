"""Blunder autopsy: find the LAST AVOIDABLE decision in each loss and
classify WHICH component failed there.

For each recorded loss:
  1. entry = first strix-to-move turn-start with a VCF-proven strix win
     (k<=16, 100k nodes).
  2. P = SealBot's turn-start immediately before entry; M = the pair it
     actually played (which provably loses).
  3. Alternative pool at P: all pairs from the top-14 policy cells, plus
     pairs built around strix's proven winning reply cells. A pair SAVES
     if after it strix has NO proven win at k=16/100k.
  4. If nothing saves, step back one seal turn and repeat (max 2 back).
  5. Classify:
       DEAD_EARLY    no saving pair exists up to 2 turns before entry —
                     the real mistake was upstream, tactically invisible
       CONSTRAINT    a generous engine (tl 2.0, k=16, 200k) finds a
                     saving move — in-game clock/k/budget is the binder
       EVAL_PREF     saving cells are ranked high by policy (both in
                     top-20) but search still prefers the losing move
       POLICY_MISS   saving cells ranked poorly (either outside top-20)
                     — the save is effectively invisible to ordering

Champion env is set inside. CPU only. Run in SealBot venv:
    ../../.venv/bin/python blunder_autopsy.py out.pkl n_losses in1.pkl [...]
"""

import itertools
import os
import pickle
import sys

os.environ.setdefault("SEAL_EVAL", "trunk")
os.environ.setdefault("SEAL_TRUNK_POLICY", "1")
os.environ.setdefault("SEAL_TRUNK_BLEND", "0")
os.environ.setdefault("SEAL_POLICY_MODE", "74")

SEAL = "/users/PAS2836/leedavis/personal/SealBot"
sys.path.insert(0, SEAL)
sys.path.insert(0, SEAL + "/current")

from game import HexGame, Player
import minimax_cpp

MAXK = 16
D2 = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
      if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)]


def mk_game(cells, mover, ml, mc):
    g = HexGame(win_length=6)
    for q, r, p in cells:
        g.board[(q, r)] = Player(p)
    g.current_player = Player(mover)
    g.moves_left_in_turn = ml
    g.move_count = mc
    return g


def cand_cells(cells):
    occ = {(q, r) for q, r, _ in cells}
    out = set()
    for q, r, _ in cells:
        for dq, dr in D2:
            c = (q + dq, r + dr)
            if c not in occ:
                out.add(c)
    return sorted(out)


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


def pair_saves(probe, cells, mover, ml, mc, pair):
    g = mk_game(cells, mover, ml, mc)
    for (q, r) in pair:
        if g.game_over:
            return True          # we win with this pair outright
        if (q, r) in g.board:
            return None
        g.make_move(q, r)
    if g.game_over:
        return g.winner == Player(mover)
    res, _ = probe.forced_win(g, MAXK)
    return res != 1


def main():
    out_path = sys.argv[1]
    n_max = int(sys.argv[2])
    games = []
    for p in sys.argv[3:]:
        games += pickle.load(open(p, "rb"))
    losses = [g for g in games if not g["sealbot_won"]][:n_max]
    print(f"analyzing {len(losses)} losses", flush=True)

    probe = minimax_cpp.MinimaxBot(0.1)
    probe.vcf_node_budget = 100000
    ranker = minimax_cpp.MinimaxBot(0.1)

    os.environ["SEAL_VCF"] = "15"
    os.environ["SEAL_VCF_K"] = "16"
    os.environ["SEAL_VCF_BUDGET"] = "200000"
    generous = minimax_cpp.MinimaxBot(2.0)

    results = []
    for li, g in enumerate(losses):
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
            results.append({"class": "NO_ENTRY"})
            continue

        # strix's proven winning first pair right at entry (defense hints)
        cells, mover, ml, mc = pos[entry_i]
        _, wline = probe.forced_win(mk_game(cells, mover, ml, mc), MAXK)
        whint = [tuple(m) for m in wline[:2]] if wline else []

        seal_turns = [i for i, (c, m, ml_, mc_) in enumerate(pos)
                      if m == seal_val and ml_ == 2 and i < entry_i]
        rec = {"class": "DEAD_EARLY", "game_idx": g.get("game_idx"),
               "entry_mc": pos[entry_i][3], "back": None}
        for back, pi in enumerate(reversed(seal_turns[-2:])):
            cells, mover, ml, mc = pos[pi]
            cand = cand_cells(cells)
            lg = ranker.policy_debug(mk_game(cells, mover, ml, mc),
                                     cand, True)
            ranked = [c for c, _ in sorted(zip(cand, lg),
                                           key=lambda t: -t[1])]
            top = ranked[:14]
            pool = list(itertools.combinations(top, 2))
            for w in whint:
                for t in ranked[:8]:
                    if w != t and (w, t) not in pool and (t, w) not in pool:
                        pool.append((w, t))
            if len(whint) == 2 and tuple(whint) not in pool:
                pool.append(tuple(whint))
            saving = []
            for pair in pool:
                ok = pair_saves(probe, cells, mover, ml, mc, pair)
                if ok:
                    saving.append(pair)
            if not saving:
                continue
            rank_of = {c: i for i, c in enumerate(ranked)}
            best_rank = min(max(rank_of.get(a, 99), rank_of.get(b, 99))
                            for a, b in saving)
            gm = generous.get_move(mk_game(cells, mover, ml, mc))
            gpair = tuple(map(tuple, gm)) if generous.pair_moves else None
            gen_saves = (gpair is not None and
                         pair_saves(probe, cells, mover, ml, mc, gpair))
            if gen_saves:
                cls = "CONSTRAINT"
            elif best_rank < 20:
                cls = "EVAL_PREF"
            else:
                cls = "POLICY_MISS"
            rec.update({"class": cls, "back": back,
                        "n_saving": len(saving), "best_rank": best_rank,
                        "gen_saves": bool(gen_saves)})
            break
        results.append(rec)
        if (li + 1) % 5 == 0:
            from collections import Counter
            print(f"[{li+1}/{len(losses)}] "
                  f"{Counter(r['class'] for r in results)}", flush=True)

    from collections import Counter
    print("\nFINAL:", Counter(r["class"] for r in results))
    backs = Counter(r["back"] for r in results if r["back"] is not None)
    print("found at turns-back:", dict(backs))
    ranks = [r["best_rank"] for r in results if r.get("best_rank") is not None]
    if ranks:
        import statistics
        print(f"saving-pair worst-cell rank: median "
              f"{statistics.median(ranks)}, ranks {sorted(ranks)[:20]}")
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

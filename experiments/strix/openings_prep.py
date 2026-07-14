"""Sample early human-game positions -> openings_human.pkl for paired
benches (each opening played twice, colors swapped).

Openings are turn-boundary positions with 5-9 stones, translated so P1's
first stone is at (0,0) (hexo frame), stored as legal move sequences
[(q, r, player), ...] replayable by both HexGame and hexo GameState.

    ../../.venv/bin/python openings_prep.py
"""

import json
import os
import pickle
import random

import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET = "/users/PAS2836/leedavis/personal/KrakenBot/distill_100k.parquet"
OUT = os.path.join(SCRIPT_DIR, "openings_human.pkl")
N_OPENINGS = 200


def reconstruct(cells, mover):
    """cells: [(q,r,p)] -> legal move sequence ending at a turn boundary
    with `mover` to play, or None if counts don't line up."""
    p1 = [(q, r) for q, r, p in cells if p == 1]
    p2 = [(q, r) for q, r, p in cells if p == 2]
    if not p1:
        return None

    def hexd(a, b):
        dq, dr = a[0] - b[0], a[1] - b[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    # anchor = P1 stone minimizing the max radius after translation
    # (hexo board is bounded; a corner anchor pushes stones out of range)
    allst = p1 + p2
    anchor = min(p1, key=lambda a: max(hexd(a, b) for b in allst))
    p1.sort(key=lambda a: (a != anchor,))
    oq, orr = p1[0]
    p1 = [(q - oq, r - orr) for q, r in p1]
    p2 = [(q - oq, r - orr) for q, r in p2]
    seq = [(p1[0][0], p1[0][1], 1)]
    i1, i2, turn = 1, 0, 2
    while i1 < len(p1) or i2 < len(p2):
        if turn == 2:
            take = p2[i2:i2 + 2]
            if len(take) < 2:
                return None
            seq += [(q, r, 2) for q, r in take]
            i2 += 2
            turn = 1
        else:
            take = p1[i1:i1 + 2]
            if len(take) < 2:
                return None
            seq += [(q, r, 1) for q, r in take]
            i1 += 2
            turn = 2
    if turn != mover:
        return None
    return seq


def main():
    t = pq.read_table(PARQUET,
                      columns=["board", "current_player", "game_id"])
    boards = t["board"].to_pylist()
    movers = t["current_player"].to_pylist()
    rng = random.Random(7)
    idx = list(range(len(boards)))
    rng.shuffle(idx)
    out, seen = [], set()
    for i in idx:
        d = json.loads(boards[i])
        if not (5 <= len(d) <= 9):
            continue
        cells = [(int(k.split(",")[0]), int(k.split(",")[1]), int(v))
                 for k, v in d.items()]
        seq = reconstruct(cells, int(movers[i]))
        if seq is None:
            continue
        key = tuple(sorted((q, r, p) for q, r, p in seq))
        if key in seen:
            continue
        seen.add(key)
        out.append(seq)
        if len(out) >= N_OPENINGS:
            break
    with open(OUT, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    from collections import Counter
    print(f"{len(out)} openings -> {OUT}")
    print("stones per opening:", Counter(len(s) for s in out))


if __name__ == "__main__":
    main()

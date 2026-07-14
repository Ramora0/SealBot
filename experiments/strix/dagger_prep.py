"""DAgger prep: positions from recorded strix benches -> human_prep-style
shards for strix labeling (reuse human_extract --stage label semantics by
writing into dagger_prep/, labeled into dagger_targets/ via env overrides).

    ../../.venv/bin/python dagger_prep.py
"""

import glob
import os
import pickle
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEAL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SEAL)

from game import HexGame, Player

OUT = os.path.join(SCRIPT_DIR, "dagger_prep")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(3)
    recs, seen = [], set()
    for pkl_path in sorted(glob.glob(os.path.join(SCRIPT_DIR,
                                                  "*.games.pkl"))):
        for g in pickle.load(open(pkl_path, "rb")):
            game = HexGame(win_length=6)
            for (q, r, pl) in g["seq"]:
                if game.game_over:
                    break
                if game.move_count >= 6 and game.moves_left_in_turn == 2:
                    cells = tuple(sorted((qq, rr, p.value)
                                         for (qq, rr), p in
                                         game.board.items()))
                    if cells not in seen:
                        seen.add(cells)
                        recs.append(([list(c) for c in cells],
                                     game.current_player.value, 2,
                                     game.move_count))
                game.make_move(q, r)
    rng.shuffle(recs)
    print(f"{len(recs)} unique positions from recorded strix games")
    for s in range(0, len(recs), 4000):
        with open(os.path.join(OUT, f"prep_{s // 4000:04d}.pkl"), "wb") as fh:
            pickle.dump([(c, m, ml, mc) for c, m, ml, mc in
                         recs[s:s + 4000]], fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    print("shards written")


if __name__ == "__main__":
    main()

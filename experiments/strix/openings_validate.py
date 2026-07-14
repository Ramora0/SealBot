"""Validate openings_human.pkl against hexo_rs board bounds: replay each
opening into a fresh GameState, keep only clean ones. Run in hexo venv.

    /users/PAS2836/leedavis/personal/hexo-strix/.venv/bin/python \
        openings_validate.py
"""

import pickle
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

sys.path.insert(0, "/users/PAS2836/leedavis/personal/hexo-strix")
import hexo_rs

from strix_bridge import CKPT  # noqa: E402


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    gc = hexo_rs.GameConfig(**ck["game_config"])
    src = SCRIPT_DIR / "openings_human.pkl"
    openings = pickle.load(open(src, "rb"))
    good, bad = [], 0
    for seq in openings:
        try:
            gs = hexo_rs.GameState(gc)
            for (q, r, pl) in seq[1:]:
                gs.apply_move(q, r)
            if gs.is_terminal():
                bad += 1
                continue
            good.append(seq)
        except ValueError:
            bad += 1
    with open(src, "wb") as fh:
        pickle.dump(good, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"kept {len(good)}, dropped {bad}")


if __name__ == "__main__":
    main()

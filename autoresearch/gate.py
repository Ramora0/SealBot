"""Gate one SealBot candidate against strix under the pinned protocol.

Wraps experiments/strix/bench_vs_strix.py with the champion env, machine
profile, and opening sets, so an experiment is exactly one command:

    python gate.py --name ctrl                                   # baseline
    python gate.py --name vcf_fk8 --env SEAL_VCF_FK=8 SEAL_VCF_FB=20000
    python gate.py --name lmr --bot-dir cand
    python gate.py --name lmr --bot-dir cand --games 200         # borderline extension
    python gate.py --name champ_transfer --held                  # openings 50-74

EVERY candidate takes the identical gate: strix sims=64 on dev openings
0-49 (100 games, paired colors) at the epoch tl (epoch.json) — there is
no cheaper screen and no per-type protocol. Decision: any positive delta
vs ctrl adopts, anything else discards. --held swaps in openings 50-74
(50 games) for transfer checks. --games overrides the count if ever
needed (dev openings are replayed, never the held-out set). Reports
seal's Elo vs strix with a 95% CI (from the Wilson win-rate bounds).

Output JSON -> results/<name>_s64[...].json (skipped if it already exists
— delete it or pass --force to re-run). Seal wins = strix_losses. Prints
a delta vs results/ctrl_s64[...].json plus a ready results.tsv row. Run
with the hexo-strix venv python. Never run two gates concurrently: seal's
clock is wall-time and the box has 8 physical cores.
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCH = REPO / "experiments" / "strix" / "bench_vs_strix.py"
OPENINGS = REPO / "experiments" / "strix" / "openings_human.pkl"
RESULTS = HERE / "results"

# ── Machine profile (this box: 5800X3D 8C/16T, RTX 4080, MSVC) ──────────
# The 35/100 anchor was measured on OSC Pitzer (20 cores, V100, icpc) with
# SEAL_THREADS=20, serial games. Thread count AND pipeline count are pinned
# INTO results (law #6): never compare runs across different values.
# Pipeline 2 = two phase-offset games, one seal search at a time (seal
# mutex), strix GPU work fills seal's think gaps (~1.8x throughput).
THREADS = os.environ.get("SEAL_THREADS", "16")
PIPELINE = os.environ.get("GATE_PIPELINE", "2")
STRIX_ROOT = Path(os.environ.get("STRIX_ROOT", REPO.parent / "hexo-strix"))
_CKPT_CANDIDATES = [
    os.environ.get("STRIX_CKPT"),
    STRIX_ROOT / "checkpoint_00237000.pt",
    Path.home() / "OneDrive/Desktop/checkpoint_00237000.pt",
]

# ── Protocol pins (experiments/strix/EVAL_SCHEME.md — do not change) ────
SIMS = 64          # real strength; weaker screens are not a valid gate
M_ACTIONS = 16

# ── Epoch (agent-owned): seal's tl is tuned to hold ctrl near 50% for
# maximum Elo resolution. Edit epoch.json (bump epoch, set tl), then
# re-run ctrl. Filenames are stamped _e<N> so epochs can never mix. ────
_EPOCH = json.loads((HERE / "epoch.json").read_text())
EPOCH, TL = int(_EPOCH["epoch"]), float(_EPOCH["tl"])
CHAMPION_ENV = {
    "SEAL_EVAL": "trunk",
    "SEAL_TRUNK_POLICY": "1",
    "SEAL_TRUNK_BLEND": "0",
    "SEAL_POLICY_MODE": "74",
    "SEAL_VCF": "15",
    "SEAL_VCF_K": "11",
    "SEAL_VCF_BUDGET": "40000",
    "SEAL_SMP_MODE": "2",     # root-split YBW
}


def _ckpt() -> Path:
    for c in _CKPT_CANDIDATES:
        if c and Path(c).exists():
            return Path(c)
    sys.exit("strix checkpoint_00237000.pt not found; set STRIX_CKPT")


def _openings_pkl(held: bool) -> Path:
    """Dev (0-49) or held-out (50-74) slice of openings_human.pkl.

    Always sliced so game counts beyond one pass REPLAY the set (bench
    plays opening i//2 % len) instead of walking into other openings —
    a 200-game dev extension must never touch the held-out set.
    """
    lo, hi, tag = (50, 75, "held") if held else (0, 50, "dev")
    out = RESULTS / f"_{tag}_openings.pkl"
    if not out.exists():
        with open(OPENINGS, "rb") as fh:
            ops = pickle.load(fh)
        with open(out, "wb") as fh:
            pickle.dump(ops[lo:hi], fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def two_prop_z(w1: int, n1: int, w2: int, n2: int) -> float:
    """z for H0: same seal win rate (candidate 1 vs ctrl 2)."""
    if min(n1, n2) == 0:
        return 0.0
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    return (w1 / n1 - w2 / n2) / se


def elo(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 400 * math.log10(p / (1 - p))


def seal_stats(r: dict):
    """Seal-perspective (rate, ci_lo, ci_hi) from a bench summary JSON."""
    return (1 - r["strix_win_rate"], 1 - r["ci"][1], 1 - r["ci"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="candidate name (ctrl = baseline)")
    ap.add_argument("--bot-dir", default="current")
    ap.add_argument("--env", nargs="*", default=[], metavar="K=V",
                    help="candidate env overrides on top of champion env")
    ap.add_argument("--games", type=int, default=None,
                    help="default 100 (50 for --held); 200 for simplification gates / extensions")
    ap.add_argument("--held", action="store_true",
                    help="transfer check on held-out openings 50-74")
    ap.add_argument("--record", action="store_true", help="dump game sequences for autopsy")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    default_games = 50 if args.held else 100
    games = args.games or default_games
    suffix = f"_e{EPOCH}_s{SIMS}" + ("_held" if args.held else "")
    ctrl_suffix = suffix  # ctrl is always the standard game count
    if games != default_games:
        suffix += f"_g{games}"
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}{suffix}.json"
    if out.exists() and out.stat().st_size > 0 and not args.force:
        print(f"exists, skipping run: {out}")
    else:
        openings = _openings_pkl(args.held)

        bot_dir = Path(args.bot_dir)
        if not bot_dir.is_absolute():
            bot_dir = REPO / bot_dir
        blob = bot_dir / "trunk_eraw.bin"
        if not blob.exists():
            blob = REPO / "current" / "trunk_eraw.bin"

        env = dict(os.environ)
        env.update(CHAMPION_ENV)
        env.update({
            "SEAL_THREADS": THREADS,
            "SEAL_TRUNK_BLOB": str(blob),
            "SEALBOT_ROOT": str(REPO),
            "STRIX_ROOT": str(STRIX_ROOT),
            "STRIX_CKPT": str(_ckpt()),
        })
        overrides = {}
        for kv in args.env:
            k, _, v = kv.partition("=")
            if not _:
                sys.exit(f"bad --env entry (want K=V): {kv}")
            overrides[k] = v
        env.update(overrides)

        cmd = [sys.executable, str(BENCH),
               "--bot-dir", str(bot_dir), "--games", str(games),
               "--tl", str(TL), "--sims", str(SIMS),
               "--m-actions", str(M_ACTIONS), "--pipeline", PIPELINE,
               "--openings", str(openings), "--out", str(out)]
        if args.record:
            cmd.append("--record")

        print(f"gate {args.name}: s{SIMS} x {games}g "
              f"({'held-out 50-74' if args.held else 'dev 0-49'}), "
              f"bot_dir={bot_dir.name}, threads={THREADS}, "
              f"pipeline={PIPELINE}, env overrides={overrides or 'none'}")
        if args.dry_run:
            print(" ".join(cmd))
            return
        subprocess.run(cmd, env=env, check=True)

    r = json.loads(out.read_text())
    seal_w, n = r["strix_losses"], r["games"]
    p, lo, hi = seal_stats(r)
    elo_s = f"{elo(p):+.0f} [{elo(lo):+.0f}, {elo(hi):+.0f}]"
    line = (f"{args.name}{suffix}: seal {seal_w}/{n} ({p:.1%}), "
            f"Elo {elo_s} vs strix")

    ctrl_path = RESULTS / f"ctrl{ctrl_suffix}.json"
    delta_s, z = "", ""
    if ctrl_path.exists() and args.name != "ctrl":
        c = json.loads(ctrl_path.read_text())
        cw, cn = c["strix_losses"], c["games"]
        cp, _, _ = seal_stats(c)
        zval = two_prop_z(seal_w, n, cw, cn)
        delta_s = f"{100 * (seal_w / n - cw / cn):+.0f}"
        z = f"{zval:+.2f}"
        line += (f"\n  vs ctrl {cw}/{cn} ({cp:.1%}): delta {delta_s} pts/100, "
                 f"delta Elo {elo(p) - elo(cp):+.0f}, z {z}"
                 f"  ->  {'ADOPT' if seal_w / n > cw / cn else 'DISCARD'}")
    print(line)
    print("results.tsv row:")
    print(f"{args.name}\t{args.bot_dir}\t{' '.join(args.env) or '-'}\t"
          f"e{EPOCH}\t{seal_w}/{n}\t{elo_s}\t{delta_s or '-'}\t{z or '-'}\t"
          f"<keep|discard>\t<description>")


if __name__ == "__main__":
    main()

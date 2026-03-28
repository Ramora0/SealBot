"""CMA-ES optimization of SealBot pattern evaluation weights.

Optimizes the 364 free parameters (from 729 total, halved by player-swap
symmetry) using CMA-ES. Fitness is measured by win rate against the
baseline (hardcoded best/ weights) over a batch of games.

Usage:
    python optimize.py                     # defaults
    python optimize.py --games 30 --popsize 80
    python optimize.py --resume            # resume from checkpoint
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

import cma
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pkl")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from symmetry import free_to_full, full_to_free, load_baseline, save_pattern_data_h


# ── Statistics (mirrors evaluate.py) ────────────────────────────────────────

def _norm_sf(x):
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    return poly * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _elo(score):
    if score <= 0.0: return float('-inf')
    if score >= 1.0: return float('inf')
    return -400 * math.log10(1.0 / score - 1.0)


def win_rate_stats(wins, losses, draws):
    """Wilson CI, p-value, and Elo from W/L/D counts."""
    n = wins + losses + draws
    if n == 0:
        return {"wr": 0.5, "ci_lo": 0.0, "ci_hi": 1.0, "p": 1.0, "elo": 0, "n": 0}
    score = wins + 0.5 * draws
    p_hat = score / n
    z = 1.96
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    ci_lo = max(0.0, centre - spread)
    ci_hi = min(1.0, centre + spread)
    z_obs = (score - 0.5 * n) / math.sqrt(0.25 * n) if n > 0 else 0
    p_value = 2 * _norm_sf(abs(z_obs)) if n > 0 else 1.0
    return {"wr": p_hat, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p": p_value, "elo": _elo(p_hat), "n": n}


# ── Global config (set by main, read by workers) ───────────────────────────

_CFG = {}


def _init_worker(cfg):
    """Initializer for each pool worker -- import the C++ module once."""
    global _CFG
    _CFG = cfg
    # Import here so each worker process has its own module instance
    sys.path.insert(0, cfg["script_dir"])
    sys.path.insert(0, cfg["root_dir"])


def _evaluate_one(free_params):
    """Fitness function: play games, return (neg_win_rate, wins, losses, draws).

    CMA-ES minimizes, so better candidates have more negative fitness.
    """
    from game import Player
    import cma_minimax_cpp

    num_games = _CFG["num_games"]
    time_limit = _CFG["time_limit"]

    full_params = free_to_full(free_params)

    wins, losses, draws = 0, 0, 0
    for game_idx in range(num_games):
        swapped = game_idx % 2 == 1

        # Fresh bots each game (clean transposition tables etc.)
        candidate = cma_minimax_cpp.MinimaxBot(time_limit)
        candidate.load_patterns(full_params)
        baseline = cma_minimax_cpp.MinimaxBot(time_limit)

        if swapped:
            bot_a, bot_b = baseline, candidate
        else:
            bot_a, bot_b = candidate, baseline

        try:
            winner = _play_game(bot_a, bot_b, time_limit)
        except Exception:
            winner = Player.NONE

        # Score from candidate's perspective
        candidate_is = Player.A if not swapped else Player.B
        if winner == candidate_is:
            wins += 1
        elif winner == Player.NONE:
            draws += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / num_games
    return (-win_rate, wins, losses, draws)


def _evaluate_one_with_cfg(free_params, cfg):
    """Like _evaluate_one but takes explicit cfg (for validation calls)."""
    global _CFG
    _CFG = cfg
    sys.path.insert(0, cfg["script_dir"])
    sys.path.insert(0, cfg["root_dir"])
    return _evaluate_one(free_params)


def _play_game(bot_a, bot_b, time_limit, max_moves=200):
    """Play one game between two bot engines. Returns the winner."""
    from game import HexGame, Player

    game = HexGame(win_length=6)
    bots = {Player.A: bot_a, Player.B: bot_b}
    total = 0

    while not game.game_over and total < max_moves:
        player = game.current_player
        bot = bots[player]
        bot.time_limit = time_limit
        moves = bot.get_move(game)

        if not moves:
            return Player.B if player == Player.A else Player.A

        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                return Player.B if player == Player.A else Player.A
        total += len(moves)

    return game.winner


# ── Main optimization loop ─────────────────────────────────────────────────

def run(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load starting point
    baseline_full = load_baseline()
    x0 = full_to_free(baseline_full)
    print(f"Free parameters: {len(x0)} (from {len(baseline_full)} total)")
    print(f"Parameter stats: median |x|={np.median(np.abs(x0)):.0f}, "
          f"mean |x|={np.mean(np.abs(x0)):.0f}, max |x|={np.max(np.abs(x0)):.0f}")

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["popsize"] = args.popsize
    opts["maxiter"] = args.max_gen
    opts["seed"] = args.seed
    opts["verb_disp"] = 1
    opts["verb_filenameprefix"] = os.path.join(OUTPUT_DIR, "outcma_")
    opts["verb_log"] = 1
    opts["tolfun"] = 1e-6

    # Resume or start fresh
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        es = state["es"]
        best_fitness = state["best_fitness"]
        best_free = state["best_free"]
        gen_offset = state["generation"]
        print(f"  Resuming at generation {gen_offset}, best fitness {best_fitness:.4f}")
    else:
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)
        best_fitness = 0.0   # worst possible (win rate = 0)
        best_free = x0.copy()
        gen_offset = 0

    # Worker config
    cfg = {
        "num_games": args.games,
        "time_limit": args.time_limit,
        "script_dir": SCRIPT_DIR,
        "root_dir": ROOT_DIR,
    }

    num_workers = args.workers or mp.cpu_count()
    print(f"\nCMA-ES: sigma0={args.sigma0}, popsize={args.popsize}, "
          f"games/eval={args.games}, time_limit={args.time_limit}s")
    print(f"Workers: {num_workers}, max generations: {args.max_gen}")
    print(f"Validate every {args.validate_every} gens with {args.validate_games} games")
    print(f"Output: {OUTPUT_DIR}\n")

    # CSV log
    csv_path = os.path.join(OUTPUT_DIR, "log.csv")
    csv_fields = ["gen", "best_wr", "gen_best_wr", "mean_wr",
                  "sigma", "gen_time", "total_time",
                  "val_wr", "val_ci_lo", "val_ci_hi", "val_elo", "val_p", "val_n"]
    write_header = not (args.resume and os.path.exists(csv_path))
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        csv_writer.writeheader()

    # Cumulative validation stats for best-so-far
    val_total = {"wins": 0, "losses": 0, "draws": 0}

    gen = gen_offset
    t_start = time.time()

    try:
        while not es.stop():
            gen += 1
            t_gen = time.time()

            # Ask for candidate solutions
            solutions = es.ask()

            # Evaluate in parallel -- returns (neg_wr, wins, losses, draws)
            with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                results = pool.map(_evaluate_one, solutions)

            fitnesses = [r[0] for r in results]
            es.tell(solutions, fitnesses)

            # Track best
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            best_changed = False
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_free = np.array(solutions[gen_best_idx])
                best_changed = True
                # Reset cumulative validation when best changes
                val_total = {"wins": 0, "losses": 0, "draws": 0}

            elapsed_gen = time.time() - t_gen
            elapsed_total = time.time() - t_start

            # Per-generation summary
            gen_w = sum(r[1] for r in results)
            gen_l = sum(r[2] for r in results)
            gen_d = sum(r[3] for r in results)
            gen_stats = win_rate_stats(gen_w, gen_l, gen_d)

            print(f"  Gen {gen}: best_wr={-best_fitness:.1%}, "
                  f"gen_best_wr={-gen_best_fit:.1%}, "
                  f"pop: {gen_w}W/{gen_l}L/{gen_d}D "
                  f"(elo {gen_stats['elo']:+.0f}), "
                  f"sigma={es.sigma:.1f}, "
                  f"{elapsed_gen:.0f}s"
                  f"{'  *new best*' if best_changed else ''}")

            # CSV row (validation fields filled below if applicable)
            row = {
                "gen": gen, "best_wr": f"{-best_fitness:.4f}",
                "gen_best_wr": f"{-gen_best_fit:.4f}",
                "mean_wr": f"{-np.mean(fitnesses):.4f}",
                "sigma": f"{es.sigma:.2f}",
                "gen_time": f"{elapsed_gen:.0f}",
                "total_time": f"{elapsed_total:.0f}",
            }

            # ── Validation of best-so-far every N generations ──
            if gen % args.validate_every == 0:
                print(f"  Validating best-so-far ({args.validate_games} games)...")
                t_val = time.time()

                with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                    val_cfg = dict(cfg, num_games=args.validate_games)
                    val_result = pool.apply(_evaluate_one_with_cfg,
                                           (best_free, val_cfg))

                _, vw, vl, vd = val_result
                val_total["wins"] += vw
                val_total["losses"] += vl
                val_total["draws"] += vd
                vs = win_rate_stats(val_total["wins"], val_total["losses"],
                                    val_total["draws"])
                print(f"  Validation: {vw}W/{vl}L/{vd}D this round, "
                      f"cumulative {val_total['wins']}W/{val_total['losses']}L/{val_total['draws']}D")
                print(f"    WR: {vs['wr']:.1%} "
                      f"[{vs['ci_lo']:.1%}, {vs['ci_hi']:.1%}] "
                      f"(95% CI), "
                      f"Elo: {vs['elo']:+.0f}, "
                      f"p={vs['p']:.4f}, "
                      f"n={vs['n']}, "
                      f"{time.time()-t_val:.0f}s")
                row.update({
                    "val_wr": f"{vs['wr']:.4f}", "val_ci_lo": f"{vs['ci_lo']:.4f}",
                    "val_ci_hi": f"{vs['ci_hi']:.4f}", "val_elo": f"{vs['elo']:.0f}",
                    "val_p": f"{vs['p']:.6f}", "val_n": vs["n"],
                })

            csv_writer.writerow(row)
            csv_file.flush()

            # Checkpoint every generation
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({
                    "es": es,
                    "best_fitness": best_fitness,
                    "best_free": best_free,
                    "generation": gen,
                    "val_total": val_total,
                }, f)

            # Save best pattern_data.h every 10 generations
            if gen % 10 == 0:
                best_full = free_to_full(best_free)
                out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
                save_pattern_data_h(best_full, out_path)
                print(f"  Saved weights: {out_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        csv_file.close()

    # Final save
    best_full = free_to_full(best_free)
    out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
    save_pattern_data_h(best_full, out_path)

    elapsed = time.time() - t_start
    vs = win_rate_stats(val_total["wins"], val_total["losses"], val_total["draws"])
    print(f"\n{'='*60}")
    print(f"  CMA-ES finished after {gen - gen_offset} generations ({elapsed/60:.0f} min)")
    print(f"  Best win rate vs baseline: {-best_fitness:.1%}")
    if vs["n"] > 0:
        print(f"  Validated: {vs['wr']:.1%} [{vs['ci_lo']:.1%}, {vs['ci_hi']:.1%}]"
              f"  Elo: {vs['elo']:+.0f}  p={vs['p']:.4f}  (n={vs['n']})")
    print(f"  CSV log:  {csv_path}")
    print(f"  Weights:  {out_path}")
    print(f"\n  To use these weights:")
    print(f"    cp {out_path} ../../current/pattern_data.h")
    print(f"    cd ../.. && make rebuild")
    print(f"    python evaluate.py -n 100 -t 0.1")
    print(f"{'='*60}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization of SealBot pattern weights")

    parser.add_argument("--games", type=int, default=20,
                        help="Games per fitness evaluation (default: 20)")
    parser.add_argument("--time-limit", type=float, default=0.02,
                        help="Seconds per move during evaluation (default: 0.02)")
    parser.add_argument("--popsize", type=int, default=50,
                        help="CMA-ES population size (default: 50)")
    parser.add_argument("--sigma0", type=float, default=50.0,
                        help="CMA-ES initial step size (default: 50.0)")
    parser.add_argument("--max-gen", type=int, default=500,
                        help="Maximum generations (default: 500)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: cpu_count)")
    parser.add_argument("--validate-every", type=int, default=10,
                        help="Run validation every N generations (default: 10)")
    parser.add_argument("--validate-games", type=int, default=40,
                        help="Games per validation run (default: 40)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    run(parser.parse_args())

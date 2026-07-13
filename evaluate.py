"""Evaluate SealBot: current/ vs best/ (or random / self-play).

Usage:
    python evaluate.py                # current vs best
    python evaluate.py --random       # current vs built-in random
    python evaluate.py --self-play    # current vs itself
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

from tqdm import tqdm

from game import HexGame, Player

ROOT = os.path.abspath(os.path.dirname(__file__))
CURRENT_DIR = os.path.join(ROOT, "current")
BEST_DIR = os.path.join(ROOT, "best")


# ── Constants ──
GRACE_FACTOR = 3.0
MAX_VIOLATIONS_PER_GAME = 10
MAX_MOVES_PER_GAME = 200


# ── Built-in random bot ──

_D2_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-2, 3)
    for dr in range(-2, 3)
    if max(abs(dq), abs(dr), abs(dq + dr)) <= 2 and (dq, dr) != (0, 0)
)


def _random_get_move(game):
    if not game.board:
        return [(0, 0)]
    candidates = set()
    for q, r in game.board:
        for dq, dr in _D2_OFFSETS:
            nb = (q + dq, r + dr)
            if nb not in game.board:
                candidates.add(nb)
    moves = []
    for _ in range(game.moves_left_in_turn):
        if not candidates:
            break
        move = random.choice(list(candidates))
        moves.append(move)
        candidates.discard(move)
    return moves


# ── Bot wrapper ──

class BotRunner:
    def __init__(self, name, get_move_fn, time_limit, bot_obj=None):
        self.name = name
        self._get_move = get_move_fn
        self._bot = bot_obj
        self.time_limit = time_limit
        self._last_depth = 0

    @property
    def last_depth(self):
        if self._bot is not None and hasattr(self._bot, 'last_depth'):
            return self._bot.last_depth
        return self._last_depth

    def get_move(self, game):
        if self._bot is not None and hasattr(self._bot, 'time_limit'):
            self._bot.time_limit = self.time_limit
        deadline = time.time() + self.time_limit
        result = self._get_move(game)
        if hasattr(result, '__next__'):
            best = None
            depth = 0
            for moves in result:
                best = moves
                depth += 1
                if time.time() >= deadline:
                    break
            result.close()
            self._last_depth = depth
            return best if best is not None else []
        self._last_depth = 0
        return result

    def __str__(self):
        return self.name


# ── Statistics ──

def _win_rate_stats(wins, losses, draws):
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.0, 1.0, 1.0, 0, 0, 0
    score = wins + 0.5 * draws
    p_hat = score / n
    z = 1.96
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    ci_lo = max(0.0, centre - spread)
    ci_hi = min(1.0, centre + spread)
    if n > 0:
        z_obs = (score - 0.5 * n) / math.sqrt(0.25 * n)
        p_value = 2 * _norm_sf(abs(z_obs))
    else:
        p_value = 1.0
    return p_hat, ci_lo, ci_hi, p_value, _score_to_elo(p_hat), _score_to_elo(ci_lo), _score_to_elo(ci_hi)


def _score_to_elo(score):
    if score <= 0.0: return float('-inf')
    if score >= 1.0: return float('inf')
    return -400 * math.log10(1.0 / score - 1.0)


def _norm_sf(x):
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    return poly * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ── Exceptions ──

class TimeLimitExceeded(Exception):
    def __init__(self, bot, violations):
        self.bot = bot
        self.violations = violations
        super().__init__(f"{bot} exceeded time limit {violations} times")


# ── Core game loop ──

def play_game(bot_a, bot_b, win_length=6, violations=None, max_moves=None):
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME
    game = HexGame(win_length=win_length)
    bots = {Player.A: bot_a, Player.B: bot_b}
    depths = {Player.A: defaultdict(int), Player.B: defaultdict(int)}
    times = {Player.A: [0.0, 0], Player.B: [0.0, 0]}
    total_moves = 0

    while not game.game_over:
        player = game.current_player
        bot = bots[player]
        t0 = time.time()
        moves = bot.get_move(game)
        elapsed = time.time() - t0

        if not moves:
            return (Player.B if player == Player.A else Player.A,
                    depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]),
                    total_moves)

        num_moves = len(moves)
        times[player][0] += elapsed
        times[player][1] += 1
        if elapsed > bot.time_limit * GRACE_FACTOR:
            if violations is not None:
                violations[bot] = violations.get(bot, 0) + 1
                if violations[bot] >= MAX_VIOLATIONS_PER_GAME:
                    raise TimeLimitExceeded(bot, violations[bot])
        depths[player][bot.last_depth] += num_moves
        total_moves += num_moves
        if total_moves >= max_moves:
            return (Player.NONE, depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]),
                    total_moves)

        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                return (Player.B if player == Player.A else Player.A,
                        depths[Player.A], depths[Player.B],
                        tuple(times[Player.A]), tuple(times[Player.B]),
                        total_moves)

    return (game.winner, depths[Player.A], depths[Player.B],
            tuple(times[Player.A]), tuple(times[Player.B]),
            total_moves)


# ── Worker for same-module modes (random / self-play) ──

def _load_current_bot(time_limit):
    sys.path.insert(0, CURRENT_DIR)
    from minimax_cpp import MinimaxBot
    sys.path.pop(0)
    return MinimaxBot(time_limit)


def _play_one_same(args):
    time_limit, game_idx, win_length, max_moves, mode, current_dir = args
    swapped = game_idx % 2 == 1

    sys.path.insert(0, current_dir)
    from minimax_cpp import MinimaxBot
    sys.path.pop(0)

    seal = MinimaxBot(time_limit)
    bot_seal = BotRunner("current", seal.get_move, time_limit, bot_obj=seal)

    if mode == "random":
        bot_opp = BotRunner("random", _random_get_move, time_limit)
    else:
        seal2 = MinimaxBot(time_limit)
        bot_opp = BotRunner("current_B", seal2.get_move, time_limit, bot_obj=seal2)

    if swapped:
        seat_a, seat_b = bot_opp, bot_seal
    else:
        seat_a, seat_b = bot_seal, bot_opp

    violations = {}
    exceeded = False
    mc = 0
    try:
        winner, d_a, d_b, t_a, t_b, mc = play_game(
            seat_a, seat_b, win_length, violations, max_moves)
    except TimeLimitExceeded:
        exceeded = True
        winner = Player.NONE
        d_a, d_b = defaultdict(int), defaultdict(int)
        t_a, t_b = (0.0, 0), (0.0, 0)

    return (winner.value, swapped, dict(d_a), dict(d_b),
            violations.get(seat_a, 0), violations.get(seat_b, 0),
            exceeded, t_a, t_b, mc)


# ── Bot server subprocess for current vs best ──
# Each bot runs in its own subprocess to avoid CPython's C extension cache
# returning the same module for both (del sys.modules doesn't work for C exts).

_BOT_SERVER_SCRIPT = r'''
import sys, json, os, time

bot_dir, root_dir = sys.argv[1], sys.argv[2]
time_limit = float(sys.argv[3])

sys.path.insert(0, bot_dir)
from minimax_cpp import MinimaxBot

sys.path.insert(0, root_dir)
from game import HexGame, Player

bot = MinimaxBot(time_limit)

for line in sys.stdin:
    line = line.strip()
    if not line or line == "quit":
        break
    req = json.loads(line)

    bot.time_limit = req.get("time_limit", time_limit)

    g = HexGame(win_length=req["win_length"])
    for pos, v in req["board"]:
        g.board[(pos[0], pos[1])] = Player(v)
    g.current_player = Player(req["current_player"])
    g.moves_left_in_turn = req["moves_left"]
    g.move_count = req["move_count"]

    t0 = time.time()
    moves = bot.get_move(g)
    elapsed = time.time() - t0

    sys.stdout.write(json.dumps({
        "moves": [list(m) for m in moves] if moves else [],
        "elapsed": elapsed,
        "depth": getattr(bot, "last_depth", 0),
    }) + "\n")
    sys.stdout.flush()
'''


def _find_so(directory):
    for name in os.listdir(directory):
        if name.startswith("minimax_cpp") and (name.endswith(".so") or name.endswith(".pyd")):
            return os.path.join(directory, name)
    return None


def _start_bot_server(python_exe, bot_dir, root_dir, time_limit):
    return subprocess.Popen(
        [python_exe, "-c", _BOT_SERVER_SCRIPT, bot_dir, root_dir, str(time_limit)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)


def _request_move(proc, game, time_limit):
    """Send game state to a bot server, return (moves, elapsed, depth) or None."""
    board = [[[q, r], v.value] for (q, r), v in game.board.items()]
    req = json.dumps({
        "win_length": game.win_length,
        "board": board,
        "current_player": game.current_player.value,
        "moves_left": game.moves_left_in_turn,
        "move_count": game.move_count,
        "time_limit": time_limit,
    }) + "\n"
    try:
        proc.stdin.write(req)
        proc.stdin.flush()
        resp_line = proc.stdout.readline()
        if not resp_line:
            return None
        resp = json.loads(resp_line)
        return [tuple(m) for m in resp["moves"]], resp["elapsed"], resp["depth"]
    except (BrokenPipeError, json.JSONDecodeError, OSError):
        return None


def _play_one_cross(root_dir, python_exe, time_limit, game_idx,
                    win_length, max_moves, opp_dir="best"):
    swapped = game_idx % 2 == 1
    current_dir = os.path.join(root_dir, "current")
    best_dir = os.path.join(root_dir, opp_dir)

    proc_cur = _start_bot_server(python_exe, current_dir, root_dir, time_limit)
    proc_best = _start_bot_server(python_exe, best_dir, root_dir, time_limit)

    try:
        game = HexGame(win_length=win_length)

        if swapped:
            procs = {Player.A: proc_best, Player.B: proc_cur}
        else:
            procs = {Player.A: proc_cur, Player.B: proc_best}

        da, db = defaultdict(int), defaultdict(int)
        ta, tb = [0.0, 0], [0.0, 0]
        total = 0
        violations = {}
        exceeded = False
        winner_val = 0
        game_deadline = time.time() + max_moves * time_limit * 4 + 30

        while not game.game_over:
            if time.time() > game_deadline:
                break

            p = game.current_player
            result = _request_move(procs[p], game, time_limit)
            if result is None:
                winner_val = 2 if p == Player.A else 1
                break

            moves, elapsed, depth = result

            if not moves:
                winner_val = 2 if p == Player.A else 1
                break

            nm = len(moves)
            t_arr = ta if p == Player.A else tb
            t_arr[0] += elapsed
            t_arr[1] += 1
            (da if p == Player.A else db)[depth] += nm

            if elapsed > time_limit * GRACE_FACTOR:
                seat = "a" if p == Player.A else "b"
                violations[seat] = violations.get(seat, 0) + 1
                if violations[seat] >= MAX_VIOLATIONS_PER_GAME:
                    exceeded = True
                    break

            total += nm
            if total >= max_moves:
                break

            bad = False
            for q, r in moves:
                if game.game_over or not game.make_move(q, r):
                    bad = True
                    break
            if bad:
                winner_val = 2 if p == Player.A else 1
                break

        if not exceeded and game.game_over:
            winner_val = game.winner.value

        return {
            "winner": winner_val, "swapped": swapped,
            "d_a": dict(da), "d_b": dict(db),
            "v_a": violations.get("a", 0), "v_b": violations.get("b", 0),
            "exceeded": exceeded, "t_a": list(ta), "t_b": list(tb),
            "move_count": total,
        }
    except Exception as e:
        sys.stderr.write(f"Game {game_idx} error: {e}\n")
        return None
    finally:
        for proc in [proc_cur, proc_best]:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()


# ── Main evaluate ──

def evaluate(num_games=20, win_length=6, time_limit=0.1, use_tqdm=True,
             max_moves=None, mode="best", opp_dir="best"):
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME

    name_a = "current"
    name_b = {"best": opp_dir, "random": "random", "self": "current_B"}[mode]

    bot_a_wins = bot_b_wins = draws = games_played = 0
    bot_a_violations = bot_b_violations = aborted_games = 0
    bot_a_depths, bot_b_depths = defaultdict(int), defaultdict(int)
    bot_a_time, bot_b_time = [0.0, 0], [0.0, 0]
    game_lengths = []
    t0 = time.time()

    def _accum(wv, sw, da, db, va, vb, exc, ta, tb, mc):
        nonlocal bot_a_wins, bot_b_wins, draws, games_played
        nonlocal bot_a_violations, bot_b_violations, aborted_games
        if exc: aborted_games += 1
        else: game_lengths.append(mc)
        if sw:
            for d, c in da.items(): bot_b_depths[d] += c
            for d, c in db.items(): bot_a_depths[d] += c
            bot_b_violations += va; bot_a_violations += vb
            bot_b_time[0] += ta[0]; bot_b_time[1] += ta[1]
            bot_a_time[0] += tb[0]; bot_a_time[1] += tb[1]
            if wv == 1: bot_b_wins += 1
            elif wv == 2: bot_a_wins += 1
            else: draws += 1
        else:
            for d, c in da.items(): bot_a_depths[d] += c
            for d, c in db.items(): bot_b_depths[d] += c
            bot_a_violations += va; bot_b_violations += vb
            bot_a_time[0] += ta[0]; bot_a_time[1] += ta[1]
            bot_b_time[0] += tb[0]; bot_b_time[1] += tb[1]
            if wv == 1: bot_a_wins += 1
            elif wv == 2: bot_b_wins += 1
            else: draws += 1
        games_played += 1

    if mode in ("random", "self"):
        workers = min(num_games, os.cpu_count() or 1)
        args = [(time_limit, i, win_length, max_moves, mode, CURRENT_DIR)
                for i in range(num_games)]
        from multiprocessing import Pool
        with Pool(workers) as pool:
            it = pool.imap_unordered(_play_one_same, args)
            if use_tqdm:
                it = tqdm(it, total=num_games, desc="Games", unit="game")
            for raw in it:
                wv, sw, da, db, va, vb, exc, ta, tb, mc = raw
                _accum(wv, sw, da, db, va, vb, exc, ta, tb, mc)
                if use_tqdm:
                    it.set_postfix(A=bot_a_wins, B=bot_b_wins, D=draws)
    else:
        # current vs best: subprocess per game
        python_exe = sys.executable
        workers = min(num_games, os.cpu_count() or 1)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_play_one_cross, ROOT, python_exe,
                                       time_limit, i, win_length, max_moves,
                                       opp_dir)
                       for i in range(num_games)]
            it = as_completed(futures)
            if use_tqdm:
                it = tqdm(it, total=num_games, desc="Games", unit="game")
            for future in it:
                raw = future.result()
                if raw is None:
                    aborted_games += 1; games_played += 1; continue
                _accum(raw["winner"], raw["swapped"],
                       {int(k): v for k, v in raw["d_a"].items()},
                       {int(k): v for k, v in raw["d_b"].items()},
                       raw["v_a"], raw["v_b"], raw["exceeded"],
                       (raw["t_a"][0], raw["t_a"][1]),
                       (raw["t_b"][0], raw["t_b"][1]),
                       raw["move_count"])
                if use_tqdm:
                    it.set_postfix(A=bot_a_wins, B=bot_b_wins, D=draws)

    elapsed = time.time() - t0
    total = max(games_played, 1)

    # ── Report ──
    print(f"\n\n{'='*50}")
    print(f"  {name_a} vs {name_b}  \u2014  {games_played} games in {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"  {name_a:>15s}: {bot_a_wins:3d} wins ({100*bot_a_wins/total:.0f}%)")
    print(f"  {name_b:>15s}: {bot_b_wins:3d} wins ({100*bot_b_wins/total:.0f}%)")
    print(f"  {'Draws':>15s}: {draws:3d}      ({100*draws/total:.0f}%)")

    wr, ci_lo, ci_hi, pv, elo, elo_lo, elo_hi = _win_rate_stats(bot_a_wins, bot_b_wins, draws)
    print(f"\n  {name_a} win rate: {100*wr:.1f}% (95% CI: {100*ci_lo:.1f}%\u2013{100*ci_hi:.1f}%)")
    fe = lambda e: ("+\u221e" if e > 0 else "-\u221e") if math.isinf(e) else f"{e:+.0f}"
    print(f"  Elo difference: {fe(elo)} (95% CI: {fe(elo_lo)} to {fe(elo_hi)})")
    ps = f"{pv:.1e}" if pv < 0.001 else f"{pv:.3f}"
    print(f"  p-value (H\u2080: equal strength): {ps} {'*' if pv < 0.05 else ''}")
    print()

    for name, depths in [(name_a, bot_a_depths), (name_b, bot_b_depths)]:
        if not depths: continue
        tm = sum(depths.values())
        avg = sum(d * c for d, c in depths.items()) / tm
        print(f"  {name} search depth: avg {avg:.1f}, range [{min(depths)}-{max(depths)}]")
        print(f"    {'  '.join(f'd{d}:{c}' for d, c in sorted(depths.items()))}")

    for name, bt in [(name_a, bot_a_time), (name_b, bot_b_time)]:
        if bt[1] > 0:
            print(f"  {name} avg turn time: {1000*bt[0]/bt[1]:.0f}ms ({bt[1]} turns)")

    if game_lengths:
        print(f"\n  Game length: avg {sum(game_lengths)/len(game_lengths):.1f} moves, "
              f"range [{min(game_lengths)}-{max(game_lengths)}]")

    if bot_a_violations or bot_b_violations or aborted_games:
        print(f"\n  TIME VIOLATIONS: {name_a}={bot_a_violations}, {name_b}={bot_b_violations}"
              f"  ({aborted_games} games forfeited)")

    print(f"{'='*50}")
    return bot_a_wins, bot_b_wins, draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SealBot: current vs best.")
    parser.add_argument("-n", "--num-games", type=int, default=20,
                        help="Number of games (default: 20)")
    parser.add_argument("-t", "--time-limit", type=float, default=0.1,
                        help="Time limit per turn in seconds (default: 0.1)")
    parser.add_argument("--random", action="store_true",
                        help="Play current against built-in random bot")
    parser.add_argument("--self-play", action="store_true",
                        help="current vs itself")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("--opp-dir", type=str, default="best",
                        help="Opponent build dir name (default: best)")
    parsed = parser.parse_args()

    if parsed.random:
        mode = "random"
    elif parsed.self_play:
        mode = "self"
    else:
        mode = "best"
        opp = os.path.join(ROOT, parsed.opp_dir)
        for d, label in [(CURRENT_DIR, "current"), (opp, parsed.opp_dir)]:
            if _find_so(d) is None:
                print(f"Error: no minimax_cpp .so in {d}/")
                print(f"  Run: make build")
                sys.exit(1)

    evaluate(num_games=parsed.num_games, time_limit=parsed.time_limit,
             use_tqdm=not parsed.no_tqdm, mode=mode, opp_dir=parsed.opp_dir)

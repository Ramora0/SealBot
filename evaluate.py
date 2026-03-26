"""Evaluate SealBot by playing it against a built-in random baseline.

Usage:
    python evaluate.py [-n NUM_GAMES] [-t TIME_LIMIT]
    python evaluate.py --self-play [-n NUM_GAMES] [-t TIME_LIMIT]
"""

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from tqdm import tqdm

from game import HexGame, Player
from minimax_cpp import MinimaxBot


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
        deadline = time.time() + self.time_limit * game.moves_left_in_turn
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
    elo_diff = _score_to_elo(p_hat)
    elo_lo = _score_to_elo(ci_lo)
    elo_hi = _score_to_elo(ci_hi)
    return p_hat, ci_lo, ci_hi, p_value, elo_diff, elo_lo, elo_hi


def _score_to_elo(score):
    if score <= 0.0:
        return float('-inf')
    if score >= 1.0:
        return float('inf')
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
                    tuple(times[Player.A]), tuple(times[Player.B]))

        num_moves = len(moves)
        times[player][0] += elapsed
        times[player][1] += num_moves
        allowed_time = bot.time_limit * num_moves
        if elapsed > allowed_time * GRACE_FACTOR:
            if violations is not None:
                violations[bot] = violations.get(bot, 0) + 1
                if violations[bot] >= MAX_VIOLATIONS_PER_GAME:
                    raise TimeLimitExceeded(bot, violations[bot])
        depths[player][bot.last_depth] += num_moves
        total_moves += num_moves
        if total_moves >= max_moves:
            return (Player.NONE, depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]))

        invalid = False
        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                invalid = True
                break
        if invalid:
            return (Player.B if player == Player.A else Player.A,
                    depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]))

    return (game.winner, depths[Player.A], depths[Player.B],
            tuple(times[Player.A]), tuple(times[Player.B]))


def _play_one(args):
    time_limit, game_idx, win_length, max_moves, self_play = args
    swapped = game_idx % 2 == 1

    bot_seal = BotRunner("SealBot", MinimaxBot(time_limit).get_move, time_limit,
                         bot_obj=MinimaxBot(time_limit))
    if self_play:
        bot_opp = BotRunner("SealBot_B", MinimaxBot(time_limit).get_move, time_limit,
                            bot_obj=MinimaxBot(time_limit))
    else:
        bot_opp = BotRunner("random", _random_get_move, time_limit)

    if swapped:
        seat_a, seat_b = bot_opp, bot_seal
    else:
        seat_a, seat_b = bot_seal, bot_opp

    violations = {}
    exceeded = False
    try:
        winner, d_a, d_b, t_a, t_b = play_game(
            seat_a, seat_b, win_length, violations, max_moves)
    except TimeLimitExceeded:
        exceeded = True
        winner = Player.NONE
        d_a, d_b = defaultdict(int), defaultdict(int)
        t_a, t_b = (0.0, 0), (0.0, 0)

    move_count = t_a[1] + t_b[1]
    return (winner, swapped, dict(d_a), dict(d_b),
            violations.get(seat_a, 0), violations.get(seat_b, 0),
            exceeded, t_a, t_b, move_count)


def evaluate(num_games=20, win_length=6, time_limit=0.1, use_tqdm=True,
             max_moves=None, self_play=False):
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME

    name_a = "SealBot"
    name_b = "SealBot_B" if self_play else "random"

    bot_a_wins = 0
    bot_b_wins = 0
    draws = 0
    games_played = 0
    bot_a_depths = defaultdict(int)
    bot_b_depths = defaultdict(int)
    bot_a_violations = 0
    bot_b_violations = 0
    aborted_games = 0
    bot_a_time = [0.0, 0]
    bot_b_time = [0.0, 0]
    game_lengths = []

    workers = min(num_games, __import__('os').cpu_count() or 1)
    args = [(time_limit, i, win_length, max_moves, self_play) for i in range(num_games)]

    t0 = time.time()
    with Pool(workers) as pool:
        results_iter = pool.imap_unordered(_play_one, args)
        if use_tqdm:
            results_iter = tqdm(results_iter, total=num_games, desc="Games", unit="game")
        for result in results_iter:
            winner, swapped, d_a, d_b, v_a, v_b, exceeded, t_a, t_b, move_count = result

            if exceeded:
                aborted_games += 1
            else:
                game_lengths.append(move_count)

            if swapped:
                for d, c in d_a.items(): bot_b_depths[d] += c
                for d, c in d_b.items(): bot_a_depths[d] += c
                bot_b_violations += v_a
                bot_a_violations += v_b
                bot_b_time[0] += t_a[0]; bot_b_time[1] += t_a[1]
                bot_a_time[0] += t_b[0]; bot_a_time[1] += t_b[1]
                if winner == Player.A:     bot_b_wins += 1
                elif winner == Player.B:   bot_a_wins += 1
                else:                      draws += 1
            else:
                for d, c in d_a.items(): bot_a_depths[d] += c
                for d, c in d_b.items(): bot_b_depths[d] += c
                bot_a_violations += v_a
                bot_b_violations += v_b
                bot_a_time[0] += t_a[0]; bot_a_time[1] += t_a[1]
                bot_b_time[0] += t_b[0]; bot_b_time[1] += t_b[1]
                if winner == Player.A:     bot_a_wins += 1
                elif winner == Player.B:   bot_b_wins += 1
                else:                      draws += 1

            games_played += 1
            if use_tqdm:
                results_iter.set_postfix(A=bot_a_wins, B=bot_b_wins, D=draws)

    elapsed = time.time() - t0
    total = max(games_played, 1)

    # ── Report ──
    print(f"\n\n{'='*50}")
    print(f"  {name_a} vs {name_b}  \u2014  {games_played} games in {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"  {name_a:>15s}: {bot_a_wins:3d} wins ({100*bot_a_wins/total:.0f}%)")
    print(f"  {name_b:>15s}: {bot_b_wins:3d} wins ({100*bot_b_wins/total:.0f}%)")
    print(f"  {'Draws':>15s}: {draws:3d}      ({100*draws/total:.0f}%)")

    win_rate, ci_lo, ci_hi, p_value, elo_diff, elo_lo, elo_hi = _win_rate_stats(bot_a_wins, bot_b_wins, draws)
    print(f"\n  {name_a} win rate: {100*win_rate:.1f}% "
          f"(95% CI: {100*ci_lo:.1f}%\u2013{100*ci_hi:.1f}%)")
    def _fmt_elo(e):
        return ("+\u221e" if e > 0 else "-\u221e") if math.isinf(e) else f"{e:+.0f}"
    print(f"  Elo difference: {_fmt_elo(elo_diff)} "
          f"(95% CI: {_fmt_elo(elo_lo)} to {_fmt_elo(elo_hi)})")
    if p_value < 0.001:
        p_str = f"{p_value:.1e}"
    else:
        p_str = f"{p_value:.3f}"
    sig = "*" if p_value < 0.05 else ""
    print(f"  p-value (H\u2080: equal strength): {p_str} {sig}")
    print()

    for name, depths in [(name_a, bot_a_depths), (name_b, bot_b_depths)]:
        if not depths:
            continue
        total_moves = sum(depths.values())
        avg = sum(d * c for d, c in depths.items()) / total_moves
        lo, hi = min(depths), max(depths)
        print(f"  {name} search depth: avg {avg:.1f}, range [{lo}-{hi}]")
        buckets = sorted(depths.items())
        dist = "  ".join(f"d{d}:{c}" for d, c in buckets)
        print(f"    {dist}")

    for name, bt in [(name_a, bot_a_time), (name_b, bot_b_time)]:
        if bt[1] > 0:
            avg_ms = 1000 * bt[0] / bt[1]
            print(f"  {name} avg move time: {avg_ms:.0f}ms ({bt[1]} moves)")

    if game_lengths:
        avg_len = sum(game_lengths) / len(game_lengths)
        lo_len, hi_len = min(game_lengths), max(game_lengths)
        print(f"\n  Game length: avg {avg_len:.1f} moves, range [{lo_len}-{hi_len}]")

    if bot_a_violations or bot_b_violations or aborted_games:
        print()
        print(f"  TIME VIOLATIONS: {name_a}={bot_a_violations}, {name_b}={bot_b_violations}"
              f"  ({aborted_games} games forfeited)")

    print(f"{'='*50}")
    return bot_a_wins, bot_b_wins, draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SealBot.")
    parser.add_argument("-n", "--num-games", type=int, default=20,
                        help="Number of games (default: 20)")
    parser.add_argument("-t", "--time-limit", type=float, default=0.1,
                        help="Time limit per move in seconds (default: 0.1)")
    parser.add_argument("--self-play", action="store_true",
                        help="SealBot vs SealBot instead of vs random")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable progress bar")
    parsed = parser.parse_args()

    evaluate(num_games=parsed.num_games, time_limit=parsed.time_limit,
             use_tqdm=not parsed.no_tqdm, self_play=parsed.self_play)

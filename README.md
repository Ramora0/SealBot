# SealBot

A competitive bot for **Hex Tic-Tac-Toe** — a two-player strategy game on an infinite hexagonal grid where the first player to get 6 in a row wins. Players alternate placing stones (1 stone on the first turn, then 2 per turn). The board is sparse and unbounded, so the game is purely tactical — there are no edges to anchor against.

SealBot uses **iterative deepening alpha-beta minimax** with a pattern-based evaluation function, implemented in C++ and exposed to Python via pybind11. Key features:

- 729 ternary pattern evaluations across 6-cell windows
- Zobrist hashing with transposition tables
- Flat 140x140 array storage for cache-friendly access
- Candidate move generation within distance-2 of existing stones
- Quiescence search for mate threats
- Graceful time management with state rollback on timeout

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./build.sh
```

## Project Structure

```
SealBot/
├── current/       # The bot you're working on — edit these files
├── best/          # The baseline bot to beat
├── game.py        # Game rules (don't modify)
├── evaluate.py    # Run current vs best (definitive quality test)
├── benchmark.py  # Fixed-depth speed benchmark
├── play.py        # Play against the bot interactively
├── build.sh       # Clean + rebuild both bots
└── Makefile
```

Both `current/` and `best/` contain the same set of C++ source files:

| File | Purpose |
|------|---------|
| `engine.h` | Core minimax search engine (~1400 lines) |
| `minimax_bot.cpp` | pybind11 wrapper |
| `types.h` | Shared type definitions |
| `pattern_data.h` | 729 baked-in pattern evaluation weights |
| `setup.py` | Build configuration |

## Running Experiments

The iteration loop:

1. **Make a change** in `current/` (e.g. tweak `engine.h`)
2. **Evaluate** against the baseline:
   ```bash
   make evaluate              # 20 games, 0.1s/move
   make evaluate N=100 T=0.2  # 100 games, 0.2s/move
   ```
3. **If current wins convincingly**, promote it:
   ```bash
   make promote   # copies current/ -> best/
   make rebuild   # recompile both
   ```
4. Repeat.

`evaluate.py` is the definitive test for whether a change is an improvement — it plays real games and measures win rate. A p-value under 0.05 is a good signal; 100+ games at longer time controls gives more reliable results.

### Speed-only benchmarking

For pure speed optimizations (no quality tradeoff), use `benchmark.py` instead. It runs both bots on the same random positions at a fixed search depth and compares execution time, node counts, and move agreement:

```bash
python benchmark.py                # 10 positions, depth 4
python benchmark.py -n 20 -d 6     # 20 positions, depth 6
python benchmark.py --seed 123     # custom seed for reproducibility
```

If move agreement is 100% and current is faster, you know the optimization is safe to promote without running full games.

### Other evaluation modes

```bash
# Current vs built-in random bot (sanity check)
.venv/bin/python evaluate.py --random

# Current vs itself (check for first-player bias, verify draws)
.venv/bin/python evaluate.py --self-play
```

### Interactive play

```bash
make play                              # play against current/
.venv/bin/python play.py --time-limit 1.0  # give the AI more time
```

Controls: click to place stones. `SPACE` to swap sides, `A` for AI-vs-AI autoplay, `N` for move numbers, `E` for board editor, arrow keys to review move history, `S` to save position, `Q` to quit.

## Game Rules

- Played on an infinite hex grid using axial coordinates (q, r)
- Player A places 1 stone first, then players alternate placing 2 stones each turn
- First to get **6 in a row** along any hex axis wins
- Three win directions: horizontal (1,0), diagonal (0,1), anti-diagonal (1,-1)

## Make Targets

| Command | Description |
|---------|-------------|
| `make build` | Build both current/ and best/ |
| `make rebuild` | Clean and rebuild both |
| `make evaluate` | Run current vs best (N=games, T=time) |
| `make play` | Play against current interactively |
| `make promote` | Copy current -> best |
| `make clean` | Remove build artifacts |

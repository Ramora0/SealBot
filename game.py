"""Hexagonal Tic-Tac-Toe game logic.

Rules:
- Played on an infinite hex grid.
- Player A places 1 stone first, then players alternate placing 2 stones each.
- First player to get win_length in a row along any hex axis wins.
- Uses axial coordinates (q, r) with implicit s = -q - r.
"""

from enum import Enum
from dataclasses import dataclass, field


class Player(Enum):
    NONE = 0
    A = 1
    B = 2


# The three line directions in axial coordinates
HEX_DIRECTIONS = [(1, 0), (0, 1), (1, -1)]


@dataclass
class HexGame:
    win_length: int = 6

    board: dict = field(default_factory=dict, init=False, repr=False)
    current_player: Player = field(default=Player.A, init=False)
    moves_left_in_turn: int = field(default=1, init=False)
    move_count: int = field(default=0, init=False)
    winner: Player = field(default=Player.NONE, init=False)
    winning_cells: list = field(default_factory=list, init=False)
    game_over: bool = field(default=False, init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.board = {}  # sparse: only occupied cells
        self.current_player = Player.A
        self.moves_left_in_turn = 1  # A gets only 1 move on the first turn
        self.move_count = 0
        self.winner = Player.NONE
        self.winning_cells = []
        self.game_over = False

    def is_valid_move(self, q, r):
        if self.game_over:
            return False
        return (q, r) not in self.board

    def save_state(self):
        """Snapshot the mutable state (for undo in search)."""
        return (
            self.current_player,
            self.moves_left_in_turn,
            self.winner,
            self.winning_cells[:],
            self.game_over,
        )

    def undo_move(self, q, r, state):
        """Undo a move and restore the saved state."""
        del self.board[(q, r)]
        self.move_count -= 1
        (self.current_player, self.moves_left_in_turn,
         self.winner, self.winning_cells, self.game_over) = state

    def make_move(self, q, r):
        """Place a stone at (q, r). Returns True if the move was valid."""
        if not self.is_valid_move(q, r):
            return False

        self.board[(q, r)] = self.current_player
        self.move_count += 1

        if self._check_win(q, r):
            self.winner = self.current_player
            self.game_over = True
            return True

        self.moves_left_in_turn -= 1
        if self.moves_left_in_turn <= 0:
            self._switch_player()

        return True

    def _switch_player(self):
        if self.current_player == Player.A:
            self.current_player = Player.B
        else:
            self.current_player = Player.A
        self.moves_left_in_turn = 2

    def _check_win(self, q, r):
        """Check if placing at (q, r) creates win_length in a row."""
        player = self.board[(q, r)]

        for dq, dr in HEX_DIRECTIONS:
            cells = [(q, r)]
            for i in range(1, self.win_length):
                nq, nr = q + dq * i, r + dr * i
                if self.board.get((nq, nr)) == player:
                    cells.append((nq, nr))
                else:
                    break
            for i in range(1, self.win_length):
                nq, nr = q - dq * i, r - dr * i
                if self.board.get((nq, nr)) == player:
                    cells.append((nq, nr))
                else:
                    break
            if len(cells) >= self.win_length:
                self.winning_cells = cells
                return True

        return False

import numpy as np
from typing import List, Optional

ROWS = 6
COLS = 7
WIN_LENGTH = 4
ACTION_SIZE = COLS  # One action per column


class Connect4:
    """
    Connect Four game state.

    Board convention:
      board[row][col] = 0 (empty), 1 (player 1), -1 (player 2)
      Row 0 is the bottom of the board.

    Player convention:
      current_player = 1 or -1, alternating each move.
      winner         = 1, -1, or 0 (draw / not over yet).
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player: int = 1
        self.num_moves: int = 0
        self.col_heights = np.zeros(COLS, dtype=np.int8)
        self.winner: int = 0
        self.game_over: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_move_string(cls, move_str: str) -> "Connect4":
        """
        Build a game from a gamesolver-format move string.
        Each character is a 1-indexed column digit ('1'–'7').
        """
        game = cls()
        for ch in move_str:
            col = int(ch) - 1  # 1-indexed → 0-indexed
            if not game.make_move(col):
                raise ValueError(
                    f"Invalid move '{ch}' (col {col}) at position {game.num_moves} "
                    f"in move string '{move_str}'"
                )
        return game

    @classmethod
    def from_moves(cls, moves: List[int]) -> "Connect4":
        """Build a game from a list of 0-indexed column moves."""
        game = cls()
        for move in moves:
            game.make_move(move)
        return game

    # ------------------------------------------------------------------
    # Core game operations
    # ------------------------------------------------------------------

    def clone(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.num_moves = self.num_moves
        g.col_heights = self.col_heights.copy()
        g.winner = self.winner
        g.game_over = self.game_over
        return g

    def get_valid_moves(self) -> List[int]:
        if self.game_over:
            return []
        return [c for c in range(COLS) if self.col_heights[c] < ROWS]

    def is_valid_move(self, col: int) -> bool:
        return (
            not self.game_over
            and 0 <= col < COLS
            and int(self.col_heights[col]) < ROWS
        )

    def make_move(self, col: int) -> bool:
        """
        Drop a piece in the given column.
        Returns True if the move was legal and applied, False otherwise.
        """
        if not self.is_valid_move(col):
            return False
        row = int(self.col_heights[col])
        self.board[row][col] = self.current_player
        self.col_heights[col] += 1
        self.num_moves += 1

        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif self.num_moves == ROWS * COLS:
            self.game_over = True  # Draw

        self.current_player = -self.current_player
        return True

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the piece just placed at (row, col) creates a winning line."""
        player = self.board[row][col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= WIN_LENGTH:
                return True
        return False

    # ------------------------------------------------------------------
    # Neural-network interface
    # ------------------------------------------------------------------

    def get_canonical_board(self) -> np.ndarray:
        """
        Board tensor from the current player's perspective.

        Shape: (3, ROWS, COLS) float32
          Channel 0: current player's pieces  (1 where present, else 0)
          Channel 1: opponent's pieces         (1 where present, else 0)
          Channel 2: all-ones                  (constant; marks this as a canonical view)
        """
        planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
        planes[0] = (self.board == self.current_player).astype(np.float32)
        planes[1] = (self.board == -self.current_player).astype(np.float32)
        planes[2] = 1.0
        return planes

    # ------------------------------------------------------------------
    # Outcome queries
    # ------------------------------------------------------------------

    def get_outcome(self, player: int) -> Optional[float]:
        """
        Outcome from `player`'s perspective.
        Returns +1.0 (win), -1.0 (loss), 0.0 (draw), or None (game still going).
        """
        if not self.game_over:
            return None
        if self.winner == player:
            return 1.0
        if self.winner == -player:
            return -1.0
        return 0.0  # Draw

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for r in range(ROWS - 1, -1, -1):
            lines.append(" ".join(symbols[int(self.board[r][c])] for c in range(COLS)))
        lines.append("-" * (COLS * 2 - 1))
        lines.append(" ".join(str(c + 1) for c in range(COLS)))
        if self.game_over:
            if self.winner == 1:
                status = "X wins"
            elif self.winner == -1:
                status = "O wins"
            else:
                status = "Draw"
        else:
            player_str = "X (player 1)" if self.current_player == 1 else "O (player 2)"
            status = f"{player_str} to move"
        lines.append(status)
        return "\n".join(lines)

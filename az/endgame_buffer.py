"""
Endgame buffer: a deduplicated rolling buffer of near-terminal positions.

Purpose
-------
Self-play games contain a sparse signal: most positions are mid-game where
the outcome is uncertain.  The last few moves before a terminal state carry
the strongest training signal because the true game outcome (value label) is
certain and the MCTS policy at those positions typically concentrates on the
winning/defending move.

This buffer captures the last `lookback` positions of every self-play game
and stores them in a deduplicated rolling buffer.  Deduplication is by board
content (canonical byte hash), so the same game position is never stored
twice even if it appears in many games (common e.g. after popular openings
that collapse into the same mid-game state).

The buffer is sampled alongside the main ReplayBuffer during each training
step, giving the network extra gradient signal on positions closest to the
ground-truth outcome.

Interface mirrors ReplayBuffer so the trainer can treat both uniformly.
"""

from collections import deque
from typing import List, Tuple

import numpy as np


class EndgameBuffer:
    """
    Deduplicated rolling buffer of near-terminal (board, policy, value) triples.

    Parameters
    ----------
    max_size : int
        Maximum number of unique positions stored.  When the buffer is full
        the oldest entry is evicted to make room (FIFO eviction).
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        # Ordered storage: each element is (board, policy, value)
        self._data: deque = deque()
        # Parallel deque of board hashes — needed to remove a hash from
        # _seen when the corresponding entry is evicted.
        self._keys: deque = deque()
        # Set of hashes for O(1) duplicate checks.
        self._seen: set = set()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_game(
        self,
        game_data: List[Tuple[np.ndarray, np.ndarray, float]],
        lookback: int,
    ) -> int:
        """
        Extract the last `lookback` positions from a finished game and add
        any that are not already in the buffer.

        game_data is the list returned by SelfPlayTrainer.self_play_game():
          [(canonical_board, mcts_policy, outcome), ...]  in chronological order.

        Returns the number of new positions actually inserted.
        """
        added = 0
        # game_data[-lookback:] gives the positions closest to the terminal state.
        for board, policy, value in game_data[-lookback:]:
            if self._add_one(board, policy, value):
                added += 1
        return added

    def _add_one(
        self, board: np.ndarray, policy: np.ndarray, value: float
    ) -> bool:
        """
        Attempt to add a single position.  Returns True if it was new.
        """
        key = board.tobytes()
        if key in self._seen:
            return False   # duplicate — already have this exact board state

        # Evict oldest entry if at capacity.
        if len(self._data) >= self.max_size:
            self._data.popleft()
            evicted_key = self._keys.popleft()
            self._seen.discard(evicted_key)

        self._data.append((board, policy, np.float32(value)))
        self._keys.append(key)
        self._seen.add(key)
        return True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw a random batch (without replacement, capped at buffer size).

        Returns
        -------
        boards   : (n, 3, ROWS, COLS) float32
        policies : (n, COLS) float32
        values   : (n,) float32
        where n = min(batch_size, len(self)).
        """
        n = min(batch_size, len(self._data))
        idx = np.random.choice(len(self._data), n, replace=False)
        data_list = list(self._data)   # O(len) but only done once per train step
        boards, policies, values = zip(*[data_list[i] for i in idx])
        return (
            np.stack(boards).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.array(values, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_ready(self, min_size: int) -> bool:
        return len(self._data) >= min_size

    def __len__(self) -> int:
        return len(self._data)

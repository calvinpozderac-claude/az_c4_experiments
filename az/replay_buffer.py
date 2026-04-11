import numpy as np
from collections import deque
from typing import List, Tuple


class ReplayBuffer:
    """
    Circular buffer of (board, policy, value) training examples.

    Data augmentation (horizontal flip) is applied automatically on insertion,
    doubling each game's contribution to the buffer.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        # Deque with maxlen enforces the circular behaviour automatically
        self._buf: deque = deque(maxlen=max_size)

    def add_game(self, game_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Store a completed game.

        Each element of game_data is (canonical_board, mcts_policy, values):
          canonical_board: (3, ROWS, COLS) float32
          mcts_policy:     (COLS,) float32  -- normalised visit counts
          values:          (NUM_VALUE_HEADS,) float32  -- one target per head
        """
        for board, policy, values in game_data:
            values = np.asarray(values, dtype=np.float32)
            self._buf.append((board, policy, values))
            # Horizontal flip augmentation (Connect4 is left-right symmetric)
            flipped_board  = board[:, :, ::-1].copy()
            flipped_policy = policy[::-1].copy()
            self._buf.append((flipped_board, flipped_policy, values))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch.

        Returns
        -------
        boards   : (batch, 3, ROWS, COLS) float32
        policies : (batch, COLS) float32
        values   : (batch,) float32
        """
        n = len(self._buf)
        assert n >= batch_size, f"Buffer has {n} samples, need {batch_size}"
        idx = np.random.choice(n, batch_size, replace=False)
        boards, policies, values = zip(*[self._buf[i] for i in idx])
        return (
            np.stack(boards).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.stack(values).astype(np.float32),   # (batch, NUM_VALUE_HEADS)
        )

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int) -> bool:
        return len(self._buf) >= min_size

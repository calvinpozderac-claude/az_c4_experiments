"""
Off-path buffer: deduplicated rolling buffer of MCTS off-path node positions.

Purpose
-------
Self-play games generate MCTS trees containing far more visited positions
than the handful that end up on the actual game line.  Any non-root node
with N ≥ min_visits (default 25) has a reliable Q estimate and minimax
targets from its subtree, even without a game-outcome label.

This buffer stores those positions with *partial* training targets:
  values[0:2] = NaN   (no game-outcome z known)
  values[2:8] = MCTS-derived targets (Q, minimax d1/d2/d3, minimax-Q, proven)

During training the off-path batch is supplemented to the main batch and
a masked MSE loss skips NaN targets, so only heads 2-7 receive gradient
from these positions.

Deduplication, capacity, and FIFO eviction are identical to EndgameBuffer.
Policy targets (visit distributions over children) are also stored so the
policy head can be trained on these additional positions.
"""

from az.endgame_buffer import EndgameBuffer
from typing import List, Tuple
import numpy as np


class OffPathBuffer(EndgameBuffer):
    """
    Deduplicated rolling buffer of off-path MCTS node positions.

    Interface is identical to EndgameBuffer except positions are added via
    add_positions() rather than add_game().

    Parameters
    ----------
    max_size : int
        Maximum number of unique positions stored (FIFO eviction).
    """

    def add_positions(
        self,
        positions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> int:
        """
        Add off-path positions from collect_off_path_nodes().

        Parameters
        ----------
        positions : list of (canonical_board, policy, values)
            values has NaN in heads 0 and 1 (no game outcome available).

        Returns
        -------
        int  — number of new (non-duplicate) positions inserted.
        """
        added = 0
        for board, policy, values in positions:
            if self._add_one(board, policy, values):
                added += 1
        return added

"""
Monte Carlo Tree Search for AlphaZero.

Value convention
----------------
Each MCTSNode stores Q from its *own* current_player's perspective:
  Q > 0  →  the player about to move at this node is winning.

Selection at a parent node uses  -child.Q + U  (negate because child's
current_player is the opponent of the parent's current_player).

Backup propagates the value up the tree, negating at each level to
convert between the two players' perspectives.

Extra value targets
-------------------
After each run() the root tree is used to compute five additional
training targets that are stored in self._last_root and can be retrieved
via compute_value_targets().  These are fed to the five auxiliary value
heads in AlphaZeroNet.

  mcts_q         : root.Q after the search (W/N aggregated over all sims)
  minimax_net_d1 : 1-ply minimax over per-node network evaluations
  minimax_net_d2 : 2-ply minimax over per-node network evaluations
  minimax_net_d3 : 3-ply minimax over per-node network evaluations
  minimax_q_n10  : recursive minimax over Q values (nodes with N ≥ 10)

All targets are in [−1, 1] from the root's current_player's perspective.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from c4.game import Connect4, COLS


class MCTSNode:
    __slots__ = [
        "state", "parent", "action", "prior",
        "children", "N", "W", "Q",
        "is_expanded", "is_terminal", "terminal_value",
        "value_net",   # network's raw evaluation when this node was first expanded
    ]

    def __init__(
        self,
        state: Connect4,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state  = state
        self.parent = parent
        self.action = action
        self.prior  = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.N: int   = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.is_expanded: bool  = False
        self.is_terminal: bool  = state.game_over
        self.value_net:   float = 0.0   # set by _expand(); used for minimax targets

        if self.is_terminal:
            if state.winner == -state.current_player:
                self.terminal_value = -1.0
            elif state.winner == state.current_player:
                self.terminal_value = 1.0
            else:
                self.terminal_value = 0.0
            # Treat terminal_value as the "network evaluation" for minimax
            self.value_net = self.terminal_value
        else:
            self.terminal_value = 0.0


class MCTS:
    """AlphaZero MCTS."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        c_puct: float = 1.5,
        num_simulations: int = 200,
    ):
        self.network        = network
        self.device         = device
        self.c_puct         = c_puct
        self.num_simulations = num_simulations
        self._last_root: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        root_state: Connect4,
        temperature: float = 1.0,
        add_dirichlet: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from root_state.

        Returns
        -------
        action_probs : (COLS,) visit-count distribution (sums to 1 over valid moves)
        root_value   : float — root.Q after search (head-0 value estimate)
        """
        if root_state.game_over:
            self._last_root = None
            return np.zeros(COLS, dtype=np.float32), 0.0

        root = MCTSNode(root_state.clone())

        # First expansion (counts as one simulation)
        init_value = self._expand(root)
        self._backup(root, init_value)

        if add_dirichlet and root.children:
            self._add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

        for _ in range(self.num_simulations - 1):
            node = root
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
            value = node.terminal_value if node.is_terminal else self._expand(node)
            self._backup(node, value)

        # Save root for compute_value_targets()
        self._last_root = root

        visit_counts = np.array(
            [root.children[a].N if a in root.children else 0 for a in range(COLS)],
            dtype=np.float32,
        )

        if temperature == 0:
            best = int(np.argmax(visit_counts))
            action_probs = np.zeros(COLS, dtype=np.float32)
            action_probs[best] = 1.0
        else:
            counts_t = visit_counts ** (1.0 / temperature)
            total    = counts_t.sum()
            action_probs = counts_t / (total + 1e-8)

        return action_probs, root.Q

    def compute_value_targets(self, min_visits_q: int = 10) -> np.ndarray:
        """
        Compute the five MCTS-derived value targets from the most recent run().

        Must be called immediately after run() (before the next run()).

        Parameters
        ----------
        min_visits_q : int
            Minimum N for a node to be included in the minimax-Q traversal.

        Returns
        -------
        targets : (5,) float32 array
            [mcts_q, minimax_net_d1, minimax_net_d2, minimax_net_d3, minimax_q_n10]
            all from the root position's current_player's perspective.
        """
        if self._last_root is None:
            return np.zeros(5, dtype=np.float32)

        root = self._last_root
        return np.array([
            root.Q,
            self._minimax_net(root, depth=1),
            self._minimax_net(root, depth=2),
            self._minimax_net(root, depth=3),
            self._minimax_q(root, min_visits=min_visits_q),
        ], dtype=np.float32)

    def get_best_move(self, state: Connect4) -> int:
        """Greedy best move (temperature=0, no noise)."""
        action_probs, _ = self.run(state, temperature=0, add_dirichlet=False)
        valid = state.get_valid_moves()
        return max(valid, key=lambda m: action_probs[m])

    # ------------------------------------------------------------------
    # Extra target helpers
    # ------------------------------------------------------------------

    def _minimax_net(self, node: MCTSNode, depth: int) -> float:
        """
        Minimax over network evaluations stored at each expanded node.

        Only children that were visited (N > 0) are eligible — unvisited
        children have value_net = 0 (uninformative default) and are skipped.

        Returns the value from node's current_player's perspective.
        """
        if node.is_terminal:
            return node.terminal_value

        if depth == 0:
            return node.value_net

        # Children evaluated by the network: visited (N > 0) and either
        # terminal (value_net set to terminal_value) or expanded by network.
        candidates = [
            c for c in node.children.values()
            if c.N > 0 and (c.is_expanded or c.is_terminal)
        ]

        if not candidates:
            return node.value_net   # no data: fall back to own estimate

        # child.value_net is from child's player perspective (opponent) → negate
        return max(-self._minimax_net(c, depth - 1) for c in candidates)

    def _minimax_q(self, node: MCTSNode, min_visits: int) -> float:
        """
        Recursive minimax over Q values, restricted to nodes with N ≥ min_visits.

        When a node has no children meeting the threshold the recursion stops
        and the node's own Q is returned as the leaf estimate.

        Returns the value from node's current_player's perspective.
        """
        if node.is_terminal:
            return node.terminal_value

        reliable = [c for c in node.children.values() if c.N >= min_visits]

        if not reliable:
            # Leaf of the reliable subtree: node's Q is the best available estimate
            return node.Q

        # child.Q is from child's player perspective (opponent) → negate
        return max(-self._minimax_q(c, min_visits) for c in reliable)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_N = math.sqrt(node.N)
        best_score  = -math.inf
        best_child: Optional[MCTSNode] = None

        for child in node.children.values():
            q     = -child.Q
            u     = self.c_puct * child.prior * sqrt_N / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """
        Evaluate node with the network, create child nodes.
        Stores the head-0 (game-outcome) value in node.value_net.
        Returns that value (used by MCTS backup).
        """
        state   = node.state
        board_t = torch.from_numpy(state.get_canonical_board()).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, values = self.network(board_t)
            # values: (1, NUM_VALUE_HEADS) — use head 0 for MCTS tree search
            policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            value  = float(values[0, 0].item())

        node.value_net = value   # store for minimax target computation

        valid_moves = state.get_valid_moves()
        policy_masked = np.zeros(COLS, dtype=np.float32)
        for m in valid_moves:
            policy_masked[m] = policy[m]
        total = policy_masked.sum()
        if total > 1e-8:
            policy_masked /= total
        else:
            for m in valid_moves:
                policy_masked[m] = 1.0 / len(valid_moves)

        for action in valid_moves:
            child_state = state.clone()
            child_state.make_move(action)
            node.children[action] = MCTSNode(
                child_state, parent=node, action=action, prior=policy_masked[action]
            )

        node.is_expanded = True
        return value

    def _backup(self, node: MCTSNode, value: float):
        current: Optional[MCTSNode] = node
        while current is not None:
            current.N += 1
            current.W += value
            current.Q  = current.W / current.N
            value      = -value
            current    = current.parent

    def _add_dirichlet_noise(self, node: MCTSNode, alpha: float, epsilon: float):
        children = list(node.children.values())
        k = len(children)
        if k == 0:
            return
        noise = np.random.dirichlet([alpha] * k).astype(np.float32)
        for child, eta in zip(children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * eta

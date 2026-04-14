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
After each run() the root tree is used to compute six training targets
that feed the auxiliary value heads in AlphaZeroNet:

  mcts_q         : root.Q after the search (W/N aggregated over all sims)
  minimax_net_d1 : 1-ply minimax over per-node network evaluations
  minimax_net_d2 : 2-ply minimax over per-node network evaluations
  minimax_net_d3 : 3-ply minimax over per-node network evaluations
  minimax_q_n10  : recursive minimax over Q values (nodes with N ≥ 10)
  proven_minimax : proven value from terminal backprop (NaN if unproven)

All targets are in [−1, 1] from the root's current_player's perspective.

Proven values
-------------
During backup, when a terminal node is reached, its proven_value (±1 or 0)
is propagated upward through ancestors:
  - A node is proven won  (+1) if any child has proven_value < 0
    (child is proven lost → parent can force a win by choosing that child)
  - A node is proven lost (−1) if ALL children have been expanded and
    every one has proven_value > 0  (all lines lose for current player)
  - A node is proven drawn (0) if all children are proven and the best
    is a draw
Proven wins propagate immediately; proven losses require all children
explored (rare in partial MCTS, common near game end).

Off-path node collection
-------------------------
collect_off_path_nodes() traverses the last tree after run() and returns
(board, policy, partial_targets) for every non-root, non-terminal node
with N >= min_visits.  These positions are added to a separate off-path
buffer and supplement the main replay buffer during training.  The
game-outcome targets (heads 0 and 1) are NaN for off-path nodes; the
MCTS-derived targets (heads 2-7) are computed from each node's subtree.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from c4.game import Connect4, COLS


class MCTSNode:
    __slots__ = [
        "state", "parent", "action", "prior",
        "children", "N", "W", "Q",
        "is_expanded", "is_terminal", "terminal_value",
        "value_net",      # network's raw evaluation when this node was first expanded
        "proven_value",   # None = unproven; float = proven optimal value (±1 or 0)
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
            self.value_net    = self.terminal_value
            self.proven_value = self.terminal_value   # terminal = immediately proven
        else:
            self.terminal_value = 0.0
            self.proven_value   = None                # not yet proven


class MCTS:
    """AlphaZero MCTS with proven-value tracking and off-path node collection."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        c_puct: float = 1.5,
        num_simulations: int = 200,
    ):
        self.network         = network
        self.device          = device
        self.c_puct          = c_puct
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

        # Save root for compute_value_targets() and collect_off_path_nodes()
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
        Compute the six MCTS-derived value targets from the most recent run().
        Maps to auxiliary heads 2-7 in AlphaZeroNet (heads 0 and 1 are z targets
        assigned separately by the trainer).

        Must be called immediately after run() (before the next run()).

        Returns
        -------
        targets : (6,) float32 array
            [mcts_q, minimax_net_d1, minimax_net_d2, minimax_net_d3,
             minimax_q_n10, proven_minimax]
            all from the root's current_player's perspective.
            proven_minimax is NaN if the root position is not yet proven.
        """
        if self._last_root is None:
            return np.array([0., 0., 0., 0., 0., np.nan], dtype=np.float32)

        root = self._last_root
        proven = root.proven_value if root.proven_value is not None else np.nan
        return np.array([
            root.Q,
            self._minimax_net(root, depth=1),
            self._minimax_net(root, depth=2),
            self._minimax_net(root, depth=3),
            self._minimax_q(root, min_visits=min_visits_q),
            proven,
        ], dtype=np.float32)

    def collect_off_path_nodes(
        self,
        min_visits: int = 25,
        num_value_heads: int = 8,
        min_visits_q: int = 10,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Collect off-path (non-root) nodes with N >= min_visits from the last tree.

        These positions are not on the main self-play line but were explored
        sufficiently to have reliable Q and minimax estimates.  They are returned
        as training triples with:
          - values[0:2] = NaN   (no game-outcome z available)
          - values[2]   = node.Q                    (mcts_q analog)
          - values[3:6] = minimax_net d1/d2/d3       from this node's subtree
          - values[6]   = minimax_q N>=10            from this node's subtree
          - values[7]   = proven_value or NaN

        Returns
        -------
        list of (canonical_board, policy, values):
          canonical_board : (3, ROWS, COLS) float32
          policy          : (COLS,) float32  — normalized visit distribution
          values          : (num_value_heads,) float32  — partial targets w/ NaN
        """
        if self._last_root is None:
            return []

        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # BFS from root's children (exclude root — it's on the main line)
        stack = list(self._last_root.children.values())

        while stack:
            node = stack.pop()

            if node.N >= min_visits and not node.is_terminal:
                board = node.state.get_canonical_board()

                # Policy: normalized visit distribution over node's children
                policy = np.zeros(COLS, dtype=np.float32)
                total_child_n = sum(c.N for c in node.children.values())
                if total_child_n > 0:
                    for action, child in node.children.items():
                        policy[action] = child.N / total_child_n

                # Targets: NaN for game-outcome heads (0,1); computed for rest
                targets = np.full(num_value_heads, np.nan, dtype=np.float32)
                # heads 0,1 remain NaN — no game-outcome label for off-path nodes
                targets[2] = node.Q
                targets[3] = self._minimax_net(node, depth=1)
                targets[4] = self._minimax_net(node, depth=2)
                targets[5] = self._minimax_net(node, depth=3)
                targets[6] = self._minimax_q(node, min_visits=min_visits_q)
                if node.proven_value is not None:
                    targets[7] = node.proven_value

                results.append((board, policy, targets))

            # Continue DFS into children regardless of N threshold
            stack.extend(node.children.values())

        return results

    def get_best_move(self, state: Connect4) -> int:
        """Greedy best move (temperature=0, no noise)."""
        action_probs, _ = self.run(state, temperature=0, add_dirichlet=False)
        valid = state.get_valid_moves()
        return max(valid, key=lambda m: action_probs[m])

    # ------------------------------------------------------------------
    # Proven value propagation
    # ------------------------------------------------------------------

    def _update_proven(self, node: MCTSNode) -> None:
        """
        Recompute node.proven_value based on its children's proven values.

        Rules (values from current node's player perspective):
          • Any child with proven_value < 0  → current player can force a win
            (that child's current player — the opponent — is proven to lose)
          • All children proven AND max(-child.proven) is the best achievable
            → node proven to that value (draw or loss)

        Only called on non-terminal expanded nodes.
        """
        if node.is_terminal or not node.children:
            return

        proven_children = [c for c in node.children.values()
                           if c.proven_value is not None]
        if not proven_children:
            return

        # Best achievable: negate child's value (opponent → current player)
        best = max(-c.proven_value for c in proven_children)

        if best > 0.5:
            # Found a proven win — propagate immediately
            node.proven_value = 1.0
        elif node.is_expanded and len(proven_children) == len(node.children):
            # All children proven → parent is fully resolved
            node.proven_value = float(best)   # 0.0 (draw) or −1.0 (loss)

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

        candidates = [
            c for c in node.children.values()
            if c.N > 0 and (c.is_expanded or c.is_terminal)
        ]

        if not candidates:
            return node.value_net

        return max(-self._minimax_net(c, depth - 1) for c in candidates)

    def _minimax_q(self, node: MCTSNode, min_visits: int) -> float:
        """
        Recursive minimax over Q values, restricted to nodes with N ≥ min_visits.

        Returns the value from node's current_player's perspective.
        """
        if node.is_terminal:
            return node.terminal_value

        reliable = [c for c in node.children.values() if c.N >= min_visits]

        if not reliable:
            return node.Q

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

    def _backup(self, node: MCTSNode, value: float) -> None:
        """
        Propagate value from node to root, negating at each level.
        Also propagates proven_value upward after each Q update.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.N += 1
            current.W += value
            current.Q  = current.W / current.N
            # After updating this node, recheck if its proven_value should change
            self._update_proven(current)
            value   = -value
            current = current.parent

    def _add_dirichlet_noise(self, node: MCTSNode, alpha: float, epsilon: float):
        children = list(node.children.values())
        k = len(children)
        if k == 0:
            return
        noise = np.random.dirichlet([alpha] * k).astype(np.float32)
        for child, eta in zip(children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * eta

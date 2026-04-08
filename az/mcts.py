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
    ]

    def __init__(
        self,
        state: Connect4,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = state.game_over

        # Terminal value from state.current_player's perspective.
        # After a winning move: state.winner = prev_player = -state.current_player.
        # So the player to move lost → terminal_value = -1.
        if self.is_terminal:
            if state.winner == -state.current_player:
                self.terminal_value = -1.0   # current player lost
            elif state.winner == state.current_player:
                self.terminal_value = 1.0    # shouldn't occur in normal play
            else:
                self.terminal_value = 0.0    # draw
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
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations

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
        action_probs : np.ndarray of shape (COLS,)
            Visit-count-based action distribution (sums to 1 over valid moves).
        root_value : float
            Value estimate from root's current_player's perspective.
        """
        if root_state.game_over:
            return np.zeros(COLS, dtype=np.float32), 0.0

        root = MCTSNode(root_state.clone())

        # First expansion (counts as one simulation)
        init_value = self._expand(root)
        self._backup(root, init_value)

        # Optional Dirichlet noise on root children for exploration during self-play
        if add_dirichlet and root.children:
            self._add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

        # Remaining simulations
        for _ in range(self.num_simulations - 1):
            node = root
            # Selection: walk to a leaf
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
            # Evaluation
            value = node.terminal_value if node.is_terminal else self._expand(node)
            # Backup
            self._backup(node, value)

        # Build action probability vector from visit counts
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
            total = counts_t.sum()
            action_probs = counts_t / (total + 1e-8)

        return action_probs, root.Q

    def get_best_move(self, state: Connect4) -> int:
        """Greedy best move (temperature=0, no noise)."""
        action_probs, _ = self.run(state, temperature=0, add_dirichlet=False)
        valid = state.get_valid_moves()
        return max(valid, key=lambda m: action_probs[m])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        PUCT selection from node's perspective.

        child.Q is from child's current_player's perspective (= node's opponent),
        so we negate it to get node's player's perspective before adding U.
        """
        sqrt_N = math.sqrt(node.N)
        best_score = -math.inf
        best_child: Optional[MCTSNode] = None

        for child in node.children.values():
            q = -child.Q  # flip to parent's perspective
            u = self.c_puct * child.prior * sqrt_N / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """
        Evaluate node with the network, create child nodes.

        Returns the value estimate from node.state.current_player's perspective.
        """
        state = node.state
        board_t = torch.from_numpy(state.get_canonical_board()).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value_t = self.network(board_t)
            policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            value = float(value_t.item())

        valid_moves = state.get_valid_moves()

        # Mask invalid moves and renormalise
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
        """
        Propagate value up the tree.

        value is from node.state.current_player's perspective.
        We store Q at each node from that node's own current_player's perspective,
        negating once per level because the player alternates.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.N += 1
            current.W += value
            current.Q = current.W / current.N
            value = -value          # Flip for the parent (opposite player)
            current = current.parent

    def _add_dirichlet_noise(
        self, node: MCTSNode, alpha: float, epsilon: float
    ):
        children = list(node.children.values())
        k = len(children)
        if k == 0:
            return
        noise = np.random.dirichlet([alpha] * k).astype(np.float32)
        for child, eta in zip(children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * eta

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

Persistent tree reuse
---------------------
MCTS maintains two internal pointers:

  _batch_root     – the root node for the entire self-play batch (the empty
                    board state).  Persists across games within one iteration.
                    Cleared by clear_retained_root() between iterations.

  _current_node   – tracks the node corresponding to the *current game state*
                    within the batch root's tree.

At the start of each game call reset_to_game_root(weight).  After every move
call descend(action) so the tree pointer stays aligned with the board.

All MCTS simulations are then added *on top of* the existing subtree for the
current position rather than starting from zero, and c_puct_bonus increases
exploration to avoid re-visiting already well-sampled branches.
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
    """
    AlphaZero MCTS with persistent tree reuse across self-play games.

    Tree lifetime
    -------------
    - clear_retained_root()      : discard everything (call before each iteration)
    - reset_to_game_root(weight) : navigate back to empty-board root, optionally
                                   discount stats by weight (call at game start)
    - descend(action)            : advance the tree pointer by one move (call
                                   after every game.make_move())
    - run(...)                   : run simulations from the current tracked node
                                   (or from a fresh root if none is tracked)
    """

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

        # Batch-level root (the empty-board node for the whole self-play batch)
        self._batch_root: Optional[MCTSNode] = None
        # Current position pointer within the batch tree
        self._current_node: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Tree lifetime management
    # ------------------------------------------------------------------

    def clear_retained_root(self):
        """Discard the retained tree entirely (call between training iterations)."""
        self._batch_root = None
        self._current_node = None

    def reset_to_game_root(self, tree_reuse_weight: float = 1.0):
        """
        Call at the start of each self-play game.

        Scales the batch root's subtree stats by tree_reuse_weight (useful to
        discount old experience; 1.0 keeps everything unchanged), then resets
        the current-node pointer back to the batch root so the new game starts
        from the beginning of the tree.
        """
        if self._batch_root is not None:
            self._scale_tree(self._batch_root, tree_reuse_weight)
        self._current_node = self._batch_root  # None for the very first game

    def descend(self, action: int):
        """
        Advance the current-node pointer to the child reached by `action`.

        Call after every game.make_move() to keep the tree pointer in sync
        with the actual board state.  If the child hasn't been expanded yet
        (the game took a path MCTS barely visited), the pointer becomes None
        and the next run() call starts a fresh sub-tree from that position.
        """
        if self._current_node is not None:
            self._current_node = self._current_node.children.get(action)

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
        c_puct_bonus: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from root_state.

        If a current node is being tracked (via reset_to_game_root /
        descend) it is used as the root and `num_simulations` additional
        simulations are added on top.  Otherwise a fresh root is created.

        c_puct_bonus is added to self.c_puct for this call, which increases
        the U term in PUCT and pushes exploration toward less-visited nodes.

        Returns
        -------
        action_probs : np.ndarray of shape (COLS,)
        root_value   : float
        """
        if root_state.game_over:
            return np.zeros(COLS, dtype=np.float32), 0.0

        effective_c_puct = self.c_puct + c_puct_bonus

        if self._current_node is not None and not self._current_node.is_terminal:
            # --- Warm-start: reuse the accumulated sub-tree -----------------
            root = self._current_node
            # Do NOT detach root from its parent: backups should propagate all
            # the way up to the batch root so ancestor Q values stay accurate.

            # Re-sample Dirichlet so fresh games don't replay prior noise.
            if add_dirichlet and root.children:
                self._add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

            # All simulations are *additive* on top of existing visit counts.
            for _ in range(self.num_simulations):
                node = root
                while node.is_expanded and not node.is_terminal:
                    node = self._select_child(node, effective_c_puct)
                value = node.terminal_value if node.is_terminal else self._expand(node)
                self._backup(node, value)

        else:
            # --- Cold-start: build a fresh tree from root_state -------------
            root = MCTSNode(root_state.clone())

            init_value = self._expand(root)
            self._backup(root, init_value)

            if add_dirichlet and root.children:
                self._add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

            for _ in range(self.num_simulations - 1):
                node = root
                while node.is_expanded and not node.is_terminal:
                    node = self._select_child(node, effective_c_puct)
                value = node.terminal_value if node.is_terminal else self._expand(node)
                self._backup(node, value)

            # First run in this batch: pin the batch root.
            if self._batch_root is None:
                self._batch_root = root

        # Keep the current-node pointer pointing at this root so that
        # descend() can navigate to the chosen child after the move.
        self._current_node = root

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
        """Greedy best move (temperature=0, no noise, no tree reuse)."""
        # Save and restore the retained-tree state so evaluation calls don't
        # corrupt the self-play batch root or current-node pointer.
        saved_batch = self._batch_root
        saved_current = self._current_node
        self._batch_root = None
        self._current_node = None
        action_probs, _ = self.run(state, temperature=0, add_dirichlet=False)
        self._batch_root = saved_batch
        self._current_node = saved_current
        valid = state.get_valid_moves()
        return max(valid, key=lambda m: action_probs[m])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_tree(self, node: MCTSNode, weight: float):
        """
        Recursively scale N and W statistics by weight.

        weight=1.0 is a no-op (fast early return).
        weight<1.0 discounts old experience so the new iteration's simulations
        carry more relative weight in the PUCT formula.
        """
        if weight == 1.0:
            return
        node.N = int(node.N * weight)
        node.W = node.W * weight
        node.Q = node.W / node.N if node.N > 0 else 0.0
        for child in node.children.values():
            self._scale_tree(child, weight)

    def _select_child(self, node: MCTSNode, c_puct: float) -> MCTSNode:
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
            u = c_puct * child.prior * sqrt_N / (1 + child.N)
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

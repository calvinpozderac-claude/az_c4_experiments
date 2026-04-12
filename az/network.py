import torch
import torch.nn as nn
import torch.nn.functional as F

from c4.game import ROWS, COLS, ACTION_SIZE


def _make_norm(norm_type: str, channels: int, h: int = ROWS, w: int = COLS) -> nn.Module:
    """
    Return a normalisation layer compatible with the active backend.

    norm_type = "batch"  →  BatchNorm2d  (default; not supported on DirectML)
    norm_type = "layer"  →  LayerNorm([C, H, W])  (works everywhere)
    """
    if norm_type == "layer":
        return nn.LayerNorm([channels, h, w])
    return nn.BatchNorm2d(channels)


class ResBlock(nn.Module):
    """Residual block with pluggable normalisation (BatchNorm or LayerNorm)."""

    def __init__(self, channels: int, norm_type: str = "batch"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class ValueHead(nn.Module):
    """
    One value head: 1-filter conv → norm → ReLU → flatten → linear(64) → tanh → scalar.
    All auxiliary value heads share this architecture but have fully independent weights.
    """

    def __init__(self, num_channels: int, norm_type: str = "batch"):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.norm = _make_norm(norm_type, 1)
        self.fc1  = nn.Linear(ROWS * COLS, 64)
        self.fc2  = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.norm(self.conv(x)))
        v = F.relu(self.fc1(v.flatten(1)))
        return torch.tanh(self.fc2(v))   # (batch, 1)


class LayeredMetaHead(nn.Module):
    """
    Layered (pure-stack) meta head for game outcome prediction.

    Sees ONLY the 5 auxiliary head outputs — no direct access to tower features.
    All board information must pass through the aux heads first, creating a clean
    information bottleneck:

        board → tower → aux heads (stop-grad) → LayeredMetaHead → game_outcome

    The caller is responsible for passing aux_values detached (.detach()) so
    that the game-outcome loss does not flow back into the aux heads or tower.

    Gradient flow
    -------------
    Meta loss → LayeredMetaHead weights only.
    Tower gets ZERO gradient from the game-outcome objective; it is trained
    entirely by the policy loss and the 5 auxiliary head MSE losses.

    Architecture
    ------------
    FC(5, 16) → ReLU → FC(16, 1) → tanh
    """

    NUM_AUX = 5

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.NUM_AUX, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, aux_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        aux_values : (B, 5)  — must be detached before calling (stop-gradient)

        Returns
        -------
        (B, 1)  tanh-clamped game-outcome estimate
        """
        h = F.relu(self.fc1(aux_values))
        return torch.tanh(self.fc2(h))


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style network for Connect Four with a layered (pure-stack)
    value architecture.

    Input : (batch, 3, ROWS, COLS)
    Output: policy_logits  (batch, ACTION_SIZE)
            values         (batch, NUM_VALUE_HEADS)  — one tanh scalar per head

    Value head semantics (canonical output order)
    ---------------------------------------------
    Head 0  game_outcome   : LayeredMetaHead — predicts final game result (+1/0/-1)
                             by learning a nonlinear combination of the 5 aux heads.
                             Sees NO tower features directly; board knowledge comes
                             entirely through the auxiliary head outputs.
                             Stop-gradient separates this head from aux objectives.
                             THIS is the head used inside the MCTS tree search.
    Head 1  mcts_q         : MCTS root Q = W/N after the search completes
    Head 2  minimax_net_d1 : 1-ply minimax over per-node network evaluations
    Head 3  minimax_net_d2 : 2-ply minimax over per-node network evaluations
    Head 4  minimax_net_d3 : 3-ply minimax over per-node network evaluations
    Head 5  minimax_q_n10  : recursive minimax over Q values (nodes with N ≥ 10)

    Training targets
    ----------------
    Head 0 (meta)   : true game outcome z ∈ {-1, 0, +1} from self-play
    Heads 1-5 (aux) : MCTS tree statistics (see compute_value_targets in mcts.py)

    Gradient flow
    -------------
    Policy loss         → policy head + tower (full gradient)
    Aux head MSE losses → aux heads + tower (full gradient)
    Meta head MSE loss  → LayeredMetaHead weights only
                          stop_grad(aux_values) blocks all gradient to aux heads
                          and to the tower from the game-outcome objective

    Comparison with GatedMetaHead branch
    -------------------------------------
    The Gated branch also passes tower features directly into the meta head
    (via conv+FC paths), so the tower does receive gradient from game_outcome.
    This branch enforces a strict bottleneck: game_outcome can only influence
    the network by adjusting how the 5 specialists' opinions are combined.
    """

    NUM_VALUE_HEADS = 6   # 1 meta + 5 aux

    VALUE_HEAD_NAMES = [
        "game_outcome",    # [0] meta head (LayeredMetaHead)
        "mcts_q",          # [1] aux
        "minimax_net_d1",  # [2] aux
        "minimax_net_d2",  # [3] aux
        "minimax_net_d3",  # [4] aux
        "minimax_q_n10",   # [5] aux
    ]

    # Names of the 5 auxiliary heads (indices 1-5 in VALUE_HEAD_NAMES)
    AUX_HEAD_NAMES = VALUE_HEAD_NAMES[1:]

    def __init__(
        self,
        num_res_blocks: int = 5,
        num_channels: int = 64,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.norm_type = norm_type

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, num_channels),
            nn.ReLU(),
        )

        # Residual tower — shared by all heads and the policy head
        self.tower = nn.Sequential(
            *[ResBlock(num_channels, norm_type) for _ in range(num_res_blocks)]
        )

        # Policy head: 2-filter conv → flatten → FC
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_norm = _make_norm(norm_type, 2)
        self.policy_fc   = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)

        # Five auxiliary value heads (heads 1-5): independent ValueHead instances
        self.aux_value_heads = nn.ModuleList([
            ValueHead(num_channels, norm_type)
            for _ in range(len(self.AUX_HEAD_NAMES))
        ])

        # Layered meta head (head 0): combines aux outputs only — no tower access
        self.meta_head = LayeredMetaHead()

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)
        values        : (batch, NUM_VALUE_HEADS)   each in (−1, 1)
                        values[:, 0] = game_outcome (LayeredMetaHead)
                        values[:, 1:] = aux heads
        """
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_norm(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        # Auxiliary heads (1-5): each produces (batch, 1); cat → (batch, 5)
        aux_values = torch.cat(
            [head(x) for head in self.aux_value_heads], dim=1
        )

        # Meta head (0): stop-gradient on aux_values so neither the game_outcome
        # loss nor its gradient reaches aux heads or the tower
        game_outcome = self.meta_head(aux_values.detach())  # (batch, 1)

        # Canonical order: [game_outcome, mcts_q, mm_d1, mm_d2, mm_d3, mm_q_n10]
        values = torch.cat([game_outcome, aux_values], dim=1)  # (batch, 6)

        return policy_logits, values

    @torch.no_grad()
    def predict(self, board_tensor: torch.Tensor):
        """
        Single-sample inference.

        Args:
            board_tensor: (3, ROWS, COLS) float tensor on the right device
        Returns:
            policy_probs : (ACTION_SIZE,) tensor
            values       : (NUM_VALUE_HEADS,) tensor
        """
        self.eval()
        policy_logits, values = self(board_tensor.unsqueeze(0))
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        return policy_probs, values.squeeze(0)

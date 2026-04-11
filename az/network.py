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


class GatedMetaHead(nn.Module):
    """
    Gated Mixture-of-Experts meta head for game outcome prediction.

    Takes tower features and the (detached) outputs of the 5 auxiliary value heads.
    Learns per-position weights over the aux heads (the MoE gate) and adds a small
    residual correction read directly from the tower, then produces a tanh scalar.

    The caller is responsible for passing aux_values with stop-gradient applied
    (.detach()) so that the game-outcome loss does not corrupt the aux head objectives.

    Architecture
    ------------
    Gate path  : tower → conv(1) → norm → ReLU → flatten
                 cat with aux_values → FC(H*W+5, 5) → softmax → weights
    Residual   : tower → conv(1) → norm → ReLU → flatten → FC(H*W,32) → ReLU → FC(32,1)
    Output     : tanh(weighted_sum + 0.1 * residual)
    """

    NUM_AUX = 5   # fixed: heads 1-5 in the canonical ordering

    def __init__(self, num_channels: int, norm_type: str = "batch"):
        super().__init__()
        flat = ROWS * COLS

        # Gate pathway
        self.gate_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.gate_norm = _make_norm(norm_type, 1)
        self.gate_fc   = nn.Linear(flat + self.NUM_AUX, self.NUM_AUX)

        # Residual correction pathway
        self.res_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.res_norm = _make_norm(norm_type, 1)
        self.res_fc1  = nn.Linear(flat, 32)
        self.res_fc2  = nn.Linear(32, 1)

    def forward(
        self, tower_feat: torch.Tensor, aux_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tower_feat : (B, C, H, W)  — residual tower output; full gradient flows here
        aux_values : (B, 5)        — detached aux head outputs (stop-gradient)

        Returns
        -------
        (B, 1)  tanh-clamped game-outcome estimate
        """
        # Gate
        g       = F.relu(self.gate_norm(self.gate_conv(tower_feat)))  # (B, 1, H, W)
        g_flat  = g.flatten(1)                                         # (B, H*W)
        gate_in = torch.cat([g_flat, aux_values], dim=1)               # (B, H*W+5)
        weights = torch.softmax(self.gate_fc(gate_in), dim=1)          # (B, 5)
        weighted = (weights * aux_values).sum(dim=1, keepdim=True)     # (B, 1)

        # Residual correction
        r        = F.relu(self.res_norm(self.res_conv(tower_feat)))    # (B, 1, H, W)
        r_flat   = r.flatten(1)                                         # (B, H*W)
        residual = torch.tanh(self.res_fc2(F.relu(self.res_fc1(r_flat))))  # (B, 1)

        return torch.tanh(weighted + 0.1 * residual)                   # (B, 1)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style network for Connect Four with a Gated Mixture-of-Experts
    value architecture.

    Input : (batch, 3, ROWS, COLS)
    Output: policy_logits  (batch, ACTION_SIZE)
            values         (batch, NUM_VALUE_HEADS)  — one tanh scalar per head

    Value head semantics (canonical output order)
    ---------------------------------------------
    Head 0  game_outcome   : GatedMetaHead — predicts final game result (+1/0/-1)
                             by learning per-position weights over the 5 aux heads
                             plus a residual correction from tower features.
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
    Aux head MSE losses → aux heads + shared tower (full gradient).
    Meta head MSE loss  → GatedMetaHead + shared tower (via gate/res paths only).
                          stop_grad(aux_values) prevents meta loss from
                          distorting the auxiliary head training objectives.
    """

    NUM_VALUE_HEADS = 6   # 1 meta + 5 aux

    VALUE_HEAD_NAMES = [
        "game_outcome",    # [0] meta head (GatedMetaHead)
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

        # Gated meta head (head 0): combines aux outputs + tower features
        self.meta_head = GatedMetaHead(num_channels, norm_type)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)
        values        : (batch, NUM_VALUE_HEADS)   each in (−1, 1)
                        values[:, 0] = game_outcome (meta head)
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

        # Meta head (0): stop-gradient on aux_values so game_outcome loss does
        # not flow back through the aux head parameters
        game_outcome = self.meta_head(x, aux_values.detach())  # (batch, 1)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from c4.game import ROWS, COLS, ACTION_SIZE


def _make_norm(norm_type: str, channels: int, h: int = ROWS, w: int = COLS) -> nn.Module:
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
    One value head: 1-filter conv → norm → ReLU → flatten → FC(64) → tanh → scalar.
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

    Takes tower features and the (detached) outputs of all auxiliary value
    heads.  Learns per-position weights over aux heads (the MoE gate) and
    adds a residual correction from tower features.

    The caller must pass aux_values already detached (.detach()) so the
    game-outcome loss does not corrupt auxiliary head objectives.

    Architecture
    ------------
    Gate path  : tower → conv(1) → norm → ReLU → flatten
                 cat with aux_values → FC(H*W + N_AUX, N_AUX) → softmax → weights
    Residual   : tower → conv(1) → norm → ReLU → flatten → FC(H*W, 32) → tanh → scalar
    Output     : tanh(Σ(weights · aux_values) + 0.1 * residual)
    """

    def __init__(self, num_channels: int, num_aux: int, norm_type: str = "batch"):
        super().__init__()
        flat = ROWS * COLS

        # Gate pathway
        self.gate_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.gate_norm = _make_norm(norm_type, 1)
        self.gate_fc   = nn.Linear(flat + num_aux, num_aux)

        # Residual correction pathway
        self.res_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.res_norm = _make_norm(norm_type, 1)
        self.res_fc1  = nn.Linear(flat, 32)
        self.res_fc2  = nn.Linear(32, 1)

    def forward(
        self, tower_feat: torch.Tensor, aux_values: torch.Tensor
    ) -> torch.Tensor:
        """
        tower_feat : (B, C, H, W)  — full gradient flows here
        aux_values : (B, N_AUX)    — must be detached before calling
        Returns    : (B, 1)
        """
        # Gate
        g       = F.relu(self.gate_norm(self.gate_conv(tower_feat)))
        g_flat  = g.flatten(1)
        gate_in = torch.cat([g_flat, aux_values], dim=1)
        weights = torch.softmax(self.gate_fc(gate_in), dim=1)
        weighted = (weights * aux_values).sum(dim=1, keepdim=True)

        # Residual
        r        = F.relu(self.res_norm(self.res_conv(tower_feat)))
        r_flat   = r.flatten(1)
        residual = torch.tanh(self.res_fc2(F.relu(self.res_fc1(r_flat))))

        return torch.tanh(weighted + 0.1 * residual)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero network with a rich multi-source value architecture.

    Input : (batch, 3, ROWS, COLS)
    Output: policy_logits  (batch, ACTION_SIZE)
            values         (batch, NUM_VALUE_HEADS)  each in (−1, 1)

    Value head layout (8 heads)
    ----------------------------
    Head 0  game_outcome   GatedMetaHead — combines all 7 aux heads + tower features
                           stop-grad at aux→meta boundary
                           TARGET: game outcome z ∈ {−1, 0, +1}
                           Used for MCTS tree search

    Head 1  z_direct       Independent ValueHead from tower
                           TARGET: same game outcome z
                           Gives the GatedMetaHead a "raw network opinion" as one
                           of its inputs; also ensures z gradients reach the tower

    Head 2  mcts_q         TARGET: Q at the root (or this node for off-path)
    Head 3  minimax_net_d1 TARGET: 1-ply minimax over network evaluations
    Head 4  minimax_net_d2 TARGET: 2-ply minimax
    Head 5  minimax_net_d3 TARGET: 3-ply minimax
    Head 6  minimax_q_n10  TARGET: minimax over Q (nodes with N ≥ 10)
    Head 7  proven_minimax TARGET: proven value from terminal backprop (NaN if none)

    Training data sources
    ----------------------
    Self-play main line   : all 8 heads (heads 0,1 get z; 2-7 get MCTS targets)
    Off-path nodes N≥25   : heads 2-7 only (heads 0,1 = NaN, no z available)
    Both use masked MSE — NaN targets are excluded from the loss.

    Gradient flow
    -------------
    z loss (heads 0,1)        → z_direct head + tower (full)
                                GatedMetaHead + tower (via gate/res paths only)
    Aux MSE losses (heads 2-7) → aux heads + tower (full)
    stop_grad(aux_values) in forward() isolates aux head objectives from z loss
    """

    NUM_VALUE_HEADS = 8

    VALUE_HEAD_NAMES = [
        "game_outcome",    # [0] GatedMetaHead — used in MCTS
        "z_direct",        # [1] independent z head
        "mcts_q",          # [2]
        "minimax_net_d1",  # [3]
        "minimax_net_d2",  # [4]
        "minimax_net_d3",  # [5]
        "minimax_q_n10",   # [6]
        "proven_minimax",  # [7]
    ]

    # Heads 1-7: fed into GatedMetaHead as aux inputs
    AUX_HEAD_NAMES = VALUE_HEAD_NAMES[1:]   # 7 names

    def __init__(
        self,
        num_res_blocks: int = 3,
        num_channels: int = 64,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.norm_type = norm_type
        _n_aux = len(self.AUX_HEAD_NAMES)   # 7

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, num_channels),
            nn.ReLU(),
        )

        # Residual tower — shared by all heads
        self.tower = nn.Sequential(
            *[ResBlock(num_channels, norm_type) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_norm = _make_norm(norm_type, 2)
        self.policy_fc   = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)

        # 7 auxiliary value heads (heads 1-7)
        self.aux_value_heads = nn.ModuleList([
            ValueHead(num_channels, norm_type) for _ in range(_n_aux)
        ])

        # GatedMetaHead (head 0): combines 7 aux outputs + tower
        self.meta_head = GatedMetaHead(num_channels, num_aux=_n_aux, norm_type=norm_type)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)
        values        : (batch, NUM_VALUE_HEADS)  — each in (−1, 1)
                        values[:, 0] = game_outcome (GatedMetaHead, used in MCTS)
                        values[:, 1:] = aux heads
        """
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_norm(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        # All 7 aux heads → (batch, 7)
        aux_values = torch.cat(
            [head(x) for head in self.aux_value_heads], dim=1
        )

        # Meta head: stop-grad so z loss cannot distort aux objectives
        game_outcome = self.meta_head(x, aux_values.detach())   # (batch, 1)

        values = torch.cat([game_outcome, aux_values], dim=1)   # (batch, 8)
        return policy_logits, values

    @torch.no_grad()
    def predict(self, board_tensor: torch.Tensor):
        """Single-sample inference."""
        self.eval()
        policy_logits, values = self(board_tensor.unsqueeze(0))
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        return policy_probs, values.squeeze(0)

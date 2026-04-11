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
    All six value heads share this architecture but have fully independent weights.
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


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-head network for Connect Four extended with six
    independent value heads that all branch off the same residual tower.

    Input : (batch, 3, ROWS, COLS)
    Output: policy_logits  (batch, ACTION_SIZE)
            values         (batch, NUM_VALUE_HEADS)  — one tanh scalar per head

    Value head semantics
    --------------------
    Head 0  game_outcome   : final self-play game result (+1/0/-1)
                             — the only head used inside the MCTS tree search
    Head 1  mcts_q         : MCTS root Q = W/N after the search completes
    Head 2  minimax_net_d1 : 1-ply minimax over per-node network evaluations
    Head 3  minimax_net_d2 : 2-ply minimax over per-node network evaluations
    Head 4  minimax_net_d3 : 3-ply minimax over per-node network evaluations
    Head 5  minimax_q_n10  : recursive minimax over Q values (nodes with N ≥ 10)

    All values represent the *current player's* expected outcome; positive
    means the current player is predicted to win.
    """

    NUM_VALUE_HEADS = 6

    VALUE_HEAD_NAMES = [
        "game_outcome",
        "mcts_q",
        "minimax_net_d1",
        "minimax_net_d2",
        "minimax_net_d3",
        "minimax_q_n10",
    ]

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

        # Residual tower — shared by all heads
        self.tower = nn.Sequential(
            *[ResBlock(num_channels, norm_type) for _ in range(num_res_blocks)]
        )

        # Policy head: 2-filter conv → flatten → FC
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_norm = _make_norm(norm_type, 2)
        self.policy_fc   = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)

        # Six independent value heads
        self.value_heads = nn.ModuleList([
            ValueHead(num_channels, norm_type)
            for _ in range(self.NUM_VALUE_HEADS)
        ])

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)
        values        : (batch, NUM_VALUE_HEADS)   each in (−1, 1)
        """
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_norm(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        # All value heads — each produces (batch, 1); cat → (batch, 6)
        values = torch.cat([head(x) for head in self.value_heads], dim=1)

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

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


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-head network for Connect Four.

    Input : (batch, 3, ROWS, COLS)  -- canonical board (see c4/game.py)
    Output: policy_logits (batch, ACTION_SIZE),  value (batch, 1)

    The policy head produces unnormalised logits; apply softmax externally.
    The value head produces a scalar in (-1, 1) via tanh.

    norm_type: "batch" (default) or "layer" — use "layer" on DirectML where
               BatchNorm2d is unsupported.
    """

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

        # Residual tower
        self.tower = nn.Sequential(
            *[ResBlock(num_channels, norm_type) for _ in range(num_res_blocks)]
        )

        # Policy head: 2-filter conv → flatten → FC
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_norm = _make_norm(norm_type, 2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)

        # Value head: 1-filter conv → flatten → 64 → 1
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_norm = _make_norm(norm_type, 1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_norm(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        # Value
        v = F.relu(self.value_norm(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, board_tensor: torch.Tensor):
        """
        Single-sample inference (no batch dim required).

        Args:
            board_tensor: (3, ROWS, COLS) float tensor (already on the right device)
        Returns:
            policy_probs: (ACTION_SIZE,) tensor (softmax applied)
            value:        float scalar
        """
        self.eval()
        policy_logits, value = self(board_tensor.unsqueeze(0))
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        return policy_probs, value.item()

import torch
import torch.nn as nn
import torch.nn.functional as F

from c4.game import ROWS, COLS, ACTION_SIZE


class ResBlock(nn.Module):
    """Standard pre-activation residual block (no pre-act here; matches AZ paper)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-head network for Connect Four.

    Input : (batch, 3, ROWS, COLS)  -- canonical board (see c4/game.py)
    Output: policy_logits (batch, ACTION_SIZE),  value (batch, 1)

    The policy head produces unnormalised logits; apply softmax externally.
    The value head produces a scalar in (-1, 1) via tanh.
    """

    def __init__(self, num_res_blocks: int = 5, num_channels: int = 64):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head: 2-filter conv → flatten → FC
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, ACTION_SIZE)

        # Value head: 1-filter conv → flatten → 64 → 1
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
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

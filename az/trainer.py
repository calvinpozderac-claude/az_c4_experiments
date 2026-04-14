import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from az.optimizer import DirectMLSafeAdam
from c4.game import Connect4, COLS
from az.network import AlphaZeroNet
from az.mcts import MCTS
from az.replay_buffer import ReplayBuffer
from az.endgame_buffer import EndgameBuffer
from az.off_path_buffer import OffPathBuffer

_NUM_HEADS = AlphaZeroNet.NUM_VALUE_HEADS   # 8


class SelfPlayTrainer:
    """
    Orchestrates the AlphaZero training loop:
      self-play  →  fill replay buffers  →  train network  →  repeat.

    Value architecture (8 heads):
      Head 0  game_outcome  GatedMetaHead (combines all 7 aux heads + tower)
      Head 1  z_direct      Independent ValueHead, trained on game outcome z
      Head 2  mcts_q        Q at root / node
      Head 3  minimax_net_d1
      Head 4  minimax_net_d2
      Head 5  minimax_net_d3
      Head 6  minimax_q_n10
      Head 7  proven_minimax  proven value from terminal backprop (NaN if none)

    Training data sources:
      replay_buffer    : self-play main-line positions (all 8 targets)
      endgame_buffer   : last-N-moves near-terminal positions (all 8 targets)
      off_path_buffer  : MCTS off-path nodes with N ≥ min_visits (heads 2-7 only)

    Masked MSE loss: NaN targets are excluded from the value loss, so off-path
    positions only train heads 2-7.  Policy loss is computed on all positions
    (off-path nodes have a visit-distribution policy target from their subtree).

    Only head 0 (game_outcome, GatedMetaHead output) is used in MCTS search.
    """

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device

        self.network = AlphaZeroNet(
            num_res_blocks=config.network.num_res_blocks,
            num_channels=config.network.num_channels,
            norm_type=config.network.norm_type,
        ).to(device)

        if config.training.adam_foreach:
            self.optimizer = Adam(
                self.network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = DirectMLSafeAdam(
                self.network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.training.num_iterations, 1),
            eta_min=config.training.lr_min,
        )

        self.replay_buffer   = ReplayBuffer(max_size=config.training.replay_buffer_size)
        self.endgame_buffer  = EndgameBuffer(max_size=config.training.endgame_buffer_size)
        self.off_path_buffer = OffPathBuffer(max_size=config.training.off_path_buffer_size)

        self.mcts = MCTS(
            network=self.network,
            device=device,
            c_puct=config.mcts.c_puct,
            num_simulations=config.mcts.num_simulations,
        )

        self.iteration: int = 0
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def self_play_game(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Play one game via self-play using MCTS.

        Returns a list of (canonical_board, mcts_policy, values) triples.

        values : (NUM_VALUE_HEADS,) float32
          [0] game_outcome target    = z ∈ {−1, 0, +1} (assigned at game end)
          [1] z_direct target        = same z
          [2] mcts_q                 = root.Q after search
          [3] minimax_net_d1         from compute_value_targets()
          [4] minimax_net_d2
          [5] minimax_net_d3
          [6] minimax_q_n10
          [7] proven_minimax         = proven value or NaN

        All from the position's current player's perspective.

        Off-path nodes (N ≥ min_visits) are also collected and added to
        self.off_path_buffer after each MCTS search.
        """
        game = Connect4()
        history: List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []

        while not game.game_over:
            temp = 1.0 if game.num_moves < self.config.mcts.temp_threshold else 0.0

            action_probs, _ = self.mcts.run(
                game,
                temperature=temp,
                add_dirichlet=True,
                dirichlet_alpha=self.config.mcts.dirichlet_alpha,
                dirichlet_epsilon=self.config.mcts.dirichlet_epsilon,
            )

            # Targets for aux heads 2-7 from root tree
            mcts_targets = self.mcts.compute_value_targets(
                min_visits_q=self.config.mcts.minimax_q_min_visits
            )  # (6,): [mcts_q, mm_d1, mm_d2, mm_d3, mm_q_n10, proven_minimax]

            # Off-path nodes: add to off_path_buffer (heads 0,1 = NaN)
            if self.config.training.off_path_batch_size > 0:
                off_path = self.mcts.collect_off_path_nodes(
                    min_visits=self.config.training.off_path_min_visits,
                    num_value_heads=_NUM_HEADS,
                    min_visits_q=self.config.mcts.minimax_q_min_visits,
                )
                self.off_path_buffer.add_positions(off_path)

            board = game.get_canonical_board()
            history.append((board, action_probs, game.current_player, mcts_targets))

            valid_moves = game.get_valid_moves()
            if temp == 0:
                action = max(valid_moves, key=lambda m: action_probs[m])
            else:
                probs = np.array([action_probs[m] for m in valid_moves], dtype=np.float64)
                probs /= probs.sum()
                action = int(np.random.choice(valid_moves, p=probs))

            game.make_move(action)

        # Assign z to heads 0 and 1 for every position on the game line
        outcome = game.winner
        training_data = []
        for board, policy, player, mcts_targets in history:
            if outcome == 0:
                game_value = 0.0
            elif outcome == player:
                game_value = 1.0
            else:
                game_value = -1.0

            values = np.empty(_NUM_HEADS, dtype=np.float32)
            values[0] = game_value   # game_outcome target (GatedMetaHead)
            values[1] = game_value   # z_direct target
            values[2:] = mcts_targets   # [mcts_q, mm_d1, mm_d2, mm_d3, mm_q_n10, proven]
            training_data.append((board, policy, values))

        return training_data

    def run_self_play(self, num_games: int):
        """Generate `num_games` self-play games and fill all replay buffers."""
        self.network.eval()
        for _ in tqdm(range(num_games), desc="Self-play", leave=False):
            game_data = self.self_play_game()
            self.replay_buffer.add_game(game_data)
            if self.config.training.endgame_batch_size > 0:
                self.endgame_buffer.add_game(
                    game_data, self.config.training.endgame_lookback
                )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_value_loss(
        pred: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-head MSE loss ignoring NaN targets.

        Returns
        -------
        per_head : (NUM_HEADS,) tensor — NaN for heads with no valid targets
        mean     : scalar — mean over heads with at least one valid target
        """
        mask   = ~torch.isnan(targets)                           # (B, H) bool
        sq_err = (pred - targets.nan_to_num(0.0)) ** 2          # (B, H)

        # Per-head mean (NaN if no valid samples for that head)
        head_counts = mask.float().sum(dim=0).clamp(min=1e-6)   # (H,)
        per_head    = (sq_err * mask.float()).sum(dim=0) / head_counts  # (H,)

        # Overall mean (only over heads that have any valid target in this batch)
        has_any = mask.any(dim=0)                                # (H,) bool
        mean    = per_head[has_any].mean() if has_any.any() else torch.zeros(1, device=pred.device).squeeze()

        return per_head, mean

    def train_step(self) -> Dict[str, float]:
        """
        Run `num_epochs` gradient updates on randomly sampled batches.
        Returns averaged loss statistics (empty dict if buffer not ready).
        """
        cfg = self.config.training
        if not self.replay_buffer.is_ready(cfg.batch_size):
            return {}

        self.network.train()

        totals: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss":  0.0,
            "total_loss":  0.0,
        }
        head_totals = np.zeros(_NUM_HEADS, dtype=np.float64)

        for _ in range(cfg.num_epochs):
            boards, policies, values = self.replay_buffer.sample(cfg.batch_size)

            # Supplement with endgame positions
            if cfg.endgame_batch_size > 0 and self.endgame_buffer.is_ready(cfg.endgame_batch_size):
                eg_b, eg_p, eg_v = self.endgame_buffer.sample(cfg.endgame_batch_size)
                boards   = np.concatenate([boards,   eg_b], axis=0)
                policies = np.concatenate([policies, eg_p], axis=0)
                values   = np.concatenate([values,   eg_v], axis=0)

            # Supplement with off-path positions (partial targets — NaN in heads 0,1)
            if cfg.off_path_batch_size > 0 and self.off_path_buffer.is_ready(cfg.off_path_batch_size):
                op_b, op_p, op_v = self.off_path_buffer.sample(cfg.off_path_batch_size)
                boards   = np.concatenate([boards,   op_b], axis=0)
                policies = np.concatenate([policies, op_p], axis=0)
                values   = np.concatenate([values,   op_v], axis=0)

            board_t  = torch.from_numpy(boards).to(self.device)
            policy_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).to(self.device)   # (batch, 8) with NaN

            policy_logits, value_pred = self.network(board_t)     # value_pred: (batch, 8)

            # Policy loss — computed on all positions (off-path have visit-dist targets)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            p_loss = -(policy_t * log_probs).sum(dim=-1).mean()

            # Masked value loss — skips NaN targets (off-path heads 0,1)
            head_losses, v_loss = self._masked_value_loss(value_pred, values_t)

            loss = p_loss + cfg.value_loss_weight * v_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            totals["policy_loss"] += p_loss.item()
            totals["value_loss"]  += v_loss.item()
            totals["total_loss"]  += loss.item()
            head_totals += head_losses.detach().cpu().numpy()

        n = cfg.num_epochs
        result = {k: v / n for k, v in totals.items()}
        for i, name in enumerate(AlphaZeroNet.VALUE_HEAD_NAMES):
            result[f"vloss_{name}"] = float(head_totals[i] / n)
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train_iteration(self) -> Dict[str, float]:
        """One full iteration: self-play + training step."""
        self.iteration += 1

        t0 = time.time()
        self.run_self_play(self.config.training.num_self_play_games)
        t_sp = time.time() - t0

        t0 = time.time()
        losses = self.train_step()
        t_tr = time.time() - t0

        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]

        # Short display aliases for per-head loss
        _HEAD_ABBREV = ["gam", "zdr", "mcq", "mn1", "mn2", "mn3", "mqn", "prv"]
        if losses:
            head_parts = "  ".join(
                f"{_HEAD_ABBREV[i]}={losses[f'vloss_{n}']:.4f}"
                for i, n in enumerate(AlphaZeroNet.VALUE_HEAD_NAMES)
            )
            loss_str = (
                f"policy={losses['policy_loss']:.4f}  "
                f"value={losses['value_loss']:.4f}  "
                f"[{head_parts}]"
            )
        else:
            loss_str = "buffer not ready"

        print(
            f"[Iter {self.iteration:3d}]  "
            f"self-play {t_sp:.1f}s  train {t_tr:.1f}s  "
            f"lr={current_lr:.2e}  "
            f"buf {len(self.replay_buffer):,}  "
            f"eg {len(self.endgame_buffer):,}  "
            f"op {len(self.off_path_buffer):,}  "
            f"{loss_str}"
        )
        return losses

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(
                self.config.training.checkpoint_dir,
                f"checkpoint_{self.iteration:04d}.pt",
            )
        torch.save(
            {
                "iteration":             self.iteration,
                "model_state_dict":      self.network.state_dict(),
                "optimizer_state_dict":  self.optimizer.state_dict(),
                "scheduler_state_dict":  self.scheduler.state_dict(),
                "buffer_size":           len(self.replay_buffer),
                "endgame_buffer_size":   len(self.endgame_buffer),
                "off_path_buffer_size":  len(self.off_path_buffer),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.iteration = ckpt["iteration"]
        print(f"Checkpoint loaded: {path} (iteration {self.iteration})")

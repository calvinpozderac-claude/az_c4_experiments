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


class SelfPlayTrainer:
    """
    Orchestrates the AlphaZero training loop:
      self-play  →  fill replay buffer  →  train network  →  repeat.
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
            # Standard Adam (CUDA / CPU)
            self.optimizer = Adam(
                self.network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            # DirectML: use lerp_-free Adam (aten::lerp.Scalar_out unsupported)
            self.optimizer = DirectMLSafeAdam(
                self.network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

        # Cosine annealing: LR decays from learning_rate → lr_min over num_iterations
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.training.num_iterations, 1),
            eta_min=config.training.lr_min,
        )

        self.replay_buffer = ReplayBuffer(max_size=config.training.replay_buffer_size)
        self.endgame_buffer = EndgameBuffer(max_size=config.training.endgame_buffer_size)

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

    def self_play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one game via self-play using MCTS.

        Returns a list of (canonical_board, mcts_policy, outcome) triples,
        where outcome is from the current player's perspective at that step.
        """
        game = Connect4()
        # Store (board, policy, player_who_moved) then assign outcomes at the end
        history: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not game.game_over:
            # Temperature: exploratory for the opening, greedy afterwards
            temp = 1.0 if game.num_moves < self.config.mcts.temp_threshold else 0.0

            action_probs, _ = self.mcts.run(
                game,
                temperature=temp,
                add_dirichlet=True,
                dirichlet_alpha=self.config.mcts.dirichlet_alpha,
                dirichlet_epsilon=self.config.mcts.dirichlet_epsilon,
            )

            board = game.get_canonical_board()
            history.append((board, action_probs, game.current_player))

            # Sample action from MCTS distribution
            valid_moves = game.get_valid_moves()
            if temp == 0:
                action = max(valid_moves, key=lambda m: action_probs[m])
            else:
                probs = np.array([action_probs[m] for m in valid_moves], dtype=np.float64)
                probs /= probs.sum()
                action = int(np.random.choice(valid_moves, p=probs))

            game.make_move(action)

        # Assign game outcome to each position
        outcome = game.winner  # 1, -1, or 0
        training_data = []
        for board, policy, player in history:
            if outcome == 0:
                value = 0.0
            elif outcome == player:
                value = 1.0
            else:
                value = -1.0
            training_data.append((board, policy, value))

        return training_data

    def run_self_play(self, num_games: int):
        """Generate `num_games` self-play games and store in both replay buffers."""
        self.network.eval()
        for _ in tqdm(range(num_games), desc="Self-play", leave=False):
            game_data = self.self_play_game()
            self.replay_buffer.add_game(game_data)
            # Add the last `endgame_lookback` positions (near-terminal ground truth)
            # to the deduplicated endgame buffer.
            if self.config.training.endgame_batch_size > 0:
                self.endgame_buffer.add_game(
                    game_data, self.config.training.endgame_lookback
                )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """
        Run `num_epochs` gradient updates on randomly sampled batches.
        Returns averaged loss statistics (empty dict if buffer not ready).
        """
        cfg = self.config.training
        if not self.replay_buffer.is_ready(cfg.batch_size):
            return {}

        self.network.train()

        total_p_loss = total_v_loss = total_loss = 0.0

        for _ in range(cfg.num_epochs):
            boards, policies, values = self.replay_buffer.sample(cfg.batch_size)

            # Supplement with endgame positions when the buffer is ready.
            if cfg.endgame_batch_size > 0 and self.endgame_buffer.is_ready(cfg.endgame_batch_size):
                eg_b, eg_p, eg_v = self.endgame_buffer.sample(cfg.endgame_batch_size)
                boards   = np.concatenate([boards,    eg_b], axis=0)
                policies = np.concatenate([policies,  eg_p], axis=0)
                values   = np.concatenate([values,    eg_v], axis=0)

            board_t = torch.from_numpy(boards).to(self.device)
            policy_t = torch.from_numpy(policies).to(self.device)
            value_t = torch.from_numpy(values).unsqueeze(1).to(self.device)

            policy_logits, value_pred = self.network(board_t)

            # Cross-entropy policy loss (target is the MCTS visit distribution)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            p_loss = -(policy_t * log_probs).sum(dim=-1).mean()

            # MSE value loss, upweighted so value gradients aren't drowned by policy
            v_loss = F.mse_loss(value_pred, value_t)

            loss = p_loss + cfg.value_loss_weight * v_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            total_loss += loss.item()

        n = cfg.num_epochs
        return {
            "policy_loss": total_p_loss / n,
            "value_loss": total_v_loss / n,
            "total_loss": total_loss / n,
        }

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

        # Decay LR after each iteration
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]

        loss_str = (
            f"policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}"
            if losses else "buffer not ready"
        )
        print(
            f"[Iter {self.iteration:3d}]  "
            f"self-play {t_sp:.1f}s  train {t_tr:.1f}s  "
            f"lr={current_lr:.2e}  "
            f"buf {len(self.replay_buffer):,}  eg {len(self.endgame_buffer):,}  "
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
                "iteration": self.iteration,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "buffer_size": len(self.replay_buffer),
                "endgame_buffer_size": len(self.endgame_buffer),
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

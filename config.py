from dataclasses import dataclass, field
from typing import List


@dataclass
class NetworkConfig:
    num_res_blocks: int = 3
    num_channels: int = 64
    # Normalisation layer: "batch" (BatchNorm2d) or "layer" (LayerNorm).
    # DirectML (AMD GPU on Windows) does not support BatchNorm2d — use "layer" there.
    # train.py sets this automatically when DirectML is detected.
    norm_type: str = "batch"


@dataclass
class MCTSConfig:
    num_simulations: int = 75
    c_puct: float = 1.0             # Original AlphaZero value
    dirichlet_alpha: float = 0.3    # Dirichlet noise alpha (root exploration)
    dirichlet_epsilon: float = 0.25 # Fraction of prior replaced by noise
    temp_threshold: int = 20        # Moves before switching to greedy


@dataclass
class TrainingConfig:
    batch_size: int = 256
    # SGD with momentum — optimizer used in the original AlphaZero paper
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # Equal 1:1 weighting as in original AlphaZero paper: L = (z-v)^2 - pi^T log p + c||theta||^2
    value_loss_weight: float = 1.0
    replay_buffer_size: int = 100_000
    num_self_play_games: int = 30
    num_epochs: int = 10            # Mini-batch updates per iteration
    num_iterations: int = 100
    # Step-wise LR schedule (original AlphaZero): decay by lr_decay_factor at each milestone
    lr_decay_milestones: List[float] = field(default_factory=lambda: [0.5, 0.75])
    lr_decay_factor: float = 0.1
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    eval_interval: int = 5


@dataclass
class EvalConfig:
    # Sims for MCTS-based position evaluation (slower, more accurate)
    mcts_simulations: int = 400
    data_dir: str = "data"
    # Gamesolver test files (relative to repo root)
    test_files: List[str] = field(default_factory=lambda: [
        "Test_L1_R1", "Test_L1_R2", "Test_L1_R3",
        "Test_L2_R1", "Test_L2_R2",
        "Test_L3_R1",
    ])
    # Preprocessed .npz (without extension; numpy adds it)
    preprocessed_file: str = "data/eval_positions"


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


DEFAULT_CONFIG = Config()

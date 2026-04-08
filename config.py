from dataclasses import dataclass, field
from typing import List


@dataclass
class NetworkConfig:
    num_res_blocks: int = 3   # 3 blocks: sufficient for C4 (6×7 board), 30% faster than 5
    num_channels: int = 64
    # Normalisation layer: "batch" (BatchNorm2d) or "layer" (LayerNorm).
    # DirectML (AMD GPU on Windows) does not support BatchNorm2d — use "layer" there.
    # train.py sets this automatically when DirectML is detected.
    norm_type: str = "batch"


@dataclass
class MCTSConfig:
    num_simulations: int = 75       # Simulations per move during self-play (↑ from 50 for better policy targets)
    c_puct: float = 1.5             # PUCT exploration constant
    dirichlet_alpha: float = 0.3    # Dirichlet noise alpha (root exploration)
    dirichlet_epsilon: float = 0.25 # Fraction of prior replaced by noise
    temp_threshold: int = 20        # Moves before switching to greedy (↑ from 15: more exploratory games)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.0005   # ↓ from 0.001: more stable value head convergence
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100_000
    num_self_play_games: int = 30   # ↑ from 20: less noisy value targets
    num_epochs: int = 20            # ↑ from 10: make better use of each batch
    num_iterations: int = 100
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5    # Save every N iterations
    eval_interval: int = 5          # Evaluate every N iterations


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

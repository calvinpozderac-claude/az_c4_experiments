from dataclasses import dataclass, field
from typing import List


@dataclass
class NetworkConfig:
    num_res_blocks: int = 5
    num_channels: int = 64


@dataclass
class MCTSConfig:
    num_simulations: int = 200      # Simulations per move during self-play
    c_puct: float = 1.5             # PUCT exploration constant
    dirichlet_alpha: float = 0.3    # Dirichlet noise alpha (root exploration)
    dirichlet_epsilon: float = 0.25 # Fraction of prior replaced by noise
    temp_threshold: int = 15        # Moves before switching to greedy (temp=0)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100_000
    num_self_play_games: int = 100  # Games per training iteration
    num_epochs: int = 10            # Training passes over sampled batch per iteration
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

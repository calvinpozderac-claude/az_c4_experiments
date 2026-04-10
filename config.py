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
    learning_rate: float = 0.001    # Restored; cosine scheduler decays it to lr_min
    lr_min: float = 1e-5            # Floor for cosine annealing
    weight_decay: float = 1e-4
    value_loss_weight: float = 2.0  # Upweight value head (policy logits dominate otherwise)
    replay_buffer_size: int = 100_000
    num_self_play_games: int = 30   # ↑ from 20: less noisy value targets
    num_epochs: int = 200           # ↑ from 20: primary fix — training was using <0.1% of wall time
    num_iterations: int = 100
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5    # Save every N iterations
    eval_interval: int = 5          # Evaluate every N iterations
    # Adam foreach mode uses _foreach_lerp_ which DirectML doesn't support.
    # train.py sets this to False automatically when DirectML is detected.
    adam_foreach: bool = True
    # Endgame buffer: a deduplicated rolling buffer of near-terminal positions.
    # Positions from the last `endgame_lookback` moves of each game are added
    # (unique boards only).  During training, `endgame_batch_size` extra samples
    # from this buffer supplement each main batch, giving the network a stronger
    # signal on positions closest to the ground-truth outcome.
    endgame_buffer_size: int = 10_000  # unique positions stored
    endgame_lookback: int = 10         # last N moves per game added to buffer
    endgame_batch_size: int = 64       # supplement per training step (0 = disabled)


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

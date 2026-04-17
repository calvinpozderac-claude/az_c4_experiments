#!/usr/bin/env python3
"""
AlphaZero Connect Four – baseline training script.

Usage
-----
    python train.py                        # train with defaults
    python train.py --resume checkpoints/checkpoint_0010.pt
    python train.py --iters 50 --sims 100  # quick smoke test

Evaluation against gamesolver optimal positions runs every
config.training.eval_interval iterations if the preprocessed data exists.
Run preprocess_eval.py first to generate it.
"""

import argparse
import os

from config import Config, DEFAULT_CONFIG
from device import get_device, is_directml
from az.trainer import SelfPlayTrainer
from az.mcts import MCTS
from eval.preprocess import preprocess_all, load_preprocessed
from eval.evaluate import evaluate_value_accuracy, evaluate_mcts_accuracy, print_evaluation_results


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero C4 trainer")
    p.add_argument("--iters", type=int, default=None,
                   help="Override num_iterations from config")
    p.add_argument("--sims", type=int, default=None,
                   help="Override MCTS simulations per move")
    p.add_argument("--games", type=int, default=None,
                   help="Override self-play games per iteration")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Evaluate every N iterations (overrides config; 1 = every iteration)")
    p.add_argument("--eval-only", action="store_true",
                   help="Run one evaluation pass then exit (requires --resume)")
    p.add_argument("--mcts-eval", action="store_true",
                   help="Include slow MCTS evaluation (default: value-head only)")
    return p.parse_args()


def load_eval_data(config: Config):
    """Return preprocessed eval data, or None if not available."""
    path = config.eval.preprocessed_file
    npz_path = path if path.endswith(".npz") else path + ".npz"

    if not os.path.exists(npz_path):
        found = [f for f in config.eval.test_files if os.path.exists(f)]
        if found:
            print("Preprocessing evaluation data (run preprocess_eval.py to do this once)...")
            preprocess_all(found, config.eval.data_dir, path)
        else:
            print("No evaluation data found; skipping eval during training.")
            return None

    try:
        data = load_preprocessed(path)
        print(f"Loaded {len(data['boards']):,} evaluation positions from {npz_path}")
        return data
    except Exception as exc:
        print(f"Warning: could not load eval data: {exc}")
        return None


def run_evaluation(trainer: SelfPlayTrainer, eval_data: dict, run_mcts: bool, config: Config):
    """Evaluate the current network and print results."""
    # Fast: value-head batch evaluation
    results = evaluate_value_accuracy(
        trainer.network, trainer.device, eval_data, batch_size=512
    )
    print_evaluation_results(results, header=f"Value-head eval (iter {trainer.iteration})")

    # Slow: MCTS evaluation on a subset
    if run_mcts:
        eval_mcts = MCTS(
            network=trainer.network,
            device=trainer.device,
            c_puct=config.mcts.c_puct,
            num_simulations=config.eval.mcts_simulations,
        )
        for lvl in [1, 2, 3]:
            mcts_results = evaluate_mcts_accuracy(
                eval_mcts, eval_data, max_positions=100, level=lvl
            )
            if mcts_results.get("n_evaluated", 0) > 0:
                print_evaluation_results(
                    mcts_results,
                    header=f"MCTS eval  L{lvl}  (iter {trainer.iteration})",
                )


def main():
    args = parse_args()
    config = DEFAULT_CONFIG  # Can swap to a custom Config() here

    # Apply CLI overrides
    if args.iters is not None:
        config.training.num_iterations = args.iters
    if args.sims is not None:
        config.mcts.num_simulations = args.sims
    if args.games is not None:
        config.training.num_self_play_games = args.games
    if args.eval_interval is not None:
        config.training.eval_interval = args.eval_interval

    device = get_device()
    if is_directml():
        config.network.norm_type = "layer"       # BatchNorm2d unsupported on DirectML

    eval_data = load_eval_data(config)
    trainer = SelfPlayTrainer(config, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        if eval_data is None:
            print("No eval data; cannot run eval-only mode.")
        else:
            run_evaluation(trainer, eval_data, run_mcts=args.mcts_eval, config=config)
        return

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(
        f"\nStarting AlphaZero training\n"
        f"  Device          : {device}\n"
        f"  Residual blocks : {config.network.num_res_blocks}\n"
        f"  Channels        : {config.network.num_channels}\n"
        f"  MCTS sims/move  : {config.mcts.num_simulations}\n"
        f"  Self-play games : {config.training.num_self_play_games} / iter\n"
        f"  Total iters     : {config.training.num_iterations}\n"
    )

    for _ in range(config.training.num_iterations):
        trainer.train_iteration()

        it = trainer.iteration
        if it % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint()

        if eval_data is not None and it % config.training.eval_interval == 0:
            run_evaluation(trainer, eval_data, run_mcts=args.mcts_eval, config=config)

    # Final checkpoint and evaluation
    trainer.save_checkpoint()
    if eval_data is not None:
        run_evaluation(trainer, eval_data, run_mcts=args.mcts_eval, config=config)

    print("Training complete.")


if __name__ == "__main__":
    main()

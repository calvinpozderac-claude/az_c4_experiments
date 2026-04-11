#!/usr/bin/env python3
"""
AlphaZero Connect Four – training script with multi-head evaluation tracking.

Usage
-----
    python train.py                              # train with defaults
    python train.py --resume checkpoints/checkpoint_0010.pt
    python train.py --iters 25 --sims 200 --games 20 --eval-interval 1
    python train.py --eval-only --resume checkpoints/checkpoint_0025.pt

Evaluation against gamesolver optimal positions runs every
config.training.eval_interval iterations if the preprocessed data exists.
Run preprocess_eval.py first to generate it.

When eval_data is available the script:
  - Prints per-head sign accuracy after each evaluation pass
  - Saves a summary table and PNG plots to results/ at the end
"""

import argparse
import json
import os

import numpy as np

from config import Config, DEFAULT_CONFIG
from device import get_device, is_directml
from az.trainer import SelfPlayTrainer
from az.network import AlphaZeroNet
from az.mcts import MCTS
from eval.preprocess import preprocess_all, load_preprocessed
from eval.evaluate import evaluate_value_accuracy, evaluate_mcts_accuracy, print_evaluation_results

HEAD_NAMES = AlphaZeroNet.VALUE_HEAD_NAMES   # 6 names

# Short display label for each head
HEAD_LABELS = {
    "game_outcome":   "GameOut",
    "mcts_q":         "MctsQ",
    "minimax_net_d1": "MM-Net1",
    "minimax_net_d2": "MM-Net2",
    "minimax_net_d3": "MM-Net3",
    "minimax_q_n10":  "MM-Q≥10",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero C4 trainer")
    p.add_argument("--iters",         type=int,  default=None)
    p.add_argument("--sims",          type=int,  default=None)
    p.add_argument("--games",         type=int,  default=None)
    p.add_argument("--resume",        type=str,  default=None)
    p.add_argument("--eval-interval", type=int,  default=None,
                   help="Evaluate every N iterations (1 = every iteration)")
    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--mcts-eval",     action="store_true",
                   help="Include slow MCTS evaluation")
    p.add_argument("--results-dir",   type=str,  default="results",
                   help="Directory for saving summary table and plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(config: Config):
    path     = config.eval.preprocessed_file
    npz_path = path if path.endswith(".npz") else path + ".npz"

    if not os.path.exists(npz_path):
        found = [f for f in config.eval.test_files if os.path.exists(f)]
        if found:
            print("Preprocessing evaluation data...")
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


# ---------------------------------------------------------------------------
# Per-iteration evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    trainer: SelfPlayTrainer,
    eval_data: dict,
    run_mcts: bool,
    config: Config,
) -> dict:
    """
    Evaluate and return a flat dict of metrics for this iteration.
    Keys example: "game_outcome_sign_acc", "game_outcome_sign_acc_L1", ...
    """
    results = evaluate_value_accuracy(
        trainer.network, trainer.device, eval_data, batch_size=512
    )
    print_evaluation_results(results, header=f"Value-head eval (iter {trainer.iteration})")

    if run_mcts:
        eval_mcts = MCTS(
            network=trainer.network,
            device=trainer.device,
            c_puct=config.mcts.c_puct,
            num_simulations=config.eval.mcts_simulations,
        )
        for lvl in [1, 2, 3]:
            mcts_res = evaluate_mcts_accuracy(
                eval_mcts, eval_data, max_positions=100, level=lvl
            )
            if mcts_res.get("n_evaluated", 0) > 0:
                print_evaluation_results(
                    mcts_res, header=f"MCTS eval  L{lvl}  (iter {trainer.iteration})"
                )
            for k, v in mcts_res.items():
                results[f"mcts_L{lvl}_{k}"] = v

    results["iteration"] = trainer.iteration
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(history: list):
    """Print a compact table: rows=iterations, cols=6 heads (overall sign acc)."""
    if not history:
        return

    col_w  = 9
    labels = [HEAD_LABELS[h] for h in HEAD_NAMES]
    iters  = [r["iteration"] for r in history]

    header = f"{'Iter':>5} " + " ".join(f"{l:>{col_w}}" for l in labels)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  SIGN ACCURACY vs GAMESOLVER  (overall, all levels)")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in history:
        it   = r["iteration"]
        vals = [r.get(f"{h}_sign_acc", float("nan")) for h in HEAD_NAMES]
        row  = f"{it:5d} " + " ".join(
            f"{v*100:>{col_w}.1f}%" if not np.isnan(v) else f"{'n/a':>{col_w}}"
            for v in vals
        )
        print(row)

    # Best-so-far row
    if len(history) > 1:
        print(sep)
        bests = [
            max(r.get(f"{h}_sign_acc", float("nan")) for r in history)
            for h in HEAD_NAMES
        ]
        best_row = f"{'BEST':>5} " + " ".join(
            f"{v*100:>{col_w}.1f}%" if not np.isnan(v) else f"{'n/a':>{col_w}}"
            for v in bests
        )
        print(best_row)

    print("=" * len(header))

    # Per-level table
    for lvl in [1, 2, 3]:
        key_suffix = f"_sign_acc_L{lvl}"
        first_key  = f"{HEAD_NAMES[0]}{key_suffix}"
        if first_key not in history[0]:
            continue
        print(f"\n{'Iter':>5} " + " ".join(f"{l:>{col_w}}" for l in labels) +
              f"  ← L{lvl}")
        print(sep)
        for r in history:
            it   = r["iteration"]
            vals = [r.get(f"{h}{key_suffix}", float("nan")) for h in HEAD_NAMES]
            row  = f"{it:5d} " + " ".join(
                f"{v*100:>{col_w}.1f}%" if not np.isnan(v) else f"{'n/a':>{col_w}}"
                for v in vals
            )
            print(row)

    # Rank table at final iteration
    final = history[-1]
    print(f"\n{'─'*40}")
    print(f"  FINAL ITER {final['iteration']} — RANKING by overall sign accuracy")
    print(f"{'─'*40}")
    ranked = sorted(
        [(HEAD_LABELS[h], final.get(f"{h}_sign_acc", float("nan"))) for h in HEAD_NAMES],
        key=lambda x: -x[1] if not np.isnan(x[1]) else -999,
    )
    for rank, (label, acc) in enumerate(ranked, 1):
        bar = "█" * int(acc * 30) if not np.isnan(acc) else ""
        print(f"  {rank}. {label:<10s} {acc*100:5.1f}%  {bar}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(history: list, results_dir: str):
    """Save PNG plots of sign accuracy over training iterations."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)
    iters = [r["iteration"] for r in history]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # --- Plot 1: overall sign accuracy per head ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, head in enumerate(HEAD_NAMES):
        vals = [r.get(f"{head}_sign_acc", np.nan) for r in history]
        ax.plot(iters, [v * 100 for v in vals],
                label=HEAD_LABELS[head], color=colors[i], marker="o", markersize=4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% chance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sign Accuracy (%)")
    ax.set_title("Value-Head Sign Accuracy vs Gamesolver (all levels)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(25, 100)
    ax.grid(True, alpha=0.3)
    path1 = os.path.join(results_dir, "sign_accuracy_all_heads.png")
    fig.tight_layout()
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path1}")

    # --- Plot 2: per-level grid ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, lvl in zip(axes, [1, 2, 3]):
        key_suffix = f"_sign_acc_L{lvl}"
        for i, head in enumerate(HEAD_NAMES):
            vals = [r.get(f"{head}{key_suffix}", np.nan) for r in history]
            ax.plot(iters, [v * 100 for v in vals],
                    label=HEAD_LABELS[head], color=colors[i], marker="o", markersize=3)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Level {lvl}")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(20, 100)
    axes[0].set_ylabel("Sign Accuracy (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-Level Sign Accuracy by Value Head", fontsize=12)
    fig.tight_layout()
    path2 = os.path.join(results_dir, "sign_accuracy_per_level.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path2}")

    # --- Plot 3: strong-position accuracy ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, head in enumerate(HEAD_NAMES):
        vals = [r.get(f"{head}_strong_sign_acc", np.nan) for r in history]
        ax.plot(iters, [v * 100 for v in vals],
                label=HEAD_LABELS[head], color=colors[i], marker="o", markersize=4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sign Accuracy (%)")
    ax.set_title("Strong-Position Sign Accuracy (|score| > 1) by Value Head")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(25, 100)
    ax.grid(True, alpha=0.3)
    path3 = os.path.join(results_dir, "strong_sign_accuracy.png")
    fig.tight_layout()
    fig.savefig(path3, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path3}")

    # --- Plot 4: final-iteration bar chart ---
    final = history[-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    for lvl_idx, (lvl, hatch) in enumerate([(None, ""), (1, "//"), (2, "\\\\"), (3, "xx")]):
        key = "_sign_acc" if lvl is None else f"_sign_acc_L{lvl}"
        label = "Overall" if lvl is None else f"L{lvl}"
        vals = [final.get(f"{h}{key}", np.nan) * 100 for h in HEAD_NAMES]
        x = np.arange(len(HEAD_NAMES)) + lvl_idx * 0.2
        ax.bar(x, vals, width=0.18, label=label, hatch=hatch, alpha=0.85)
    ax.set_xticks(np.arange(len(HEAD_NAMES)) + 0.3)
    ax.set_xticklabels([HEAD_LABELS[h] for h in HEAD_NAMES], fontsize=9)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Sign Accuracy (%)")
    ax.set_title(f"Final Iteration ({final['iteration']}) — Sign Accuracy by Head and Level")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    path4 = os.path.join(results_dir, "final_iteration_bar.png")
    fig.tight_layout()
    fig.savefig(path4, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = DEFAULT_CONFIG

    if args.iters         is not None: config.training.num_iterations     = args.iters
    if args.sims          is not None: config.mcts.num_simulations         = args.sims
    if args.games         is not None: config.training.num_self_play_games = args.games
    if args.eval_interval is not None: config.training.eval_interval       = args.eval_interval

    device = get_device()
    if is_directml():
        config.network.norm_type      = "layer"
        config.training.adam_foreach  = False

    eval_data = load_eval_data(config)
    trainer   = SelfPlayTrainer(config, device)

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
        f"  Value heads     : {AlphaZeroNet.NUM_VALUE_HEADS} "
        f"({', '.join(AlphaZeroNet.VALUE_HEAD_NAMES)})\n"
    )

    eval_history = []   # list of result dicts, one per evaluated iteration

    for _ in range(config.training.num_iterations):
        trainer.train_iteration()

        it = trainer.iteration
        if it % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint()

        if eval_data is not None and it % config.training.eval_interval == 0:
            result = run_evaluation(trainer, eval_data, run_mcts=args.mcts_eval, config=config)
            eval_history.append(result)

            # Save running results as JSON so they survive a crash
            os.makedirs(args.results_dir, exist_ok=True)
            with open(os.path.join(args.results_dir, "eval_history.json"), "w") as f:
                json.dump(eval_history, f, indent=2)

    # Final checkpoint
    trainer.save_checkpoint()

    # Final evaluation if not already done this iteration
    if eval_data is not None:
        last_it = trainer.iteration
        if not eval_history or eval_history[-1]["iteration"] != last_it:
            result = run_evaluation(trainer, eval_data, run_mcts=args.mcts_eval, config=config)
            eval_history.append(result)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if eval_history:
        print_summary_table(eval_history)
        save_plots(eval_history, args.results_dir)

        # Also persist the final JSON
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "eval_history.json"), "w") as f:
            json.dump(eval_history, f, indent=2)
        print(f"Results saved to {args.results_dir}/eval_history.json")

    print("Training complete.")


if __name__ == "__main__":
    main()

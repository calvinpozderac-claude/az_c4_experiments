"""
Evaluation of AlphaZero network against optimal gamesolver positions.

Two evaluation modes
--------------------
1. value_accuracy  (fast, batched)
   Runs the network's value heads on the pre-computed canonical board tensors.
   Reports sign accuracy for every value head.

2. mcts_accuracy   (slow, per-position)
   Reconstructs each game state from its move string and runs full MCTS.
   Measures sign accuracy of the MCTS root Q (head 0 after search).

Score sign convention (gamesolver)
-----------------------------------
  score > 0  →  current player wins under optimal play
  score < 0  →  current player loses
  score == 0 →  draw

AlphaZero value convention
---------------------------
  value > 0  →  current player is predicted to win
  value < 0  →  current player is predicted to lose
  value ≈ 0  →  draw predicted
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional

from c4.game import Connect4
from az.network import AlphaZeroNet
from az.mcts import MCTS
from eval.preprocess import load_preprocessed


# ---------------------------------------------------------------------------
# Fast value-head evaluation (batched) — all six heads
# ---------------------------------------------------------------------------

def evaluate_value_accuracy(
    network: AlphaZeroNet,
    device: torch.device,
    data: dict,
    max_positions: Optional[int] = None,
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Evaluate all value heads against optimal gamesolver scores.

    Ground truth: sign(gamesolver_score) ∈ {-1, 0, +1}, on the same scale
    as the tanh value head outputs.

    For each head, reports:
      sign_acc             fraction where sign(pred) == sign(score)   (higher better)
      sign_acc_L{n}        per difficulty level
      strong_sign_acc      sign_acc on positions where |score| > 1
      sign_mse             MSE(pred, sign(score))                     (lower better)
      sign_mse_L{n}        per difficulty level
      strong_sign_mse      sign_mse on positions where |score| > 1

    MSE baselines (for reference):
      always predict  0 : MSE ≈ 0.86  (fraction of non-draw positions)
      always predict +1 : MSE ≈ 1.49  (penalises losses heavily)
      perfect predictor : MSE = 0.00

    Keys are prefixed with the head name, e.g. "game_outcome_sign_mse".
    """
    boards = data["boards"]
    scores = data["scores"]
    levels = data["levels"]

    if max_positions is not None:
        boards = boards[:max_positions]
        scores = scores[:max_positions]
        levels = levels[:max_positions]

    n = len(boards)
    num_heads = network.NUM_VALUE_HEADS
    pred_values = np.zeros((n, num_heads), dtype=np.float32)

    network.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Value eval", leave=False):
            end   = min(start + batch_size, n)
            batch = torch.from_numpy(boards[start:end]).to(device)
            _, values = network(batch)                      # (batch, 6)
            pred_values[start:end] = values.cpu().numpy()

    true_sign   = np.sign(scores).astype(np.float32)   # {-1, 0, +1} ground truth
    strong_mask = np.abs(scores) > 1
    unique_levels = sorted(set(levels.tolist()))

    results: Dict[str, float] = {}
    for h, head_name in enumerate(network.VALUE_HEAD_NAMES):
        pred       = pred_values[:, h]                     # tanh output in (-1, 1)
        pred_sign  = np.sign(pred)
        correct    = pred_sign == true_sign
        sq_err     = (pred - true_sign) ** 2              # element-wise squared error

        prefix = head_name
        # Sign accuracy
        results[f"{prefix}_sign_acc"]        = float(correct.mean())
        results[f"{prefix}_strong_sign_acc"] = (
            float(correct[strong_mask].mean()) if strong_mask.sum() > 0 else float("nan")
        )
        # Sign MSE
        results[f"{prefix}_sign_mse"]        = float(sq_err.mean())
        results[f"{prefix}_strong_sign_mse"] = (
            float(sq_err[strong_mask].mean()) if strong_mask.sum() > 0 else float("nan")
        )
        for lvl in unique_levels:
            mask = levels == lvl
            if mask.sum() > 0:
                results[f"{prefix}_sign_acc_L{lvl}"] = float(correct[mask].mean())
                results[f"{prefix}_sign_mse_L{lvl}"] = float(sq_err[mask].mean())

    return results


# ---------------------------------------------------------------------------
# Slow MCTS-based evaluation (per-position)
# ---------------------------------------------------------------------------

def evaluate_mcts_accuracy(
    mcts: MCTS,
    data: dict,
    max_positions: int = 200,
    level: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate full MCTS value estimates (root Q) against optimal scores.

    Reconstructs game states from move strings so MCTS can search from them.
    Measures sign accuracy of the MCTS root Q, which corresponds to head 0
    (game_outcome) trained via many simulations.

    NOTE: creates a fresh MCTS context for each position (no tree reuse).
    """
    scores       = data["scores"]
    levels       = data["levels"]
    move_strings = data["move_strings"]

    if level is not None:
        mask         = levels == level
        scores       = scores[mask]
        move_strings = move_strings[mask]

    n = len(scores)
    if n > max_positions:
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(n, max_positions, replace=False)
        scores       = scores[idx]
        move_strings = move_strings[idx]

    pred_values = np.zeros(len(scores), dtype=np.float32)

    for i, move_str in enumerate(tqdm(move_strings, desc="MCTS eval", leave=False)):
        game = Connect4.from_move_string(str(move_str))
        # run() returns root Q — the head-0 backed-up estimate after full search
        _, value = mcts.run(game, temperature=0, add_dirichlet=False)
        pred_values[i] = value

    pred_sign = np.sign(pred_values)
    true_sign = np.sign(scores)
    correct   = pred_sign == true_sign

    results: Dict[str, float] = {
        "mcts_sign_accuracy": float(correct.mean()),
        "n_evaluated":        len(scores),
    }
    strong_mask = np.abs(scores) > 1
    if strong_mask.sum() > 0:
        results["mcts_strong_sign_accuracy"] = float(correct[strong_mask].mean())

    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_evaluation_results(results: Dict[str, float], header: str = "Evaluation"):
    width = 44
    print(f"\n{'=' * width}")
    print(f"  {header}")
    print(f"{'=' * width}")
    for key in sorted(results.keys()):
        val = results[key]
        if isinstance(val, float):
            if np.isnan(val):
                print(f"  {key:<39s}  {'n/a':>7s}")
            else:
                print(f"  {key:<39s}  {val:.4f}  ({val * 100:.1f}%)")
        else:
            print(f"  {key:<39s}  {val}")
    print(f"{'=' * width}\n")

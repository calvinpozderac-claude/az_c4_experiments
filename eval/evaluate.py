"""
Evaluation of AlphaZero network against optimal gamesolver positions.

Two evaluation modes
--------------------
1. value_accuracy  (fast, batched)
   Runs the network's value head on the pre-computed canonical board tensors.
   O(N / batch_size) network forward passes.

2. mcts_accuracy   (slow, per-position)
   Reconstructs each game state from its move string and runs full MCTS.
   More accurate but ~100x slower; use on a small subset.

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

Primary benchmark metric: sign_mse
-----------------------------------
  MSE between the network's tanh value output and sign(optimal_score) ∈ {-1, 0, +1}.

  sign_accuracy is insufficient when draws are possible: a prediction of 0.9
  for a draw position (score == 0) scores 100% on sign accuracy but is badly
  wrong. sign_mse correctly penalises this as (0.9 - 0)^2 = 0.81.

  MSE baselines:
    always predict  0 : MSE ≈ fraction of non-draw positions (~0.86 on L1-L3 mix)
    always predict +1 : MSE ≈ 1.49
    perfect predictor : MSE = 0.00
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional

from c4.game import Connect4
from az.network import AlphaZeroNet
from az.mcts import MCTS
from eval.preprocess import load_preprocessed


# ---------------------------------------------------------------------------
# Fast value-head evaluation (batched)
# ---------------------------------------------------------------------------

def evaluate_value_accuracy(
    network: AlphaZeroNet,
    device: torch.device,
    data: dict,
    max_positions: Optional[int] = None,
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Evaluate network value head against optimal gamesolver scores.

    Metric: sign_mse = mean((pred - sign(score))^2)
      Measures how close the tanh prediction is to the {-1, 0, +1} target.
      Lower is better. Properly handles draws (sign == 0).

    Per-level breakdowns (L1/L2/L3) and a strong_* variant that excludes
    near-draw positions (|score| <= 1) are also reported.
    """
    boards = data["boards"]
    scores = data["scores"]
    levels = data["levels"]

    if max_positions is not None:
        boards = boards[:max_positions]
        scores = scores[:max_positions]
        levels = levels[:max_positions]

    n = len(boards)
    pred_values = np.zeros(n, dtype=np.float32)

    network.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Value eval", leave=False):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(boards[start:end]).to(device)
            _, values = network(batch)
            pred_values[start:end] = values.squeeze(-1).cpu().numpy()

    true_sign = np.sign(scores).astype(np.float32)   # {-1, 0, +1}
    sq_err = (pred_values - true_sign) ** 2

    strong_mask = np.abs(scores) > 1
    unique_levels = sorted(set(levels.tolist()))

    results: Dict[str, float] = {}
    results["sign_mse"] = float(sq_err.mean())
    if strong_mask.sum() > 0:
        results["strong_sign_mse"] = float(sq_err[strong_mask].mean())
    for lvl in unique_levels:
        mask = levels == lvl
        if mask.sum() > 0:
            results[f"sign_mse_L{lvl}"] = float(sq_err[mask].mean())

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
    Evaluate full MCTS value estimates against optimal scores.

    Reconstructs game states from move strings so MCTS can search from them.
    Reports both sign_mse and sign_accuracy for the MCTS root Q value.
    """
    scores = data["scores"]
    levels = data["levels"]
    move_strings = data["move_strings"]

    if level is not None:
        mask = levels == level
        scores = scores[mask]
        move_strings = move_strings[mask]

    n = len(scores)
    if n > max_positions:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_positions, replace=False)
        scores = scores[idx]
        move_strings = move_strings[idx]

    pred_values = np.zeros(len(scores), dtype=np.float32)

    for i, move_str in enumerate(tqdm(move_strings, desc="MCTS eval", leave=False)):
        game = Connect4.from_move_string(str(move_str))
        _, value = mcts.run(game, temperature=0, add_dirichlet=False)
        pred_values[i] = value

    true_sign = np.sign(scores).astype(np.float32)
    sq_err = (pred_values - true_sign) ** 2

    results: Dict[str, float] = {
        "mcts_sign_mse": float(sq_err.mean()),
        "n_evaluated": len(scores),
    }
    strong_mask = np.abs(scores) > 1
    if strong_mask.sum() > 0:
        results["mcts_strong_sign_mse"] = float(sq_err[strong_mask].mean())

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
            print(f"  {key:<39s}  {val:.4f}")
        else:
            print(f"  {key:<39s}  {val}")
    print(f"{'=' * width}\n")

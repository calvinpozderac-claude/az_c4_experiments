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
    Evaluate network value head against optimal scores.

    Metrics
    -------
    sign_accuracy        overall fraction where sign(pred) == sign(score)
    sign_accuracy_L{n}   per difficulty level
    strong_sign_accuracy fraction on positions where |score| > 1 (no borderline draws)
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

    pred_sign = np.sign(pred_values)
    true_sign = np.sign(scores)
    correct = pred_sign == true_sign

    results: Dict[str, float] = {}
    results["sign_accuracy"] = float(correct.mean())

    for lvl in sorted(set(levels.tolist())):
        mask = levels == lvl
        if mask.sum() > 0:
            results[f"sign_accuracy_L{lvl}"] = float(correct[mask].mean())

    strong_mask = np.abs(scores) > 1
    if strong_mask.sum() > 0:
        results["strong_sign_accuracy"] = float(correct[strong_mask].mean())

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

    Parameters
    ----------
    mcts          : configured MCTS instance (wraps a trained network)
    data          : dict from load_preprocessed()
    max_positions : cap on how many positions to evaluate (chosen randomly)
    level         : if set, evaluate only positions from this difficulty level
    seed          : random seed for position sampling
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

    pred_sign = np.sign(pred_values)
    true_sign = np.sign(scores)
    correct = pred_sign == true_sign

    results: Dict[str, float] = {
        "mcts_sign_accuracy": float(correct.mean()),
        "n_evaluated": len(scores),
    }
    strong_mask = np.abs(scores) > 1
    if strong_mask.sum() > 0:
        results["mcts_strong_sign_accuracy"] = float(correct[strong_mask].mean())

    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_evaluation_results(results: Dict[str, float], header: str = "Evaluation"):
    width = 40
    print(f"\n{'=' * width}")
    print(f"  {header}")
    print(f"{'=' * width}")
    for key in sorted(results.keys()):
        val = results[key]
        if isinstance(val, float):
            print(f"  {key:<35s}  {val:.4f}  ({val * 100:.1f}%)")
        else:
            print(f"  {key:<35s}  {val}")
    print(f"{'=' * width}\n")

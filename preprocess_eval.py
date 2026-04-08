#!/usr/bin/env python3
"""
Preprocess gamesolver.org evaluation position files into a fast .npz cache.

Run once from the repository root:
    python preprocess_eval.py

The result is saved to data/eval_positions.npz and loaded by train.py
during training to evaluate progress against optimal play.
"""

import os
from config import DEFAULT_CONFIG
from eval.preprocess import preprocess_all, load_preprocessed


def main():
    cfg = DEFAULT_CONFIG.eval

    # Collect files that actually exist in the repo root
    found = [f for f in cfg.test_files if os.path.exists(f)]
    missing = [f for f in cfg.test_files if f not in found]

    if missing:
        print(f"Note: these test files were not found and will be skipped: {missing}")
    if not found:
        print("No test files found.  Make sure Test_L*_R* files are in the repo root.")
        return

    out = cfg.preprocessed_file  # without .npz
    print(f"Preprocessing {len(found)} file(s) → {out}.npz\n")
    preprocess_all(found, cfg.data_dir, out)

    # Quick sanity check
    data = load_preprocessed(out)
    n = len(data["boards"])
    print(f"\nLoaded {n:,} positions from {out}.npz  ✓")
    for lvl in sorted(set(data["levels"].tolist())):
        mask = data["levels"] == lvl
        print(f"  Level {lvl}: {mask.sum():,} positions")


if __name__ == "__main__":
    main()

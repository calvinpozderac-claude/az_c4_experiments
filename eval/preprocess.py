"""
Preprocess gamesolver.org evaluation position files into a fast-loading .npz.

File format (one position per line):
    <move_string> <score>

move_string: sequence of 1-indexed column digits, e.g. "32164625" = 8 moves
score:       optimal negamax score (positive = current player wins,
             negative = current player loses, 0 = draw)

The magnitude encodes how many moves remain until the game ends under optimal play:
    score = +(43 - total_moves) / 2   if current player wins
    score = -(43 - total_moves) / 2   if current player loses
"""

import os
import numpy as np
from typing import List, Tuple

from c4.game import Connect4, ROWS, COLS


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_position_file(filepath: str) -> List[Tuple[str, int]]:
    """
    Parse one gamesolver test file.

    Returns a list of (move_string, optimal_score) pairs.
    """
    positions: List[Tuple[str, int]] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            positions.append((parts[0], int(parts[1])))
    return positions


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_all(test_files: List[str], data_dir: str, output_path: str):
    """
    Convert all test files into a single compressed .npz archive.

    The function accepts output_path *without* the .npz extension (numpy adds it).

    Saved arrays
    ------------
    boards        : (N, 3, ROWS, COLS) float32   canonical board tensors
    scores        : (N,) int32                   optimal negamax scores
    move_lengths  : (N,) int16                   number of moves played
    levels        : (N,) uint8                   difficulty level (1/2/3)
    move_strings  : (N,) object (str)            original move strings for MCTS eval
    file_names    : (N,) object (str)            source filename
    """
    os.makedirs(data_dir, exist_ok=True)

    all_boards: List[np.ndarray] = []
    all_scores: List[int] = []
    all_move_lengths: List[int] = []
    all_levels: List[int] = []
    all_move_strings: List[str] = []
    all_file_names: List[str] = []

    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"  [skip] {filepath} not found")
            continue

        basename = os.path.basename(filepath)
        # Filename format: Test_L<n>_R<m>  →  level = n
        try:
            level = int(basename.split("_")[1][1])
        except (IndexError, ValueError):
            level = 0

        positions = parse_position_file(filepath)
        print(f"  Processing {basename}: {len(positions)} positions (level {level})")

        skipped = 0
        for move_str, score in positions:
            try:
                game = Connect4.from_move_string(move_str)
            except (ValueError, AssertionError) as exc:
                print(f"    Warning: skipping '{move_str}': {exc}")
                skipped += 1
                continue

            all_boards.append(game.get_canonical_board())
            all_scores.append(score)
            all_move_lengths.append(game.num_moves)
            all_levels.append(level)
            all_move_strings.append(move_str)
            all_file_names.append(basename)

        if skipped:
            print(f"    Skipped {skipped} invalid positions")

    if not all_boards:
        print("No positions were processed – check that test files exist.")
        return

    boards_arr = np.array(all_boards, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.int32)
    move_lengths_arr = np.array(all_move_lengths, dtype=np.int16)
    levels_arr = np.array(all_levels, dtype=np.uint8)
    move_strings_arr = np.array(all_move_strings, dtype=object)
    file_names_arr = np.array(all_file_names, dtype=object)

    np.savez_compressed(
        output_path,
        boards=boards_arr,
        scores=scores_arr,
        move_lengths=move_lengths_arr,
        levels=levels_arr,
        move_strings=move_strings_arr,
        file_names=file_names_arr,
    )

    saved_path = output_path + ".npz"
    size_kb = os.path.getsize(saved_path) / 1024
    print(
        f"\nSaved {len(all_boards):,} positions → {saved_path}  ({size_kb:.0f} KB)\n"
        f"  Board array shape: {boards_arr.shape}\n"
        f"  Score range: [{scores_arr.min()}, {scores_arr.max()}]\n"
        f"  Move length range: [{move_lengths_arr.min()}, {move_lengths_arr.max()}]"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_preprocessed(path: str) -> dict:
    """
    Load preprocessed evaluation data.

    Accepts path with or without the .npz extension.
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    data = np.load(path, allow_pickle=True)
    return {
        "boards": data["boards"],
        "scores": data["scores"],
        "move_lengths": data["move_lengths"],
        "levels": data["levels"],
        "move_strings": data["move_strings"],
        "file_names": data["file_names"],
    }

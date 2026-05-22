"""Tests for the FEN orientation / 180-degree rotation layer."""

import chess
import pytest

from chess_vision.orientation import (
    apply_orientation,
    rotate_fen_180,
    rotate_grid_180,
    rotate_placement_180,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_rotate_180_is_involution():
    """Rotating any position twice returns the original placement."""
    for fen in [
        START,
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w - - 0 1",
        "8/8/8/3k4/8/4P3/3K4/8 w - - 0 1",
    ]:
        assert rotate_fen_180(rotate_fen_180(fen)) == fen


def test_rotate_single_pawn():
    """A white pawn on e5 rotates 180 degrees to d4 (color unchanged)."""
    assert rotate_placement_180("8/8/8/4P3/8/8/8/8") == "8/8/8/8/3P4/8/8/8"


def test_rotate_starting_position_swaps_king_and_queen_files():
    """The starting position is not symmetric under 180-degree rotation:
    the king and queen swap files."""
    rotated = rotate_placement_180("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert rotated == "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr"


def test_rotate_keeps_non_placement_fields():
    """Side-to-move / castling / clock fields are left untouched."""
    rotated = rotate_fen_180(START)
    assert rotated.split()[1:] == START.split()[1:]


def test_rotated_position_is_still_a_valid_board():
    rotated = rotate_fen_180(START)
    board = chess.Board(rotated)
    assert board.board_fen() == rotated.split()[0]


def test_apply_orientation_white_is_identity():
    assert apply_orientation(START, "white") == START


def test_apply_orientation_black_rotates():
    assert apply_orientation(START, "black") == rotate_fen_180(START)


def test_apply_orientation_rejects_bad_value():
    with pytest.raises(ValueError):
        apply_orientation(START, "sideways")


def test_rotate_grid_180():
    grid = [[f"{r}{c}" for c in range(8)] for r in range(8)]
    rotated = rotate_grid_180(grid)
    assert rotated[0][0] == "77"
    assert rotated[7][7] == "00"
    assert rotate_grid_180(rotated) == grid

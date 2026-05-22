"""Board orientation handling.

A top-down photo can be taken with either side closer to the camera. The
vision pipeline always reads the warped image top-to-bottom as rank 8 .. rank 1
(standard FEN order). If the photo was taken with Black's side closer to the
camera, the read-out board is rotated 180 degrees from the standard view, so we
rotate the parsed placement back.
"""

from __future__ import annotations

import chess


def rotate_placement_180(placement: str) -> str:
    """Rotate a FEN piece-placement field 180 degrees.

    180-degree rotation == vertical flip + horizontal flip. Piece colors are
    unchanged (this is a geometric rotation of the squares only).
    """
    board = chess.BaseBoard(placement)
    board.apply_transform(chess.flip_vertical)
    board.apply_transform(chess.flip_horizontal)
    return board.board_fen()


def rotate_fen_180(fen: str) -> str:
    """Rotate the placement field of a full FEN 180 degrees, keeping the
    side-to-move / castling / en-passant / clock fields untouched."""
    parts = fen.split()
    parts[0] = rotate_placement_180(parts[0])
    return " ".join(parts)


def rotate_grid_180(grid: list[list]) -> list[list]:
    """Rotate an 8x8 grid (list of 8 rows) 180 degrees."""
    return [row[::-1] for row in grid[::-1]]


def apply_orientation(fen: str, orientation: str) -> str:
    """Apply the camera orientation to a parsed FEN.

    orientation == "white": White's side was closer to the camera (default,
        no change). orientation == "black": Black's side was closer, so rotate.
    """
    if orientation == "black":
        return rotate_fen_180(fen)
    if orientation == "white":
        return fen
    raise ValueError(f"orientation must be 'white' or 'black', got {orientation!r}")

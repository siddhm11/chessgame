"""Position checks shared by the web server and the in-browser app.

Pure python-chess: no web framework, so it also runs under Pyodide.
"""

from __future__ import annotations

import chess

from chess_vision.sanity import material_problems

# Per-square confidence thresholds for tinting the board.
CONF_CHECK = 0.55  # below this: red tint ("definitely check this square")
CONF_MAYBE = 0.80  # below this: amber tint ("uncertain")

TINT_CHECK = "#e0413188"
TINT_MAYBE = "#f0a93188"


def fen_problem(fen: str | None) -> str | None:
    """Return a human-readable problem string if the FEN is not a legal
    position, or None if it is legal. Uses chess.Board's own validation."""
    if not fen:
        return "No position yet -- analyze a photo first."
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return f"The FEN is malformed: {exc}"

    status = board.status()
    impossible = material_problems(fen.split()[0])
    if status == chess.STATUS_VALID and not impossible:
        return None

    checks = [
        (chess.STATUS_EMPTY, "the board is empty"),
        (chess.STATUS_NO_WHITE_KING, "there is no white king"),
        (chess.STATUS_NO_BLACK_KING, "there is no black king"),
        (chess.STATUS_TOO_MANY_KINGS, "there is more than one king of a color"),
        (chess.STATUS_PAWNS_ON_BACKRANK, "a pawn is on the 1st or 8th rank"),
        (chess.STATUS_TOO_MANY_WHITE_PIECES, "too many white pieces"),
        (chess.STATUS_TOO_MANY_BLACK_PIECES, "too many black pieces"),
        (chess.STATUS_TOO_MANY_WHITE_PAWNS, "too many white pawns"),
        (chess.STATUS_TOO_MANY_BLACK_PAWNS, "too many black pawns"),
        (chess.STATUS_OPPOSITE_CHECK, "the side not to move is in check"),
    ]
    found = [msg for flag, msg in checks if status & flag] + impossible
    if not found:
        found = ["the position is illegal"]
    return "Invalid position: " + "; ".join(found) + "."


def confidence_fill(psc: dict | None) -> dict:
    """Map low-confidence squares to a tint color for the board SVG."""
    fill: dict = {}
    for name, conf in (psc or {}).items():
        try:
            square = chess.parse_square(name)
        except ValueError:
            continue
        if conf < CONF_CHECK:
            fill[square] = TINT_CHECK
        elif conf < CONF_MAYBE:
            fill[square] = TINT_MAYBE
    return fill

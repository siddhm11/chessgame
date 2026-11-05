"""On a piece set the model wasn't trained on, the read can be very wrong --
but it must never LOOK right.

Measured on test_photos/chess_starting_position.jpg (a wooden Staunton set,
not the training set): the model read 37 of 64 squares wrong, many with high
confidence (silent). Now the sanity layer's material rule ("12 white queens
and 0 pawns: no game can reach that") tints every square, and the board is
still shown (not replaced by a 'no board' fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_vision import vision
from chess_vision.vision import BackendOutput

ROOT = Path(__file__).resolve().parent.parent
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _grid(placement):
    g = []
    for row in placement.split("/"):
        r = []
        for ch in row:
            r.extend(["."] * int(ch) if ch.isdigit() else [ch])
        g.append(r)
    return g


def test_unfamiliar_piece_set_is_flagged_not_silently_wrong():
    try:
        from chess_vision.model_backend import ModelBackend
        backend = ModelBackend()
        backend._load()
    except (ImportError, FileNotFoundError):
        pytest.skip("model backend unavailable")
    res = vision.analyze_image(str(ROOT / "test_photos" / "chess_starting_position.jpg"),
                               backend=backend)
    assert not res.detection_failed and res.board_corners, "board was found: show the read"
    truth, pred = _grid(START), _grid(res.fen.split()[0])
    files = "abcdefgh"
    silent = [f"{files[c]}{8 - r}" for r in range(8) for c in range(8)
              if pred[r][c] != truth[r][c] and res.per_square_confidence[f"{files[c]}{8 - r}"] >= 0.80]
    assert silent == [], f"wrong squares shown as trustworthy: {silent}"
    assert any("no game can reach" in n for n in res.notes)


class _LowConfBackend:
    """Board found, every square low confidence."""
    name = "stub"

    def __init__(self, gate):
        self.overall_confidence_gate = gate

    def analyze(self, img, corners=None):
        grid = [["."] * 8 for _ in range(8)]
        grid[0][4], grid[7][4] = "k", "K"
        return BackendOutput(grid=grid, conf=[[0.2] * 8 for _ in range(8)],
                             board_found=True, corners=[[0, 0], [1, 0], [1, 1], [0, 1]])


def test_low_average_confidence_gate_is_per_backend():
    import numpy as np
    img = np.full((200, 200, 3), 128, np.uint8)
    assert vision.analyze_image(img, backend=_LowConfBackend(gate=True)).detection_failed
    res = vision.analyze_image(img, backend=_LowConfBackend(gate=False))
    assert not res.detection_failed and res.fen.startswith("4k3/")

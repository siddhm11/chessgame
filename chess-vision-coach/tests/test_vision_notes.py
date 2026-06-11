"""Sanity-layer notes must survive the trip from backend to VisionResult,
with square names remapped when the caller asks for black orientation."""

from __future__ import annotations

import numpy as np

from chess_vision import vision
from chess_vision.vision import BackendOutput, VisionResult


class _StubBackend:
    """Backend returning a fixed plausible grid plus repair notes."""

    name = "stub"

    def __init__(self, notes):
        self._notes = notes

    def analyze(self, image_bgr) -> BackendOutput:
        grid = [["." for _ in range(8)] for _ in range(8)]
        grid[0][4] = "k"
        grid[7][4] = "K"
        conf = [[0.95 for _ in range(8)] for _ in range(8)]
        return BackendOutput(grid=grid, conf=conf, board_found=True, notes=self._notes)


def _img():
    return np.full((640, 640, 3), 128, dtype=np.uint8)


def test_rotate_note_square():
    assert vision._rotate_note_square("e8: reread Q as K") == "d1: reread Q as K"
    assert vision._rotate_note_square("a1: flagged for review") == "h8: flagged for review"
    # Notes without a leading square pass through untouched.
    assert vision._rotate_note_square("no square here") == "no square here"


def test_notes_flow_through_analyze_image():
    notes = ["e8: no k found anywhere; reread q as k"]
    result = vision.analyze_image(
        _img(), orientation="white", side_to_move="w",
        backend=_StubBackend(notes),
    )
    assert isinstance(result, VisionResult)
    assert result.notes == notes


def test_notes_squares_rotated_for_black_orientation():
    result = vision.analyze_image(
        _img(), orientation="black", side_to_move="w",
        backend=_StubBackend(["e8: reread Q as K"]),
    )
    assert result.notes == ["d1: reread Q as K"]


def test_no_notes_means_empty_list():
    result = vision.analyze_image(
        _img(), orientation="white", side_to_move="w",
        backend=_StubBackend([]),
    )
    assert result.notes == []

"""Tests for the vision pipeline's failure / fallback path.

When board detection fails or overall confidence is too low, the pipeline
must return the standard starting position with detection_failed=True so the
web app can prompt the user to set the board up manually.
"""

import numpy as np

from chess_vision import vision


def test_fallback_result_returns_starting_position():
    result = vision.fallback_result()
    assert result.detection_failed is True
    assert result.fen.split()[0] == vision.STARTING_PLACEMENT
    assert result.confidence == 0.0
    assert result.per_square_confidence is None


def test_fallback_result_respects_side_to_move():
    assert vision.fallback_result("b").fen.split()[1] == "b"
    assert vision.fallback_result("w").fen.split()[1] == "w"


def test_noise_image_falls_back():
    """An image with no board structure -> low confidence -> fallback."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    result = vision.analyze_image(noise)
    assert result.detection_failed is True
    assert result.fen.split()[0] == vision.STARTING_PLACEMENT


def test_tiny_image_falls_back():
    """An image too small to hold a board -> fallback."""
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    result = vision.analyze_image(tiny)
    assert result.detection_failed is True
    assert result.fen.split()[0] == vision.STARTING_PLACEMENT


def test_missing_file_falls_back():
    """A bad path raises inside the pipeline but is caught -> fallback."""
    result = vision.analyze_image("does/not/exist.jpg")
    assert result.detection_failed is True
    assert result.fen.split()[0] == vision.STARTING_PLACEMENT


def test_fallback_preserves_side_to_move_through_analyze():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    result = vision.analyze_image(noise, side_to_move="b")
    assert result.detection_failed is True
    assert result.fen.split()[1] == "b"

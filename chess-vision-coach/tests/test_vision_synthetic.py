"""Round-trip tests against the synthetic fixtures.

NOTE: these fixtures are rendered by python-chess itself, so this test
exercises board segmentation, piece template-matching and FEN assembly on
CLEAN top-down boards only. It is NOT a test of real-world vision (camera
angle, lighting, real piece sets) -- that requires real photos.
"""

from pathlib import Path

import pytest

from chess_vision import vision
from chess_vision.orientation import rotate_placement_180

FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED = {
    "starting": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "midgame": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
    "endgame": "8/8/8/3k4/8/4P3/3K4/8",
}


@pytest.mark.parametrize("name,placement", list(EXPECTED.items()))
def test_synthetic_fixture_round_trips(name, placement):
    result = vision.analyze_image(FIXTURES / f"{name}.png")
    assert result.detection_failed is False
    assert result.fen.split()[0] == placement
    assert result.confidence > 0.8


@pytest.mark.parametrize("name,placement", list(EXPECTED.items()))
def test_per_square_confidence_is_complete(name, placement):
    result = vision.analyze_image(FIXTURES / f"{name}.png")
    assert result.per_square_confidence is not None
    assert len(result.per_square_confidence) == 64
    for f in "abcdefgh":
        for r in range(1, 9):
            assert f"{f}{r}" in result.per_square_confidence


def test_black_orientation_rotates_parsed_board():
    """Parsing with orientation='black' yields the 180-degree rotation of
    the orientation='white' parse."""
    white = vision.analyze_image(FIXTURES / "midgame.png", orientation="white")
    black = vision.analyze_image(FIXTURES / "midgame.png", orientation="black")
    assert black.fen.split()[0] == rotate_placement_180(white.fen.split()[0])


def test_side_to_move_written_into_fen():
    result = vision.analyze_image(FIXTURES / "starting.png", side_to_move="b")
    assert result.fen.split()[1] == "b"

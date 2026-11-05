"""A photo with no chessboard must say so -- not invent a position.

Threshold evidence (bench/board_presence_eval.py): 508 real board photos
scored a grid fit of 0.732-1.01; 96 ordinary non-chess photos 0.48-0.63;
NO_BOARD_FIT = 0.68 sits in the gap. Fixtures in not_a_board/ are CC0
photos from scikit-image (Chelsea the cat, Stefan van der Walt; coffee
cup, Rachel Michetti).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from chess_vision import vision

ROOT = Path(__file__).resolve().parent
NOT_BOARDS = sorted((ROOT / "fixtures" / "not_a_board").glob("*.jpg"))
REAL_BOARDS = sorted((ROOT / "fixtures" / "real").glob("*.jpg")) + sorted(
    (ROOT.parent / "test_photos").glob("*.jpg"))


def _backends():
    yield vision.ClassicalBackend()
    try:
        from chess_vision.model_backend import ModelBackend
        mb = ModelBackend()
        if mb.model_path.exists():
            yield mb
    except ImportError:
        pass


@pytest.mark.parametrize("photo", NOT_BOARDS, ids=lambda p: p.name)
def test_non_chess_photo_is_rejected(photo):
    for backend in _backends():
        res = vision.analyze_image(str(photo), backend=backend)
        assert res.detection_failed, f"{type(backend).__name__} invented a position for {photo.name}"
        assert res.board_corners is None


@pytest.mark.parametrize("photo", REAL_BOARDS, ids=lambda p: p.name)
def test_real_boards_are_not_rejected(photo):
    """Includes chess_midgame_position.jpg, the lowest-fit real board
    measured (0.732) -- a different board style from the training photos."""
    finder = vision.ClassicalBackend()
    fit = finder.find_board_scored(vision._load_bgr(str(photo)))[1]
    assert fit >= vision.NO_BOARD_FIT + 0.03, f"{photo.name}: fit {fit:.3f} too close to the cut-off"


def test_user_corners_bypass_the_no_board_gate():
    """If the user marked the board, the no-board gate must not refuse it --
    they know better. (Whether pieces can then be read is a separate matter.)"""
    img = vision._load_bgr(str(NOT_BOARDS[0]))
    corners = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    for backend in _backends():
        assert not backend.analyze(img).board_found          # gate fires...
        out = backend.analyze(img, corners=corners)          # ...unless the user marked it
        assert out.board_found, type(backend).__name__
        assert out.corners is not None


def test_page_explains_and_offers_corner_dragging():
    client = TestClient(app)
    photo = NOT_BOARDS[0]
    with open(photo, "rb") as fh:
        r = client.post("/analyze", files={"image": (photo.name, fh, "image/jpeg")},
                        data={"orientation": "white", "side_to_move": "white"})
    assert r.status_code == 200
    assert 'data-failed="1"' in r.text
    assert "find a chessboard in this photo" in r.text
    assert 'data-corners=""' in r.text            # UI starts from its default outline
    assert "Re-read board" in r.text

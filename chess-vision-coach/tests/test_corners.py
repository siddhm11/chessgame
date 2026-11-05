"""Board corners: the app reports where it found the board, and the user can
drag 4 corners to tell it where the board really is."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, parse_corners
from chess_vision import vision

FIXTURES = Path(__file__).parent / "fixtures"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _photo_with_board(margin=60, canvas=(1100, 1500), at=(170, 400)):
    """Rendered board on a light mat with a label strip, on a darker floor.
    Returns (image, true board corners in [0,1] TL,TR,BR,BL)."""
    board = cv2.imread(str(FIXTURES / "starting.png"))
    h, w = canvas
    img = np.full((h, w, 3), (90, 95, 100), np.uint8)
    side = board.shape[0]
    y, x = at
    mat = np.full((side + 2 * margin, side + 2 * margin, 3), (225, 230, 230), np.uint8)
    for i, ch in enumerate("abcdefgh"):
        cv2.putText(mat, ch, (margin + i * side // 8 + side // 16, margin // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    mat[margin:margin + side, margin:margin + side] = board
    img[y:y + mat.shape[0], x:x + mat.shape[1]] = mat
    bx, by = x + margin, y + margin
    corners = [[bx / w, by / h], [(bx + side) / w, by / h],
               [(bx + side) / w, (by + side) / h], [bx / w, (by + side) / h]]
    return img, corners


# ---- parse_corners -----------------------------------------------------------

def test_parse_corners_accepts_any_order_and_clamps():
    pts = parse_corners(json.dumps([[0.9, 0.9], [0.1, 0.1], [0.9, 0.1], [1.03, 0.9]]))
    assert len(pts) == 4 and all(0 <= v <= 1 for p in pts for v in p)


@pytest.mark.parametrize("raw", [
    "not json",
    json.dumps([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]),             # 3 points
    json.dumps([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1]]),       # malformed point
    json.dumps([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [{}, 0.9]]),   # not a number
    json.dumps([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [True, 0.9]]), # bool
    json.dumps([[0.1, 0.1], [5, 0.1], [0.9, 0.9], [0.1, 0.9]]),    # far off the photo
    json.dumps([[0.5, 0.5], [0.52, 0.5], [0.52, 0.52], [0.5, 0.52]]),  # tiny
    json.dumps([[0.1, 0.1], [0.9, 0.1], [0.5, 0.2], [0.1, 0.9]]),  # not convex
    '[[NaN, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]',
])
def test_parse_corners_rejects_unusable_input(raw):
    with pytest.raises(ValueError):
        parse_corners(raw)


def test_parse_corners_absent_means_auto():
    assert parse_corners(None) is None and parse_corners("  ") is None


# ---- vision ------------------------------------------------------------------

def test_auto_detection_reports_board_corners():
    img, truth = _photo_with_board()
    res = vision.analyze_image(img, backend=vision.ClassicalBackend())
    assert res.fen.split()[0] == START
    got = np.array(res.board_corners)
    assert np.abs(got - np.array(truth)).max() < 0.01, got.round(3).tolist()


def test_user_corners_read_the_board():
    img, truth = _photo_with_board()
    res = vision.analyze_image(img, backend=vision.ClassicalBackend(), corners=truth)
    assert res.fen.split()[0] == START


def test_rough_user_corners_are_snapped_onto_the_grid():
    """Users click roughly -- here every corner ~1/3 of a square outside the
    grid, on the mat. The crop must still land exactly on the squares."""
    img, truth = _photo_with_board()
    t = np.array(truth)
    rough = (t + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [0.018, 0.024]).tolist()
    res = vision.analyze_image(img, backend=vision.ClassicalBackend(), corners=rough)
    assert res.fen.split()[0] == START
    assert np.abs(np.array(res.board_corners) - t).max() < 0.01


def test_user_corners_never_jump_far():
    """If the snap would move a corner more than ~3/4 of a square, the user's
    own corners are kept: their placement beats a doubtful automatic fit."""
    img, truth = _photo_with_board()
    t = np.array(truth)
    off = (t + [0.05, 0.0]).tolist()     # outline shifted 75 px = ~0.94 squares right
    res = vision.analyze_image(img, backend=vision.ClassicalBackend(), corners=off)
    # Kept exactly where the user put it (the true grid is 0.05 away).
    assert np.abs(np.array(res.board_corners) - np.array(off)).max() < 0.002


# ---- route -------------------------------------------------------------------

def _upload(client, img):
    ok, buf = cv2.imencode(".png", img)
    return client.post("/analyze", files={"image": ("board.png", buf.tobytes(), "image/png")},
                       data={"orientation": "white", "side_to_move": "white"})


def test_page_draws_detected_outline():
    client = TestClient(app)
    img, truth = _photo_with_board()
    r = _upload(client, img)
    assert r.status_code == 200
    assert 'id="photo-wrap"' in r.text and 'class="corner-handle"' in r.text
    assert 'data-failed="0"' in r.text
    m = __import__("re").search(r'data-corners="([^"]+)"', r.text)
    corners = json.loads(m.group(1).replace("&#34;", '"'))
    assert np.abs(np.array(corners) - np.array(truth)).max() < 0.01


def test_reread_with_user_corners_and_keep_them_on_orientation_change():
    client = TestClient(app)
    img, truth = _photo_with_board()
    _upload(client, img)
    r = client.post("/analyze", data={"orientation": "white", "side_to_move": "white",
                                      "corners": json.dumps(truth)})
    assert START in r.text and "Using the corners you marked" in r.text
    # Flip orientation without sending corners: the user's outline is reused.
    r = client.post("/analyze", data={"orientation": "black", "side_to_move": "white"})
    assert "Using the corners you marked" in r.text
    assert "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr" in r.text


def test_new_upload_forgets_previous_corners():
    client = TestClient(app)
    img, truth = _photo_with_board()
    _upload(client, img)
    client.post("/analyze", data={"orientation": "white", "side_to_move": "white",
                                  "corners": json.dumps(truth)})
    r = _upload(client, img)
    assert "Using the corners you marked" not in r.text


def test_bad_corners_show_an_error_not_a_crash():
    client = TestClient(app)
    img, _ = _photo_with_board()
    _upload(client, img)
    r = client.post("/analyze", data={"orientation": "white", "side_to_move": "white",
                                      "corners": "[[0.1,0.1],[0.2,0.2]]"})
    assert r.status_code == 200
    assert "Couldn&#39;t use those corners" in r.text or "Couldn't use those corners" in r.text

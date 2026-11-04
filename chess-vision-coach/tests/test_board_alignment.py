"""Board-finder alignment: the crop must sit exactly on the 8x8 grid.

Regression for a real failure on the samryan18 photos: the board is printed
on a mat with a coordinate-label margin. The old finder cropped the mat
(margin included) or a coarse sliding-scan window, landing up to half a
square off -- every piece then straddled two cells and was read on the
wrong square, with high confidence.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from chess_vision import vision

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _on_mat(board: np.ndarray, *, margin: int, canvas=(1100, 1500), at=(170, 400)):
    """Paste a rendered board onto a light 'mat' with a label strip, inside a
    darker 'floor' -- the layout of a real tournament roll-up board."""
    h, w = canvas
    img = np.full((h, w, 3), (90, 95, 100), np.uint8)           # floor
    side = board.shape[0]
    y, x = at
    mat = np.full((side + 2 * margin, side + 2 * margin, 3), (225, 230, 230), np.uint8)
    for i, ch in enumerate("abcdefgh"):                          # file labels
        cv2.putText(mat, ch, (margin + i * side // 8 + side // 16, margin // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    mat[margin:margin + side, margin:margin + side] = board
    img[y:y + mat.shape[0], x:x + mat.shape[1]] = mat
    return img, (x + margin, y + margin, side)


@pytest.mark.parametrize("margin", [40, 70])
@pytest.mark.parametrize("at", [(170, 400), (60, 90)])
def test_refined_corners_land_on_true_grid(margin, at):
    """Start from the outer mat (what the contour detector really finds) and
    require every refined corner within 3% of a square of the true grid."""
    board = cv2.imread(str(FIXTURES / "starting.png"))
    img, (x, y, side) = _on_mat(board, margin=margin, at=at)
    finder = vision.ClassicalBackend()
    mat_quad = np.array(
        [[[x - margin, y - margin]], [[x + side + margin, y - margin]],
         [[x + side + margin, y + side + margin]], [[x - margin, y + side + margin]]],
        dtype="float32",
    )
    refined, score = finder._refine_quad(img, mat_quad)
    got = finder._order_points(refined)
    want = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], float)
    err = np.abs(got - want).max()
    assert err <= 0.03 * side / 8, f"corner error {err:.1f}px; got {got.round().tolist()}"
    assert score > 0.85


@pytest.mark.parametrize("margin", [40, 70])
def test_find_board_crop_matches_board_pixels(margin):
    board = cv2.imread(str(FIXTURES / "starting.png"))
    img, _ = _on_mat(board, margin=margin)
    crop = vision.ClassicalBackend()._find_board(img)
    # Calibrated on this fixture: a 1/2/3 px shift gives a mean abs diff of
    # 5.7/10.9/14.4; a half-square shift (the old failure) gives ~56.
    # Require <= ~3 px, the same tolerance as the corner test.
    diff = np.abs(crop.astype(int) - board.astype(int)).mean()
    assert diff < 15, f"crop does not line up with the board (mean abs diff {diff:.1f})"


def test_grid_fit_rejects_half_square_offset():
    """The refinement score must clearly prefer the true grid over a grid
    shifted by half a square -- the property the old median score lacked."""
    board = cv2.imread(str(FIXTURES / "starting.png"))
    pad = np.full((800, 800, 3), 128, np.uint8)
    pad[80:720, 80:720] = board
    ii = cv2.integral(cv2.cvtColor(pad, cv2.COLOR_BGR2GRAY).astype(np.float64))
    fit = vision.ClassicalBackend._grid_fit
    aligned = fit(ii, 80, 80, 720, 720)
    half = fit(ii, 120, 120, 760, 760)       # half a square (40 px) off
    whole = fit(ii, 160, 80, 800, 720)       # one whole square off in x
    # python-chess glyphs are large enough to touch some corner patches, so
    # the synthetic board tops out around 0.9; the gaps are what matter.
    assert aligned > 0.85
    assert half < aligned - 0.25, (aligned, half)
    assert whole < aligned - 0.15, (aligned, whole)


def test_full_pipeline_reads_board_on_mat():
    """End to end through analyze_image with the classical backend (always
    available in CI): the position must come out exactly right."""
    board = cv2.imread(str(FIXTURES / "starting.png"))
    img, _ = _on_mat(board, margin=60)
    result = vision.analyze_image(img, orientation="white", side_to_move="w",
                                  backend=vision.ClassicalBackend())
    assert not result.detection_failed
    assert result.fen.split()[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

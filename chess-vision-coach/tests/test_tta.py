"""Unit tests for the test-time-augmentation merge in ModelBackend.

The merge rules are safety-critical (they decide whether a wrong read can
sit on the board untinted), so they are pinned here with scripted views —
no ONNX inference involved.
"""

from __future__ import annotations

import numpy as np
import pytest

import chess_vision.model_backend as mb


def _empty_grid():
    return [["." for _ in range(8)] for _ in range(8)]


def _empty_conf():
    return [[mb._EMPTY_CONF for _ in range(8)] for _ in range(8)]


def _empty_alts():
    return [[None for _ in range(8)] for _ in range(8)]


def _backend_with_views(primary, mirror_view):
    """ModelBackend whose two inference passes return scripted outputs.

    `mirror_view` must be given in MIRRORED coordinates (what inference on
    the flipped image would return); the merge un-mirrors it itself.
    """
    backend = mb.ModelBackend.__new__(mb.ModelBackend)
    calls = iter([primary, mirror_view])
    backend._infer_single = lambda board: next(calls)
    return backend


def _mirrored(grid, conf, alts):
    """Convert straight-coordinate outputs into mirrored coordinates."""
    return (
        [row[::-1] for row in grid],
        [row[::-1] for row in conf],
        [row[::-1] for row in alts],
    )


BOARD = np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def force_tta(monkeypatch):
    monkeypatch.setattr(mb, "_TTA_ENABLED", True)


def _views(primary_cells, mirror_cells):
    """Build (primary, mirror_view) from {(r,c): (sym, conf)} dicts."""
    pg, pc, pa = _empty_grid(), _empty_conf(), _empty_alts()
    for (r, c), (sym, cf) in primary_cells.items():
        pg[r][c], pc[r][c] = sym, cf
    mg, mc, ma = _empty_grid(), _empty_conf(), _empty_alts()
    for (r, c), (sym, cf) in mirror_cells.items():
        mg[r][c], mc[r][c] = sym, cf
    return (pg, pc, pa), _mirrored(mg, mc, ma)


def test_agreement_reinforces_trusted_cells():
    primary, mirror = _views({(4, 4): ("Q", 0.85)}, {(4, 4): ("Q", 0.95)})
    backend = _backend_with_views(primary, mirror)
    grid, conf, _ = backend._infer_board(BOARD)
    assert grid[4][4] == "Q"
    assert conf[4][4] == 0.95


def test_agreement_never_promotes_across_amber_threshold():
    # Both views agree but the primary was below the tint threshold: the
    # cell must STAY tinted — agreement on a wrong read must not hide it.
    primary, mirror = _views({(4, 4): ("Q", 0.60)}, {(4, 4): ("Q", 0.95)})
    backend = _backend_with_views(primary, mirror)
    _, conf, _ = backend._infer_board(BOARD)
    assert conf[4][4] == 0.60


def test_type_disagreement_keeps_primary_and_tints():
    primary, mirror = _views({(2, 3): ("K", 0.92)}, {(2, 3): ("Q", 0.97)})
    backend = _backend_with_views(primary, mirror)
    grid, conf, alts = backend._infer_board(BOARD)
    assert grid[2][3] == "K"  # primary read stands even at lower conf
    assert conf[2][3] <= mb._TTA_DISAGREE_CONF  # always tints
    assert alts[2][3][0][0] == "Q"  # mirror read offered to sanity layer


def test_mirror_fills_missed_cell_only_when_confident():
    primary, mirror = _views({}, {(6, 1): ("p", 0.90)})
    backend = _backend_with_views(primary, mirror)
    grid, conf, _ = backend._infer_board(BOARD)
    assert grid[6][1] == "p"
    assert conf[6][1] == pytest.approx(0.90 * mb._TTA_OCCUPANCY_DAMP)

    primary, mirror = _views({}, {(6, 1): ("p", 0.40)})
    backend = _backend_with_views(primary, mirror)
    grid, _, _ = backend._infer_board(BOARD)
    assert grid[6][1] == "."  # below _TTA_FILL_CONF: primary's empty wins


def test_primary_only_piece_is_kept_but_damped():
    primary, mirror = _views({(0, 0): ("r", 0.95)}, {})
    backend = _backend_with_views(primary, mirror)
    grid, conf, _ = backend._infer_board(BOARD)
    assert grid[0][0] == "r"
    assert conf[0][0] == pytest.approx(0.95 * mb._TTA_OCCUPANCY_DAMP)


def test_mirror_coordinates_are_unflipped():
    # A piece on c-file (col 2) in reality appears on col 5 in the mirrored
    # view; after the merge it must land back on col 2.
    primary, mirror = _views({}, {(3, 2): ("N", 0.90)})
    backend = _backend_with_views(primary, mirror)
    grid, _, _ = backend._infer_board(BOARD)
    assert grid[3][2] == "N"
    assert all(grid[3][c] == "." for c in range(8) if c != 2)


def test_tta_disabled_is_single_view_passthrough(monkeypatch):
    monkeypatch.setattr(mb, "_TTA_ENABLED", False)
    primary, _ = _views({(5, 5): ("B", 0.50)}, {})
    backend = mb.ModelBackend.__new__(mb.ModelBackend)
    backend._infer_single = lambda board: primary
    grid, conf, _ = backend._infer_board(BOARD)
    assert grid[5][5] == "B"
    assert conf[5][5] == 0.50

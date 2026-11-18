"""The learned board-corner candidate (chess_vision/board_corners.py) and its
training-label builder (bench/board_corners_prepare.py). Synthetic only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import board_corners_prepare as bcp  # noqa: E402
from chess_vision import board_corners as bc  # noqa: E402


def _raw(boxes):
    """YOLO-style output [1, 5, N] from (cx, cy, w, h, score) rows."""
    return np.array(boxes, dtype=np.float32).T[None]


def test_decode_maps_back_through_the_letterbox_and_suppresses_duplicates():
    # 1280x960 photo -> scale 0.5, pad 80 px top/bottom
    r, px, py = 0.5, 0, 80
    raw = _raw([
        (100, 180, 20, 20, 0.9), (102, 181, 20, 20, 0.6),  # same corner twice
        (500, 190, 20, 20, 0.8), (520, 500, 20, 20, 0.7), (90, 510, 20, 20, 0.95),
        (300, 300, 20, 20, 0.1),                            # below threshold
    ])
    pts, scores = bc.corner_points(raw, r, px, py)
    assert len(pts) == 4 and list(scores) == sorted(scores, reverse=True)
    assert [180.0, 860.0] in pts.tolist() and [200.0, 200.0] in pts.tolist()


def test_plausible_quad_rejects_degenerate_sets():
    shape = (600, 800, 3)
    good = np.array([[100, 100], [700, 120], [650, 500], [120, 480]], float)
    assert bc.plausible_quad(good, shape)
    assert not bc.plausible_quad(good[:3], shape)
    inside = good.copy()
    inside[3] = [400, 300]  # one point inside the triangle of the others
    assert not bc.plausible_quad(inside, shape)
    tiny = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert not bc.plausible_quad(tiny, shape)


def test_no_model_file_means_no_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv("CVC_CORNER_MODEL", str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(bc, "_runner", None)
    try:
        assert bc.predict_quad(np.zeros((100, 100, 3), np.uint8)) is None
        assert not bc.available()
    finally:
        bc._runner = None  # let later tests load the real configuration


def test_label_builder_boxes_each_visible_corner():
    corners = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], float)
    lines = bcp.corner_boxes(corners, 640, 640)
    assert len(lines) == 4
    cls, cx, cy, w, h = map(float, lines[0].split())
    assert cls == 0 and abs(cx - 100 / 640) < 1e-6 and abs(w - 50 / 640) < 1e-6
    off = corners.copy()
    off[2] = [700, 700]  # outside the photo
    assert len(bcp.corner_boxes(off, 640, 640)) == 3

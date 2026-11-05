"""bench/chessred_prepare.py converts ChessReD's annotation format, and
bench/full_eval.py --photos reads the result. Real ChessReD can't be
downloaded here (network policy), so a two-image dataset is built in its
exact JSON layout from a synthetic board with known corners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import chessred_prepare as cp  # noqa: E402
import full_eval as fe  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _fake_chessred(tmp: Path):
    board = cv2.imread(str(ROOT / "tests/fixtures/starting.png"))
    img = np.full((900, 1200, 3), 90, np.uint8)
    img[130:770, 280:920] = board                          # board at x 280-920, y 130-770
    (tmp / "images").mkdir()
    cv2.imwrite(str(tmp / "images/0.jpg"), img)
    cv2.imwrite(str(tmp / "images/1.jpg"), cv2.rotate(img, cv2.ROTATE_180))  # Black nearer
    cats = [{"id": 0, "name": "white-pawn"}, {"id": 1, "name": "black-king"},
            {"id": 2, "name": "empty"}]
    pieces = [{"image_id": 0, "category_id": 1, "chessboard_position": "e8"},
              {"image_id": 0, "category_id": 0, "chessboard_position": "a2"},
              {"image_id": 1, "category_id": 1, "chessboard_position": "e8"}]
    # Corners named from White's view: bottom_left = a1.
    white_view = {"top_left": [280, 130], "top_right": [920, 130],
                  "bottom_right": [920, 770], "bottom_left": [280, 770]}
    rot = {k: [1200 - x, 900 - y] for k, (x, y) in white_view.items()}
    data = {
        "categories": cats,
        "images": [{"id": 0, "path": "images/0.jpg", "width": 1200, "height": 900},
                   {"id": 1, "path": "images/1.jpg", "width": 1200, "height": 900},
                   {"id": 2, "path": "images/missing.jpg", "width": 1200, "height": 900}],
        "annotations": {"pieces": pieces,
                        "corners": [{"image_id": 0, "corners": white_view},
                                    {"image_id": 1, "corners": rot}]},
        "splits": {"chessred2k": {"test": {"n_samples": 3, "image_ids": [0, 1, 2]}}},
    }
    (tmp / "annotations.json").write_text(json.dumps(data))


def test_prepare_names_orientation_and_corners(tmp_path):
    _fake_chessred(tmp_path)
    out = tmp_path / "out"
    assert cp.main(["--annotations", str(tmp_path / "annotations.json"),
                    "--images-root", str(tmp_path), "--split", "test", "--out", str(out)]) == 0
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert names == ["4k3-8-8-8-8-8-8-8__b__1.jpg", "4k3-8-8-8-8-8-P7-8__w__0.jpg"]
    corners = json.loads((out / "corners.json").read_text())
    assert np.allclose(corners["4k3-8-8-8-8-8-P7-8__w__0.jpg"][0], [280 / 1200, 130 / 900])


def test_photo_folder_reads_truth_side_and_corners(tmp_path):
    _fake_chessred(tmp_path)
    out = tmp_path / "out"
    cp.main(["--annotations", str(tmp_path / "annotations.json"),
             "--images-root", str(tmp_path), "--split", "test", "--out", str(out)])
    items = {p.name: (orient, corners, give) for p, _truth, orient, corners, give
             in fe.photo_folder(out, use_corners=True)}
    assert items["4k3-8-8-8-8-8-8-8__b__1.jpg"][0] == "black"
    assert items["4k3-8-8-8-8-8-P7-8__w__0.jpg"][0] == "white"
    assert all(give and len(c) == 4 for _o, c, give in items.values())


def test_symbol_and_placement_helpers():
    assert cp.symbol("white-knight") == "N" and cp.symbol("black-queen") == "q"
    assert cp.symbol("empty") is None
    sq = {f + r: s for f, r, s in [("e", "1", "K"), ("e", "8", "k")]}
    assert cp.placement(sq) == "4k3/8/8/8/8/8/8/4K3"

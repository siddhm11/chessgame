"""Converters for the extra piece sets (bench/set3_prepare.py,
bench/set4_prepare.py) and the scorer's helpers (bench/score_more_sets.py).
Everything is synthetic: no dataset files are needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import set3_prepare as s3  # noqa: E402
import set4_prepare as s4  # noqa: E402
import score_more_sets as sm  # noqa: E402

POS = "4k3/pppp4/8/8/8/8/4PPPP/4K3"
# set-3 class ids (0 Hand, 1 bB ... 12 wR)
SET3_ID = {v: k for k, v in s3.SET3_TO_SYMBOL.items()}


# ── set 3 ────────────────────────────────────────────────────────────────────

def test_parse_name_groups_copies_by_video_and_frame():
    a = s3.parse_name(Path("8_Move_student_frame_000120_jpg.rf.9d3d8aa.jpg"))
    b = s3.parse_name(Path("8_Move_student_frame_000120_jpg.rf.1234abc.jpg"))
    assert (a.video, a.frame, a.source) == ("8_Move_student", 120, "8_Move_student_frame_000120")
    assert a.source == b.source and a.rf_hash != b.rf_hash
    ext = s3.parse_name(Path("ext_0059cba4-dd39-4fc6-9266-50e05dab087f_jpg.rf.ab12.jpg"))
    assert ext.video == "ext" and ext.frame is None


def test_read_boxes_handles_boxes_and_polygons():
    boxes = s3.read_boxes("8 0.5 0.5 0.1 0.2\n4 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n\n")
    assert boxes[0] == s3.Box(8, 0.5, 0.5, 0.1, 0.2)
    assert boxes[1].cls == 4
    assert boxes[1].cx == pytest.approx(0.2) and boxes[1].w == pytest.approx(0.2)


def test_base_point_sits_at_the_end_opposite_the_lean():
    b = s3.Box(10, 0.5, 0.5, 0.05, 0.10)  # tall box: 0.05 extra along y
    assert s3.base_point(b, (0, 0)) == pytest.approx((0.5, 0.5))
    assert s3.base_point(b, (0, 1)) == pytest.approx((0.5, 0.475))   # leans down: base on top
    assert s3.base_point(b, (0, -1)) == pytest.approx((0.5, 0.525))  # leans up: base below
    wide = s3.Box(10, 0.5, 0.5, 0.10, 0.05)
    assert s3.base_point(wide, (-1, 0)) == pytest.approx((0.525, 0.5))  # leans left


def _leaning_boxes(placement: str, lean_y: float = 0.4) -> list[s3.Box]:
    """Boxes for a position, each piece leaning down by lean_y * row / 7 squares."""
    grid = sm.grid_from_placement(placement)
    out = []
    for r in range(8):
        for c in range(8):
            if grid[r][c] == ".":
                continue
            extra = lean_y * r / 7 / 8
            d = 0.6 / 8
            # base disc centred on the square, top leaning downwards
            out.append(s3.Box(SET3_ID[grid[r][c]], (c + .5) / 8,
                              (r + .5) / 8 - d / 2 + (d + extra) / 2, d, d + extra))
    return out


def test_lean_is_estimated_and_the_position_recovered():
    boxes = _leaning_boxes("pppppppp/8/8/8/8/8/8/PPPPPPPP", lean_y=0.6)
    assert s3.estimate_lean(boxes) == (0, 1)
    pos_boxes = _leaning_boxes(POS, lean_y=0.6)
    d = s3.derive_grid(pos_boxes, (0, 1))
    assert d.clean and s3.placement(d.grid) == POS
    # Placing by box centre instead puts the long, leaning bottom pieces a square low.
    by_centre = s3.derive_grid(_leaning_boxes(POS, lean_y=1.4), (0, 0))
    assert s3.placement(by_centre.grid) != POS


def test_hand_and_collisions_make_a_frame_unclean():
    boxes = _leaning_boxes(POS) + [s3.Box(0, 0.5, 0.5, 0.3, 0.3)]
    assert "hand" in s3.derive_grid(boxes).issues
    dup = _leaning_boxes(POS)
    dup.append(s3.Box(dup[0].cls, dup[0].cx, dup[0].cy, dup[0].w, dup[0].h))
    assert any("two pieces" in i for i in s3.derive_grid(dup).issues)


def test_chess_issues():
    assert s3.chess_issues(sm.grid_from_placement(POS)) == []
    assert "pawn on rank 1/8" in s3.chess_issues(sm.grid_from_placement("P3k3/8/8/8/8/8/8/4K3"))
    assert "2 K" in s3.chess_issues(sm.grid_from_placement("4k3/8/8/8/8/8/8/3KK3"))


def test_transforms_move_points_like_pixels():
    img = np.zeros((8, 8), np.uint8)
    img[1, 6] = 255  # pixel centre (6.5, 1.5) -> point (0.8125, 0.1875)
    for flip in (False, True):
        for k in range(4):
            out = s3.tf_array(img, flip, k)
            y, x = np.argwhere(out == 255)[0]
            px, py = s3.tf_point(6.5 / 8, 1.5 / 8, flip, k)
            assert (x, y) == (int(px * 8), int(py * 8)), (flip, k)
    grid = sm.grid_from_placement(POS)
    for name, (f, k) in s3.FLIPS.items():  # each Roboflow flip is its own inverse
        twice = s3.rotate_grid(s3.rotate_grid(grid, f, k), f, k)
        assert twice == grid, name


def _board_image(placement: str, seed: int = 0) -> np.ndarray:
    """640x640 top-down board, a8 light, with an uneven light falloff (as
    from a lamp) so flips are detectable, and dark discs for pieces."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:640, 0:640]
    light = ((yy // 80 + xx // 80) % 2 == 0)
    img = np.where(light, 225, 60).astype(np.float32)
    img *= 0.65 + 0.35 * (xx / 640) * (1 - yy / 1280)
    grid = sm.grid_from_placement(placement)
    for r in range(8):
        for c in range(8):
            if grid[r][c] != ".":
                cv2.circle(img, (c * 80 + 40, r * 80 + 40), 22,
                           245 if grid[r][c].isupper() else 20, -1)
    img += rng.normal(0, 2, img.shape)
    return cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_detect_flip_identifies_each_roboflow_flip():
    ref = cv2.cvtColor(_board_image(POS), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(ref, (64, 64), interpolation=cv2.INTER_AREA)
    for name, (f, k) in s3.FLIPS.items():
        got, margin = s3.detect_flip(s3.tf_array(small, f, k), small)
        assert got == name and margin > 0.05


def test_cell_labels_use_our_class_order():
    lines = s3.cell_label_lines(sm.grid_from_placement("k7/8/8/8/8/8/8/7K"))
    assert lines == ["7 0.062500 0.062500 0.125000 0.125000",
                     "1 0.937500 0.937500 0.125000 0.125000"]


def _write_frame(root: Path, split: str, name: str, img, boxes):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "images" / f"{name}.jpg"), img)
    (root / split / "labels" / f"{name}.txt").write_text(
        "\n".join(f"{b.cls} {b.cx} {b.cy} {b.w} {b.h}" for b in boxes) + "\n")


def test_set3_main_end_to_end(tmp_path):
    """A tiny export: two originals of one video, plus a frame's 3 Roboflow
    copies (one unflipped, one flipped left-right, one flipped both ways)."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    video = "8_Move_student"  # a test-split video
    for i, seed in ((5, 1), (10, 2)):
        _write_frame(raw, "valid", f"{video}_frame_{i:06d}_jpg.rf.aa{i}", _board_image(POS, seed),
                     _leaning_boxes(POS))
    img, boxes = _board_image(POS, 3), _leaning_boxes(POS)
    for tag, (f, k) in (("c1", (False, 0)), ("c2", (True, 0)), ("c3", (False, 2))):
        _write_frame(raw, "train", f"{video}_frame_000015_jpg.rf.{tag}",
                     s3.tf_array(img, f, k), [s3.tf_box(b, f, k) for b in boxes])
    assert s3.main(["--raw", str(raw), "--out", str(out)]) == 0
    recs = [json.loads(line) for line in (out / "positions.jsonl").read_text().splitlines()]
    assert len(recs) == 5
    assert all(r["clean"] and r["placement"] == POS for r in recs), recs
    flips = {r["image"].split("__")[-1][:2]: r["flip_undone"] for r in recs if r["copies"] == 3}
    assert flips == {"c1": "none", "c2": "h", "c3": "hv"}
    # one evaluation copy per source frame, never a flipped one
    ev = [r for r in recs if r["eval"]]
    assert len(ev) == 3 and all(r["flip_undone"] == "none" for r in ev)
    # written images are unflipped: the c2 copy matches the original pixels
    c1 = cv2.imread(str(out / next(r["image"] for r in recs if r["image"].endswith("c1.jpg"))))
    c2 = cv2.imread(str(out / next(r["image"] for r in recs if r["image"].endswith("c2.jpg"))))
    assert np.abs(c1.astype(int) - c2.astype(int)).mean() < 3
    label = (out / "labels" / "test" / Path(ev[0]["image"]).with_suffix(".txt").name).read_text()
    assert "1 0.562500 0.937500" in label  # white king on e1, our class 1


# ── set 4 ────────────────────────────────────────────────────────────────────

def test_set4_label_validation_and_camera_side():
    assert s4.validate_label({"white_turn": True, "fen": POS}) == POS
    assert s4.camera_side({"white_turn": False, "fen": POS}) == "b"
    with pytest.raises(ValueError):
        s4.validate_label({"white_turn": True, "fen": "8/8/8"})
    with pytest.raises(ValueError):
        s4.validate_label({"fen": POS})


def test_order_quad():
    q = s4.order_quad([[10, 90], [90, 85], [12, 8], [95, 10]])
    assert q.tolist() == [[12, 8], [95, 10], [90, 85], [10, 90]]


def _angled_photo():
    """A board with a black frame on a wooden-ish table, seen at an angle."""
    sq = 50
    board = np.full((10 * sq, 10 * sq), 25, np.uint8)  # 1-square black frame
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                board[sq + r * sq: sq + (r + 1) * sq, sq + c * sq: sq + (c + 1) * sq] = 215
            else:
                board[sq + r * sq: sq + (r + 1) * sq, sq + c * sq: sq + (c + 1) * sq] = 35
    src = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
    dst = np.float32([[210, 70], [590, 75], [700, 520], [100, 515]])
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(board, m, (800, 600), borderValue=150)
    inner = cv2.perspectiveTransform(np.float32([[[50, 50], [450, 50], [450, 450], [50, 450]]]), m)[0]
    return photo, inner


def test_set4_find_corners_on_a_synthetic_angled_board():
    photo, truth = _angled_photo()
    quad, info = s4.find_corners(photo)
    assert quad is not None, info
    assert np.abs(quad - s4.order_quad(truth)).max() < 4, (quad, truth, info)


def test_set4_main_writes_full_eval_layout(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "test").mkdir(parents=True)
    (raw / "train").mkdir()
    photo, truth = _angled_photo()
    cv2.imwrite(str(raw / "test" / "01.png"), cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR))
    (raw / "test" / "01.json").write_text(json.dumps({"white_turn": False, "fen": POS}))
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"test-01": truth.tolist()}))
    assert s4.main(["--raw", str(raw), "--out", str(out), "--corners-override", str(override)]) == 0
    name = "4k3-pppp4-8-8-8-8-4PPPP-4K3__b__test-01.png"
    assert (out / "photos" / name).is_file()
    assert json.loads((out / "photos" / "side_to_move.json").read_text()) == {name: "b"}
    c = json.loads((out / "photos" / "corners.json").read_text())[name]
    assert c[0] == pytest.approx([truth[0][0] / 800, truth[0][1] / 600], abs=1e-4)
    # the scorer reads it back: Black nearer the camera, Black to move
    items = sm.photo_items(out / "photos", use_corners=True)
    assert len(items) == 1 and items[0]["orientation"] == "black" and items[0]["stm"] == "b"
    assert items[0]["truth"] == sm.grid_from_placement(POS) and items[0]["corners"] == c


# ── scorer helpers ───────────────────────────────────────────────────────────

def test_scorer_compare_counts_silent_errors():
    truth = sm.grid_from_placement(POS)
    pred = [row[:] for row in truth]
    pred[7][4] = "Q"  # wrong, confident
    pred[1][0] = "."  # wrong, tinted
    conf = [[1.0] * 8 for _ in range(8)]
    conf[1][0] = 0.5
    cmp = sm.compare(pred, conf, truth)
    assert cmp["wrong"] == 2 and cmp["silent"] == 1
    assert sm.placement_from_grid(truth) == POS

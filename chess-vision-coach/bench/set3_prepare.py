"""Prepare "set 3" (Hugging Face surawut/chess-pieces-dataset-yolo) for this app.

Source: https://huggingface.co/datasets/surawut/chess-pieces-dataset-yolo
(Roboflow export, workspace chess-detection-8s15a, project
chess-pieces-detection-1yb2h v4; license CC BY 4.0). Plastic tournament
Staunton pieces on a green/white vinyl board, filmed top-down, every frame
already warped to the board edges and stretched to 640x640 -- the same
framing our model sees (vision.locate_board warps the board edge-to-edge to
BOARD_PX=640, no margin, and ModelBackend resizes that to the model's 640).

Download (131 MB) and unzip into an empty folder of its own, then:

    python -I bench/set3_prepare.py --raw /data/set3/raw --out /data/set3_prepared

What it does
------------
1. Groups images by source video (filename before `_frame_`) and by source
   frame (filename before `_jpg.rf.<hash>`; Roboflow's 3 augmented copies of
   one frame share it). The `ext_<uuid>` images (Roboflow v3 "external") are
   mostly close-ups of a few squares, not whole boards: they cannot give a
   64-square position and are left out (counted in the summary).
2. Splits BY VIDEO (never by frame: neighbouring frames are near-duplicates):
   see SPLIT_BY_VIDEO.
3. Detects each copy's flip state (Roboflow flipped copies at random,
   horizontally and/or vertically, which mirrors the board) by matching it
   against the un-augmented frames of the same video. Flipped copies are
   flipped back, so every written image is a real, unmirrored board.
4. Rotates every frame of a video by the same multiple of 90 degrees so White
   is at the bottom and a8 is a light square (the orientation the app reads),
   voted by the video's clean frames (after undoing flips).
5. Derives the 64-square position from the boxes, placing each piece by the
   base of its box (pieces lean away from the camera axis, so the box centre
   can sit on the neighbouring square), and checks it: no Hand box, no two
   pieces on one square, one king per side, <= 8 pawns per side, no pawn on
   rank 1/8, base points not on a square edge, a8 light.
6. Writes, per split:
     images/<split>/<name>.jpg          the canonical 640x640 board crop
     labels/<split>/<name>.txt          YOLO labels in OUR model's class order
                                        (1=white_king ... 12=black_pawn), one
                                        full-cell box per piece -- the same
                                        style as prepare_finetune_dataset.py
     labels_boxes/<split>/<name>.txt    the original tight boxes, re-mapped to
                                        our classes and orientation
     positions.jsonl                    one record per written image: split,
                                        video, frame, copy, flip, placement,
                                        clean, issues
   Evaluation uses only clean frames, one copy per source frame (the
   un-augmented copy when there is one). Unclean frames go to
   images/rejected for inspection and are not meant for training.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Dataset classes (data.yaml): 0 Hand, then bB bK bN bP bQ bR wB wK wN wP wQ wR.
SET3_TO_SYMBOL = {1: "b", 2: "k", 3: "n", 4: "p", 5: "q", 6: "r",
                  7: "B", 8: "K", 9: "N", 10: "P", 11: "Q", 12: "R"}
HAND = 0
# Our model's class ids (ONNX metadata 'names'; prepare_finetune_dataset.py).
SYMBOL_TO_OURS = {"K": 1, "Q": 2, "R": 3, "B": 4, "N": 5, "P": 6,
                  "k": 7, "q": 8, "r": 9, "b": 10, "n": 11, "p": 12}

# Held-out splits by source video. Every video shows the same board, pieces
# and camera; holding out whole videos keeps near-duplicate neighbouring
# frames from leaking between train and test.
SPLIT_BY_VIDEO = {
    "Bonus-Long-Video-Label": "train",
    "2_move_student": "train",
    "6_Move_student": "val",
    "8_Move_student": "test",
    "2_Move_rotate_student": "test",
}

NAME_RE = re.compile(r"^(?P<stem>.+?)_(?:jpe?g|png)\.rf\.(?P<hash>[0-9a-f]+)\.jpe?g$", re.I)
# A base point closer than this (fraction of a square) to a square edge is
# ambiguous: the frame is not used for evaluation.
EDGE_MARGIN = 0.10


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class FrameInfo:
    path: Path
    split_dir: str          # 'train' / 'valid' in the Roboflow export
    video: str              # 'ext' for the external close-ups
    frame: int | None
    source: str             # source-frame key shared by augmented copies
    rf_hash: str


def parse_name(path: Path, split_dir: str = "") -> FrameInfo:
    m = NAME_RE.match(path.name)
    stem, h = (m.group("stem"), m.group("hash")) if m else (path.stem, "")
    if "_frame_" in stem:
        video, num = stem.rsplit("_frame_", 1)
        frame = int(num) if num.isdigit() else None
    else:
        video, frame = "ext", None
    return FrameInfo(path, split_dir, video, frame, stem, h)


def read_boxes(text: str) -> list[Box]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        if len(parts) > 5:  # polygon row: class x1 y1 x2 y2 ... -> bounding box
            xs = [float(v) for v in parts[1::2]]
            ys = [float(v) for v in parts[2::2]]
            boxes.append(Box(c, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                             max(xs) - min(xs), max(ys) - min(ys)))
        else:
            boxes.append(Box(c, *(float(v) for v in parts[1:5])))
    return boxes


def estimate_lean(boxes: list[Box], min_diff: float = 0.005) -> tuple[int, int]:
    """Direction pieces lean in this video, as (sx, sy) in {-1, 0, 1}.

    The frames were warped from a camera that sat beyond one edge of the
    board, so every piece leans the same way: away from the camera. The
    further a piece is from the camera, the more it leans, so its box grows
    along the lean axis. Measured on pawns (round, so any elongation is lean):
    compare how far boxes extend past their short side in each half of the
    board. +1 = leaning right / down, -1 = left / up, 0 = no measurable lean.
    """
    pawns = [b for b in boxes if SET3_TO_SYMBOL.get(b.cls) in ("P", "p")]

    def sign(along: str) -> int:
        lo, hi = [], []
        for b in pawns:
            pos = b.cx if along == "x" else b.cy
            ext = (b.w if along == "x" else b.h) - min(b.w, b.h)
            (lo if pos < 0.5 else hi).append(ext)
        if len(lo) < 5 or len(hi) < 5:
            return 0
        diff = float(np.mean(hi) - np.mean(lo))
        return 0 if abs(diff) < min_diff else (1 if diff > 0 else -1)

    return sign("x"), sign("y")


def base_point(b: Box, lean: tuple[int, int] = (0, 0)) -> tuple[float, float]:
    """Where the piece stands, in [0,1] image coordinates.

    A leaning piece's top lands further along the lean direction than its
    base. The base is a disc about as wide as the box's short side, at the
    end of the box opposite the lean; along an axis with no lean it is the
    box centre.
    """
    d = min(b.w, b.h)
    x0, x1 = b.cx - b.w / 2, b.cx + b.w / 2
    y0, y1 = b.cy - b.h / 2, b.cy + b.h / 2
    bx = {1: x0 + d / 2, -1: x1 - d / 2}.get(lean[0], b.cx)
    by = {1: y0 + d / 2, -1: y1 - d / 2}.get(lean[1], b.cy)
    return bx, by


# ── orientation: D4 transforms on images, grids and boxes ───────────────────
# A transform is (flip, k): first mirror left-right if flip, then rotate k
# quarter turns counter-clockwise (np.rot90 convention).

def tf_array(a: np.ndarray, flip: bool, k: int) -> np.ndarray:
    if flip:
        a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, k % 4))


def tf_point(x: float, y: float, flip: bool, k: int) -> tuple[float, float]:
    """Same transform for a point in [0,1] image coordinates (x right, y down)."""
    if flip:
        x = 1.0 - x
    for _ in range(k % 4):  # one CCW quarter turn: (x, y) -> (y, 1 - x)
        x, y = y, 1.0 - x
    return x, y


def tf_box(b: Box, flip: bool, k: int) -> Box:
    cx, cy = tf_point(b.cx, b.cy, flip, k)
    w, h = (b.h, b.w) if k % 2 else (b.w, b.h)
    return Box(b.cls, cx, cy, w, h)


# Roboflow's flips as (flip, k): none, horizontal, vertical, both (= 180 turn).
FLIPS = {"none": (False, 0), "h": (True, 0), "v": (True, 2), "hv": (False, 2)}


def detect_flip(img_gray: np.ndarray, reference: np.ndarray) -> tuple[str, float]:
    """Which of Roboflow's 4 flips maps `reference` (an un-augmented frame of
    the same static camera, e.g. the per-video median) onto this image.
    Returns (flip name, margin between the best and second-best score)."""
    def norm(a):
        a = a.astype(np.float32)
        return (a - a.mean()) / (a.std() + 1e-6)
    x = norm(img_gray)
    scores = {}
    for name, (f, k) in FLIPS.items():
        scores[name] = float((norm(tf_array(reference, f, k)) * x).mean())
    best = sorted(scores, key=scores.get, reverse=True)
    return best[0], scores[best[0]] - scores[best[1]]


# ── per-square position from boxes ──────────────────────────────────────────

@dataclass
class Derived:
    grid: list[list[str]]
    issues: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.issues


def derive_grid(boxes: list[Box], lean: tuple[int, int] = (0, 0)) -> Derived:
    """8x8 grid in IMAGE orientation (row 0 = top of the image)."""
    grid = [["."] * 8 for _ in range(8)]
    issues: list[str] = []
    for b in boxes:
        if b.cls == HAND:
            issues.append("hand")
            continue
        sym = SET3_TO_SYMBOL.get(b.cls)
        if sym is None:
            issues.append(f"unknown class {b.cls}")
            continue
        x, y = base_point(b, lean)
        fx, fy = x * 8, y * 8
        col, row = int(np.clip(fx, 0, 7.999)), int(np.clip(fy, 0, 7.999))
        edge = min(fx - col, col + 1 - fx, fy - row, row + 1 - fy)
        if edge < EDGE_MARGIN and not (fx < EDGE_MARGIN or fx > 8 - EDGE_MARGIN
                                       or fy < EDGE_MARGIN or fy > 8 - EDGE_MARGIN):
            issues.append(f"{sym} on a square edge")
        if grid[row][col] != ".":
            issues.append(f"two pieces on one square ({grid[row][col]}, {sym})")
            continue
        grid[row][col] = sym
    return Derived(grid, sorted(set(issues)))


def chess_issues(grid: list[list[str]]) -> list[str]:
    """Checks that need the canonical orientation (row 0 = rank 8)."""
    flat = [s for row in grid for s in row]
    out = []
    for king in "Kk":
        if flat.count(king) != 1:
            out.append(f"{flat.count(king)} {king}")
    for pawn in "Pp":
        if flat.count(pawn) > 8:
            out.append(f"{flat.count(pawn)} {pawn}")
    if any(s in "Pp" for s in grid[0] + grid[7]):
        out.append("pawn on rank 1/8")
    for colour in (str.isupper, str.islower):
        if sum(1 for s in flat if s != "." and colour(s)) > 16:
            out.append("more than 16 pieces of one colour")
    return out


def light_top_left(img_gray: np.ndarray, grid: list[list[str]]) -> bool | None:
    """True if the square at the top-left (row 0, col 0 parity) is light,
    judged from the unoccupied squares' centres. None if undecidable."""
    n = img_gray.shape[0] // 8
    even, odd = [], []
    for r in range(8):
        for c in range(8):
            if grid[r][c] != ".":
                continue
            patch = img_gray[r * n + n // 4: (r + 1) * n - n // 4,
                             c * n + n // 4: (c + 1) * n - n // 4]
            (even if (r + c) % 2 == 0 else odd).append(float(np.median(patch)))
    if not even or not odd:
        return None
    return float(np.median(even)) > float(np.median(odd))


def white_below(grid: list[list[str]]) -> float:
    """Mean row of White's pieces minus Black's (positive: White lower)."""
    w = [r for r in range(8) for c in range(8) if grid[r][c].isupper()]
    b = [r for r in range(8) for c in range(8) if grid[r][c].islower()]
    if not w or not b:
        return 0.0
    return sum(w) / len(w) - sum(b) / len(b)


def rotate_grid(grid, flip: bool, k: int):
    return [list(r) for r in tf_array(np.array(grid, dtype=object), flip, k)]


def placement(grid: list[list[str]]) -> str:
    rows = []
    for row in grid:
        s, run = "", 0
        for ch in row:
            if ch == ".":
                run += 1
            else:
                s += (str(run) if run else "") + ch
                run = 0
        rows.append(s + (str(run) if run else ""))
    return "/".join(rows)


def cell_label_lines(grid: list[list[str]]) -> list[str]:
    """Full-cell YOLO boxes in our class order (prepare_finetune_dataset.py style)."""
    return [f"{SYMBOL_TO_OURS[s]} {(c + .5) / 8:.6f} {(r + .5) / 8:.6f} 0.125000 0.125000"
            for r, row in enumerate(grid) for c, s in enumerate(row) if s in SYMBOL_TO_OURS]


def box_label_lines(boxes: list[Box]) -> list[str]:
    return [f"{SYMBOL_TO_OURS[SET3_TO_SYMBOL[b.cls]]} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
            for b in boxes if b.cls in SET3_TO_SYMBOL]


def video_rotation(frames: list[tuple[np.ndarray, list[list[str]]]]) -> tuple[int, dict]:
    """Quarter turns (CCW) that put this video's un-augmented frames in the
    canonical orientation: a8 light (top-left) and White at the bottom.
    `frames` are (gray image, image-orientation grid) of clean, unflipped frames."""
    votes = collections.Counter()
    for gray, grid in frames:
        for k in range(4):
            g = rotate_grid(grid, False, k)
            light = light_top_left(tf_array(gray, False, k), g)
            if light and white_below(g) > 0:
                votes[k] += 1
    if not votes:
        raise ValueError("could not determine the video's orientation")
    return votes.most_common(1)[0][0], dict(votes)


def main(argv=None) -> int:
    import cv2  # local: keeps the pure helpers importable without OpenCV

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, required=True, help="unzipped dataset (has train/, valid/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    infos = []
    for split_dir in ("train", "valid", "test"):
        for p in sorted((args.raw / split_dir / "images").glob("*.jpg")):
            infos.append(parse_name(p, split_dir))
    if not infos:
        print(f"no images under {args.raw}/*/images", file=sys.stderr)
        return 1

    def label_path(fi: FrameInfo) -> Path:
        return fi.path.parent.parent / "labels" / (fi.path.stem + ".txt")

    def load(fi: FrameInfo):
        img = cv2.imread(str(fi.path))
        if img is None or img.shape[:2] != (640, 640):
            raise ValueError(f"{fi.path.name}: unreadable or not 640x640")
        lp = label_path(fi)
        boxes = read_boxes(lp.read_text()) if lp.exists() else []
        return img, boxes

    summary = collections.Counter()
    summary["images"] = len(infos)
    summary["ext_closeups_skipped"] = sum(fi.video == "ext" for fi in infos)
    by_video = collections.defaultdict(list)
    for fi in infos:
        if fi.video != "ext":
            by_video[fi.video].append(fi)
    unknown = sorted(set(by_video) - set(SPLIT_BY_VIDEO))
    if unknown:
        print(f"videos without a split assignment: {unknown}", file=sys.stderr)
        return 1

    for d in ("images", "labels", "labels_boxes"):
        for s in ("train", "val", "test", "rejected"):
            (args.out / d / s).mkdir(parents=True, exist_ok=True)
    records = []
    video_meta = {}
    for video, fis in sorted(by_video.items()):
        split = SPLIT_BY_VIDEO[video]
        copies = collections.Counter(fi.source for fi in fis)
        # Roboflow keeps the original among each frame's 3 copies and only
        # augments its own train split, so single-copy frames are originals.
        originals = [fi for fi in fis if copies[fi.source] == 1]

        def small(img):
            return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64),
                              interpolation=cv2.INTER_AREA)

        loaded = {fi.path: load(fi) for fi in fis}
        ref = np.median(np.stack([small(loaded[fi.path][0]) for fi in originals]), axis=0)
        # 1. Undo each copy's flip: everything below is in the camera's frame.
        unflipped = {}
        for fi in fis:
            img, boxes = loaded[fi.path]
            if copies[fi.source] == 1:
                flip_name, margin = "none", float("inf")
            else:
                flip_name, margin = detect_flip(small(img), ref)
            f0, k0 = FLIPS[flip_name]  # each Roboflow flip is its own inverse
            unflipped[fi.path] = (tf_array(img, f0, k0), [tf_box(b, f0, k0) for b in boxes],
                                  flip_name, margin)
        # 2. The video's lean direction, then its canonical rotation.
        lean = estimate_lean([b for v in unflipped.values() for b in v[1]])
        orig_frames = []  # every copy is unflipped by now, so all can vote
        for fi in fis:
            img, boxes, _, _ = unflipped[fi.path]
            d = derive_grid(boxes, lean)
            if d.clean:
                orig_frames.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), d.grid))
        k, votes = video_rotation(orig_frames)
        video_meta[video] = {"split": split, "lean_xy": lean, "quarter_turns_ccw": k,
                             "rotation_votes": votes, "source_frames": len(copies),
                             "images": len(fis), "unaugmented": len(originals)}
        # 3. Per copy: derive, rotate to canonical, check, write. One
        #    evaluation copy per source frame: the unflipped one.
        eval_pick: dict[str, Path] = {}
        for fi in sorted(fis, key=lambda f: f.path.name):
            img_u, boxes_u, flip_name, margin = unflipped[fi.path]
            d = derive_grid(boxes_u, lean)
            grid = rotate_grid(d.grid, False, k)
            img_c = tf_array(img_u, False, k)
            boxes_c = [tf_box(b, False, k) for b in boxes_u]
            issues = list(d.issues)
            if "hand" not in issues:
                issues += chess_issues(grid)
                if light_top_left(cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY), grid) is False:
                    issues.append("a8 dark")
            if margin < 0.05:
                issues.append(f"flip uncertain ({margin:.3f})")
            clean = not issues
            used_for_eval = (clean and split in ("val", "test") and flip_name == "none"
                             and fi.source not in eval_pick)
            if used_for_eval:
                eval_pick[fi.source] = fi.path
            out_split = split if clean else "rejected"
            name = f"{fi.source}__{fi.rf_hash[:8]}"
            cv2.imwrite(str(args.out / "images" / out_split / f"{name}.jpg"), img_c,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (args.out / "labels" / out_split / f"{name}.txt").write_text(
                "\n".join(cell_label_lines(grid)) + "\n")
            (args.out / "labels_boxes" / out_split / f"{name}.txt").write_text(
                "\n".join(box_label_lines(boxes_c)) + "\n")
            records.append({
                "image": f"images/{out_split}/{name}.jpg", "split": split, "video": video,
                "frame": fi.frame, "source": fi.source, "copies": copies[fi.source],
                "flip_undone": flip_name, "placement": placement(grid),
                "clean": clean, "eval": used_for_eval, "issues": issues,
            })
            summary[f"{split}:{'clean' if clean else 'unclean'}"] += 1
            for i in issues:
                summary["issue:" + re.sub(r"\(.*\)|\d+ ", "", i).strip()] += 1
        summary[f"{split}:eval_frames"] += len(eval_pick)

    with open(args.out / "positions.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(
        {"counts": dict(sorted(summary.items())), "videos": video_meta}, indent=1))
    (args.out / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: board\n  1: white_king\n  2: white_queen\n  3: white_rook\n"
        "  4: white_bishop\n  5: white_knight\n  6: white_pawn\n  7: black_king\n"
        "  8: black_queen\n  9: black_rook\n  10: black_bishop\n  11: black_knight\n"
        "  12: black_pawn\n")
    print(json.dumps({"counts": dict(sorted(summary.items())), "videos": video_meta}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

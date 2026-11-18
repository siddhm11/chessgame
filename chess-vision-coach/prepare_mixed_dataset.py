"""Build a YOLO training set from ChessReD2K + the samryan18 train split.

Every training image is produced exactly the way the app produces the
model's input, so the model trains on what it will see:

  photo -> vision._load_bgr (EXIF-upright, capped at MAX_WORK_SIDE)
        -> ClassicalBackend.locate_board(img, corners)   (warp to 640x640)
        -> if the warp is sideways, rotate it upright (as ModelBackend does)

ChessReD2K: the annotated board corners are passed as the "user corners"
(the same call bench/full_eval.py --use-corners makes). The square of every
piece comes from the annotation (`chessboard_position`), never from where its
box lands: each square centre is mapped board -> photo (annotated corners,
named from White's side: top_left = a8) -> warped crop (the quad
locate_board used), giving the cell it occupies in the crop.

Labels use the same convention as prepare_finetune_dataset.py: one
cell-sized box (1/8 x 1/8) centred on the piece's square. That is the
contract ModelBackend decodes (box centre -> square). Boxes around the whole
piece would not work on angled photos: a tall piece leans up to two squares
"north" in the warp, so its box centre lands on the wrong square, and its
box overlaps the piece behind it enough for NMS to delete one of them.

samryan18: the 388 photos of the shipped model's training split
(bench/samryan18_split.json), re-cropped with today's board finder.

ChessReD is CC BY-NC-SA 4.0: weights trained on it are non-commercial.
Treat the archive as untrusted: run with `python -I`, paths as arguments.

  python -I prepare_mixed_dataset.py --annotations /data/chessred/annotations.json \
      --images-root /data/chessred/extracted --out /data/mix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chess_vision import vision  # noqa: E402
from chess_vision.vision import BOARD_PX, ClassicalBackend  # noqa: E402

CELL = BOARD_PX // 8
FILES = "abcdefgh"
CLASS_NAMES = [
    "board",
    "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
]
FEN_TO_CLASS = {"K": 1, "Q": 2, "R": 3, "B": 4, "N": 5, "P": 6,
                "k": 7, "q": 8, "r": 9, "b": 10, "n": 11, "p": 12}
PIECE = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}


def symbol(name: str) -> str | None:
    if "-" not in name:
        return None
    colour, kind = name.split("-", 1)
    s = PIECE[kind]
    return s.upper() if colour == "white" else s


def label_lines(cells: dict[tuple[int, int], str]) -> list[str]:
    f = 1.0 / 8
    return [f"{FEN_TO_CLASS[s]} {(c + .5) * f:.6f} {(r + .5) * f:.6f} {f:.6f} {f:.6f}"
            for (r, c), s in sorted(cells.items())]


def chessred_sample(finder, path: Path, corners: dict, size: tuple[int, int],
                    pieces: dict[str, str]):
    """(crop, {(row, col): symbol}) in the app's input frame, or None + reason."""
    img = vision._load_bgr(str(path))
    h, w = img.shape[:2]
    W, H = size
    if abs(w / h - W / H) > 0.01:
        return None, "orientation mismatch between annotation and decoded image"
    k = w / W
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    board_pts = np.array([corners[n] for n in names], dtype=np.float32) * k
    found = finder.locate_board(img, (board_pts / [w, h]).tolist())

    # board (file, rank-from-top) in [0, 8]^2 -> photo -> crop
    to_photo = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [8, 0], [8, 8], [0, 8]]), board_pts)
    to_crop = cv2.getPerspectiveTransform(
        np.float32(found.quad).reshape(4, 2),
        np.float32([[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]]))
    m = to_crop @ to_photo
    centres = np.float32([[[c + .5, r + .5]] for r in range(8) for c in range(8)])
    mapped = cv2.perspectiveTransform(centres, m).reshape(64, 2)
    cell_of = {}
    for i, (x, y) in enumerate(mapped):
        r, c = divmod(i, 8)
        cr, cc = int(np.floor(y / CELL)), int(np.floor(x / CELL))
        # The square centre must sit well inside its cell, or the crop is not
        # on the grid and the labels would be ambiguous.
        fx, fy = x / CELL - cc, y / CELL - cr
        if not (0 <= cr < 8 and 0 <= cc < 8) or min(fx, fy, 1 - fx, 1 - fy) < 0.2:
            return None, "crop not on the annotated grid"
        cell_of[f"{FILES[c]}{8 - r}"] = (cr, cc)
    if len(set(cell_of.values())) != 64:
        return None, "square -> cell mapping not one-to-one"

    crop = found.image
    # Sideways warp (a8 at a right-angle corner): rotate upright, as the app does.
    a8 = cell_of["a8"]
    if a8 in ((0, 7), (7, 0)):
        code = cv2.ROTATE_90_COUNTERCLOCKWISE if a8 == (0, 7) else cv2.ROTATE_90_CLOCKWISE
        crop = cv2.rotate(crop, code)
        if code == cv2.ROTATE_90_COUNTERCLOCKWISE:   # (r, c) -> (7 - c, r)
            cell_of = {s: (7 - c, r) for s, (r, c) in cell_of.items()}
        else:                                        # (r, c) -> (c, 7 - r)
            cell_of = {s: (c, 7 - r) for s, (r, c) in cell_of.items()}
    if (cell_of["a8"], cell_of["h8"]) not in (((0, 0), (0, 7)), ((7, 7), (7, 0))):
        return None, "mirrored mapping"
    return crop, {cell_of[sq]: s for sq, s in pieces.items()}


def write(out: Path, split: str, stem: str, crop, cells) -> None:
    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines(cells)))


def fen_cells(stem: str) -> dict[tuple[int, int], str]:
    cells = {}
    for r, row in enumerate(stem.split("-")):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                cells[(r, c)] = ch
                c += 1
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samryan", type=Path, default=Path("/tmp/chess-dataset/labeled_originals"))
    ap.add_argument("--samryan-split", type=Path, default=HERE / "bench" / "samryan18_split.json")
    ap.add_argument("--samryan-repeat", type=int, default=1,
                    help="write each samryan18 train photo this many times (rebalancing)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    cats = {c["id"]: symbol(c["name"]) for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    corners = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}
    pieces: dict[int, dict[str, str]] = {}
    for p in data["annotations"]["pieces"]:
        s = cats.get(p["category_id"])
        if s:
            pieces.setdefault(p["image_id"], {})[p["chessboard_position"]] = s

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    finder = ClassicalBackend()
    counts = {}
    for split in ("train", "val"):
        ids = data["splits"]["chessred2k"][split]["image_ids"]
        if args.limit:
            ids = ids[: args.limit]
        ok, skipped = 0, {}
        for n, image_id in enumerate(ids, 1):
            im = images[image_id]
            src = args.images_root / im["path"]
            if not src.is_file():
                skipped["missing file"] = skipped.get("missing file", 0) + 1
                continue
            crop, cells = chessred_sample(finder, src, corners[image_id],
                                          (im["width"], im["height"]), pieces.get(image_id, {}))
            if crop is None:
                skipped[cells] = skipped.get(cells, 0) + 1
                continue
            write(args.out, split, f"cr_{image_id}", crop, cells)
            ok += 1
            if n % 100 == 0:
                print(f"  chessred {split}: {n}/{len(ids)}", flush=True)
        counts[f"chessred_{split}"] = ok
        print(f"chessred2k {split}: {ok} written, skipped {skipped}", flush=True)

    split = json.loads(args.samryan_split.read_text())
    for name in ("train", "val"):
        ok = 0
        for stem in split[name]:
            img = vision._load_bgr(str(args.samryan / f"{stem}.JPG"))
            crop = finder.locate_board(img).image
            reps = args.samryan_repeat if name == "train" else 1
            for k in range(reps):
                write(args.out, name, f"sr_{stem}" + (f"_{k}" if k else ""), crop, fen_cells(stem))
            ok += 1
        counts[f"samryan_{name}"] = ok
        print(f"samryan18 {name}: {ok} photos", flush=True)

    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
prepare_finetune_dataset.py
===========================
Converts the samryan18/chess-dataset into YOLO detection format for
fine-tuning the yolov8m model on real tournament Staunton photos.

What it does:
  1. Reads each labeled photo from /tmp/chess-dataset/labeled_originals/
  2. Decodes the FEN from the filename (dashes replace slashes)
  3. Crops and warps the board to 640x640 using the existing board detector
  4. Generates YOLO bounding boxes: one box per piece, using the exact grid cell
  5. Filters out photos where board detection looks unreliable (low score)
  6. Writes train/val splits to finetune_data/

Output layout:
  finetune_data/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt    (YOLO format: class cx cy w h, all normalized)
    labels/val/*.txt
    dataset.yaml          (Ultralytics training config)

Run:
  python prepare_finetune_dataset.py

Then upload finetune_data/ to Colab and run the fine-tuning notebook.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from chess_vision.vision import ClassicalBackend

# ── config ────────────────────────────────────────────────────────────────────
PHOTOS_DIR   = Path("/tmp/chess-dataset/labeled_originals")
OUT_DIR      = Path("finetune_data")
BOARD_PX     = 640
CELL         = BOARD_PX // 8
VAL_FRACTION = 0.15
MIN_SCORE    = 0.70   # discard photos where board detection looks unreliable
RANDOM_SEED  = 42

# yolov8m class IDs (must match the ONNX model's metadata 'names' dict)
FEN_TO_CLASS = {
    "K": 1, "Q": 2, "R": 3, "B": 4, "N": 5, "P": 6,
    "k": 7, "q": 8, "r": 9, "b": 10, "n": 11, "p": 12,
}

CLASS_NAMES = [
    "board",
    "white_king", "white_queen", "white_rook", "white_bishop",
    "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop",
    "black_knight", "black_pawn",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def filename_to_fen(stem: str) -> str:
    """'1B2b3-Kp6-8-8-2k5-8-8-8' → '1B2b3/Kp6/8/8/2k5/8/8/8'"""
    return stem.replace("-", "/")


def fen_to_grid(fen_placement: str) -> list[list[str]]:
    """FEN placement string → 8x8 list, row 0 = rank 8 (top of board)."""
    grid: list[list[str]] = []
    for rank_str in fen_placement.split("/"):
        row: list[str] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["."] * int(ch))
            else:
                row.append(ch)
        if len(row) != 8:
            raise ValueError(f"Bad rank: {rank_str!r}")
        grid.append(row)
    if len(grid) != 8:
        raise ValueError(f"FEN has {len(grid)} ranks, expected 8")
    return grid


def grid_to_yolo_lines(grid: list[list[str]]) -> list[str]:
    """
    Convert grid to YOLO label lines.
    Each piece occupies its cell; box = full cell (0.125 x 0.125).
    cx = (col + 0.5) / 8,  cy = (row + 0.5) / 8
    """
    lines = []
    cell_frac = 1.0 / 8
    for r, row in enumerate(grid):
        for c, sym in enumerate(row):
            cls = FEN_TO_CLASS.get(sym)
            if cls is None:
                continue
            cx = (c + 0.5) * cell_frac
            cy = (r + 0.5) * cell_frac
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {cell_frac:.6f} {cell_frac:.6f}")
    return lines


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)

    photos = sorted(PHOTOS_DIR.glob("*.JPG"))
    if not photos:
        print(f"ERROR: no .JPG files found in {PHOTOS_DIR}")
        sys.exit(1)
    print(f"Found {len(photos)} photos in {PHOTOS_DIR}")

    finder = ClassicalBackend()

    accepted: list[tuple[np.ndarray, list[str], str]] = []
    skipped = 0

    for i, photo in enumerate(photos, 1):
        stem = photo.stem
        try:
            fen_placement = filename_to_fen(stem)
            grid = fen_to_grid(fen_placement)
        except ValueError as e:
            print(f"  [skip] {stem}: bad FEN — {e}")
            skipped += 1
            continue

        img = cv2.imread(str(photo))
        if img is None:
            print(f"  [skip] {stem}: unreadable image")
            skipped += 1
            continue

        board_crop = finder._find_board(img)
        score = finder._checkerboard_score(board_crop)

        if score < MIN_SCORE:
            print(f"  [skip] {stem}: low board score {score:.3f}")
            skipped += 1
            continue

        yolo_lines = grid_to_yolo_lines(grid)
        if not yolo_lines:
            print(f"  [skip] {stem}: no pieces in FEN (empty board?)")
            skipped += 1
            continue

        accepted.append((board_crop, yolo_lines, stem))
        if i % 50 == 0:
            print(f"  processed {i}/{len(photos)}, accepted {len(accepted)}, skipped {skipped}")

    print(f"\nAccepted {len(accepted)} / {len(photos)} photos (skipped {skipped})")
    if len(accepted) < 20:
        print("ERROR: too few photos accepted — check MIN_SCORE or PHOTOS_DIR")
        sys.exit(1)

    # ── train / val split ────────────────────────────────────────────────────
    random.shuffle(accepted)
    n_val = max(1, int(len(accepted) * VAL_FRACTION))
    val_items   = accepted[:n_val]
    train_items = accepted[n_val:]
    print(f"Split: {len(train_items)} train / {len(val_items)} val")

    # ── write files ──────────────────────────────────────────────────────────
    for split_name, items in [("train", train_items), ("val", val_items)]:
        img_dir = OUT_DIR / "images" / split_name
        lbl_dir = OUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for board_crop, yolo_lines, stem in items:
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), board_crop)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_path = OUT_DIR / "dataset.yaml"
    abs_out = OUT_DIR.resolve()
    yaml_content = f"""\
# Chess piece detection — fine-tuning dataset
# Generated by prepare_finetune_dataset.py

path: {abs_out}
train: images/train
val:   images/val

nc: 13
names: {CLASS_NAMES}
"""
    yaml_path.write_text(yaml_content)

    print(f"\nDataset written to: {OUT_DIR.resolve()}/")
    print(f"  {len(train_items)} train images")
    print(f"  {len(val_items)}   val images")
    print(f"  dataset.yaml ready for Ultralytics training")
    print("\nNext step: upload finetune_data/ to Google Colab and run:")
    print("  yolo detect train model=yolov8m.pt data=finetune_data/dataset.yaml epochs=50 imgsz=640")


if __name__ == "__main__":
    main()

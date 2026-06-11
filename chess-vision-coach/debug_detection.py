"""
Debug: for a few benchmark photos, save the warped board crop and overlay
YOLO raw detections so we can see whether the problem is board detection
or YOLO recognition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import cv2, numpy as np
from pathlib import Path
from chess_vision.vision import ClassicalBackend
from chess_vision.model_backend import ModelBackend

OUT = Path("debug_out")
OUT.mkdir(exist_ok=True)

PHOTOS_DIR = Path("/tmp/chess-dataset/labeled_originals")
photos = sorted(PHOTOS_DIR.glob("*.JPG"))[:6]

classical = ClassicalBackend()
model = ModelBackend()

BOARD_PX = 640
CELL = BOARD_PX // 8

for photo in photos:
    img = cv2.imread(str(photo))
    name = photo.stem

    # --- 1. Save the board crop (what classical sees after warp) ---
    board_crop = classical._find_board(img)
    cv2.imwrite(str(OUT / f"{name}__board_crop.jpg"), board_crop)

    # --- 2. Run YOLO on the cropped board and overlay raw detections ---
    raw = model.analyze(board_crop)

    vis = board_crop.copy()
    # Draw grid
    for i in range(9):
        cv2.line(vis, (i * CELL, 0), (i * CELL, BOARD_PX), (80, 80, 80), 1)
        cv2.line(vis, (0, i * CELL), (BOARD_PX, i * CELL), (80, 80, 80), 1)

    # Overlay per-square predictions from the raw FEN grid
    fen_rows = raw.grid  # list of 8 rows, each 8 chars
    for r, row in enumerate(fen_rows):
        for c, sym in enumerate(row):
            if sym == '.':
                continue
            x1, y1 = c * CELL, r * CELL
            x2, y2 = x1 + CELL, y1 + CELL
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, sym, (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    cv2.imwrite(str(OUT / f"{name}__yolo_detections.jpg"), vis)

    # --- 3. Show truth vs predicted summary ---
    truth_fen = name.replace('-', '/')
    print(f"\n{name[:40]}")
    print(f"  Truth FEN : {truth_fen}")
    print(f"  YOLO grid : {''.join(''.join(r) for r in fen_rows)}")
    print(f"  Pieces detected: {sum(1 for r in fen_rows for s in r if s != '.')}")
    truth_pieces = sum(1 for ch in truth_fen if ch.isalpha())
    print(f"  Truth pieces:    {truth_pieces}")

print(f"\nDebug images saved to: {OUT}/")
print("Files:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")

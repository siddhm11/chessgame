"""How often does the sideways-board detector get it right?

For every samryan18 photo: locate the board, then turn the board crop by
0/90/180/270 degrees. 90 and 270 must be detected as sideways, 0 and 180 not
(180 is the orientation setting's job). Compares the detectors available in
ModelBackend.

    CVC_BACKEND=model .venv/bin/python bench/rotation_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_vision import vision  # noqa: E402
from chess_vision.model_backend import ModelBackend  # noqa: E402

DATASET = Path("/tmp/chess-dataset/labeled_originals")
TURNS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main() -> None:
    backend = ModelBackend()
    finder = backend._board_finder
    detectors = {
        "two_corners": ModelBackend._board_needs_rotation_two_corners,
        "all_squares": ModelBackend._board_needs_rotation,
    }
    errors = {name: [] for name in detectors}
    n = 0
    for photo in sorted(DATASET.iterdir()):
        img = vision._load_bgr(str(photo))
        crop = finder.locate_board(img)
        if crop is None or crop.image is None:
            continue
        board = crop.image
        for deg, code in TURNS.items():
            b = board if code is None else cv2.rotate(board, code)
            truth = deg in (90, 270)
            n += 1
            for name, det in detectors.items():
                if det(b) != truth:
                    errors[name].append((photo.name, deg))
    for name, errs in errors.items():
        print(f"{name:12s} wrong {len(errs):4d} / {n}")
        for e in errs[:8]:
            print("    ", *e)


if __name__ == "__main__":
    main()

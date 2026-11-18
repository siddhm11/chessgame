"""Build a YOLO training set for the board-corner detector from ChessReD2K.

ChessReD2K (data.4tu.nl dataset 99b5c721-280b-450b-b058-b2900b69a90f,
CC BY-NC-SA 4.0) annotates the 4 corners of the playing area on 2,078 phone
photos. The detector learns ONE class, "corner": a small box centred on each
corner. Corners are deliberately unordered: any fixed order (top-left
first, a1 first, ...) flips discontinuously as the board rotates in the
photo, which a keypoint model would learn as averaged, wrong positions.
vision.py orders the 4 detected points geometrically.

    python -I bench/board_corners_prepare.py --annotations /data/chessred/annotations.json \
        --images-root /data/chessred --out /data/board_corners

writes images/{train,val}/<id>.jpg (longest side 640) and labels/{train,val}/<id>.txt
and dataset.yaml. Only the ChessReD2K train and val splits are used; its test
split is kept for evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")
SIZE = 640
# Box side as a fraction of one board square (mean quad side / 8): enough
# context around the corner (frame edge, first squares) to recognise it.
BOX_SQUARES = 1.0


def corner_boxes(corners: np.ndarray, w: int, h: int, box_squares: float = BOX_SQUARES) -> list[str]:
    """YOLO lines (class 0, normalised cx cy bw bh) for the 4 corners that lie
    inside the image. `corners`: (4, 2) pixel coordinates, in quad order."""
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    s = max(4.0, float(np.mean(sides)) / 8 * box_squares)
    lines = []
    for x, y in corners:
        if not (0 <= x < w and 0 <= y < h):
            continue  # corner outside the photo: nothing to see
        x0, x1 = max(0.0, x - s / 2), min(float(w), x + s / 2)
        y0, y1 = max(0.0, y - s / 2), min(float(h), y + s / 2)
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    corners = {}
    for item in data["annotations"]["corners"]:
        c = {k.replace("-", "_"): v for k, v in item["corners"].items()}
        if all(k in c for k in CORNER_KEYS):
            corners[item["image_id"]] = np.array([c[k] for k in CORNER_KEYS], float)
    splits = data["splits"]["chessred2k"]
    counts = {}
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for image_id in splits[split]["image_ids"]:
            im, c = images.get(image_id), corners.get(image_id)
            src = args.images_root / im["path"] if im else None
            if c is None or src is None or not src.is_file():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = SIZE / max(h, w)
            small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            lines = corner_boxes(c * scale, sw, sh)
            cv2.imwrite(str(args.out / "images" / split / f"{image_id}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (args.out / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
            n += 1
        counts[split] = n
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: corner\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

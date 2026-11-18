"""Prepare "set 4" (chesscog's transfer-learning sample) as a held-out test set.

Source: Georg Wolflein's chesscog (github.com/georg-wolflein/chesscog),
chesscog/transfer_learning/download_dataset.py -> Google Drive zip
1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W (~54 MB). 29 real 1200x900 photos of a
wooden Staunton set on a black/maple board, shot from a player's seat
(angled): train/{white,black}.png (starting position from each side) and
test/01..27.png (one game). Each has a JSON label:
    {"white_turn": bool, "fen": "<placement>"}
`white_turn` is both the side to move and the side the camera sits on (the
photo is taken from the mover's seat; checked by eye on every photo). There
are no corner annotations.

This set is for EVALUATION ONLY: never train on it.

    python -I bench/set4_prepare.py --raw /data/set4/raw --out /data/set4_prepared

writes
    photos/<placement>__<w|b>__<id>.png   test photos, bench/full_eval.py --photos format
                                          (w|b = side nearer the camera)
    photos/corners.json                   playing-area corners, [0,1] coords, TL TR BR BL
    photos/side_to_move.json              'w'/'b' per photo
    photos_chesscog_train/...             the 2 starting-position photos chesscog
                                          fine-tunes on (same layout)
    corner_check/<id>.jpg                 the photo with the fitted 8x8 grid drawn,
                                          for checking the corners by eye

Corners: OpenCV's checkerboard detector (findChessboardCornersSBWithMeta,
CALIB_CB_LARGER) finds a sub-grid of the board's lattice points, often
including the outer edge where the light squares meet the black frame. A
homography from lattice indices to pixels then lets us try every placement
of the 8x8 playing area that contains the detected sub-grid; the right one
has a strong light/dark checker pattern inside and the dark frame just
outside. Override any photo with --corners-override FILE.json
({"<id>": [[x, y] x4 in pixels]}).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def placement_to_name(placement: str) -> str:
    return placement.replace("/", "-")


def camera_side(label: dict) -> str:
    return "w" if label["white_turn"] else "b"


def validate_label(label: dict) -> str:
    """Return the placement field, checked: 8 ranks of 8 squares."""
    fen = str(label["fen"]).split()[0]
    ranks = fen.split("/")
    if len(ranks) != 8:
        raise ValueError(f"bad fen {fen!r}")
    for r in ranks:
        n = 0
        for ch in r:
            if ch.isdigit():
                n += int(ch)
            elif ch in "prnbqkPRNBQK":
                n += 1
            else:
                raise ValueError(f"bad fen {fen!r}")
        if n != 8:
            raise ValueError(f"bad fen {fen!r}")
    if not isinstance(label.get("white_turn"), bool):
        raise ValueError("white_turn missing")
    return fen


def order_quad(pts) -> np.ndarray:
    """4 points -> image order TL, TR, BR, BL."""
    p = np.asarray(pts, dtype=float).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    q = np.c_[pts, np.ones(len(pts))] @ h.T
    return q[:, :2] / q[:, 2:3]


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Median brightness of a 5x5 patch around each pixel point (NaN outside)."""
    h, w = gray.shape
    out = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 2 <= xi < w - 2 and 2 <= yi < h - 2:
            out.append(float(np.median(gray[yi - 2:yi + 3, xi - 2:xi + 3])))
        else:
            out.append(np.nan)
    return np.array(out)


def window_score(gray: np.ndarray, h: np.ndarray, ox: int, oy: int) -> float:
    """How well the 8x8 window with lattice origin (ox, oy) looks like the
    playing area: checker contrast inside, dark frame just outside."""
    # inside: 4 points per square near its corners (pieces stand in the middle)
    inside, parity = [], []
    for r in range(8):
        for c in range(8):
            for dx, dy in ((.2, .2), (.8, .2), (.2, .8), (.8, .8)):
                inside.append((ox + c + dx, oy + r + dy))
                parity.append((r + c) % 2)
    v = _sample(gray, _apply(h, np.array(inside)))
    parity = np.array(parity)
    a, b = np.nanmedian(v[parity == 0]), np.nanmedian(v[parity == 1])
    contrast = abs(a - b)
    # outside: a quarter square beyond each edge, at each square's middle
    ring = []
    for i in range(8):
        ring += [(ox + i + .5, oy - .25), (ox + i + .5, oy + 8.25),
                 (ox - .25, oy + i + .5), (ox + 8.25, oy + i + .5)]
    ring_v = _sample(gray, _apply(h, np.array(ring)))
    light = max(a, b)
    # share of ring points as dark as the dark squares (frame), not light
    dark_share = float(np.nanmean(ring_v < (a + b) / 2)) if np.isfinite(ring_v).any() else 0.0
    return contrast / max(light, 1.0) + dark_share


# window_score of a correct fit on the real photos: 1.72-1.88; failures < 1.1.
GOOD_FIT = 1.7


def find_corners(gray: np.ndarray):
    """Playing-area corners (pixels, TL TR BR BL) or None, plus diagnostics."""
    import cv2

    overall = None
    for larger in (True, False):
        if overall is not None and overall[0] >= GOOD_FIT:
            break
        flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        flags += cv2.CALIB_CB_LARGER if larger else 0
        for pattern in ((7, 7), (7, 5), (5, 7), (7, 3), (3, 7), (5, 5), (5, 3), (3, 5), (3, 3)):
            if overall is not None and overall[0] >= GOOD_FIT:
                break
            ok, pts, meta = cv2.findChessboardCornersSBWithMeta(gray, pattern, flags)
            if not ok:
                continue
            rows, cols = meta.shape[:2] if larger else (pattern[1], pattern[0])
            if rows > 9 or cols > 9 or rows * cols != len(pts):
                continue
            lattice = np.array([(c, r) for r in range(rows) for c in range(cols)], np.float32)
            h, _ = cv2.findHomography(lattice, pts.reshape(-1, 2).astype(np.float32), 0)
            if h is None:
                continue
            for oy in range(rows - 9, 1):
                for ox in range(cols - 9, 1):
                    s = window_score(gray, h, ox, oy)
                    if np.isfinite(s) and (overall is None or s > overall[0]):
                        overall = (s, ox, oy, h, (rows, cols), pattern, larger)
    if overall is None:
        return None, {"error": "no checkerboard found"}
    s, ox, oy, h, (rows, cols), pattern, larger = overall
    quad = _apply(h, np.array([(ox, oy), (ox + 8, oy), (ox + 8, oy + 8), (ox, oy + 8)], float))
    return order_quad(quad), {"grid": [rows, cols], "pattern": list(pattern), "larger": larger,
                              "offset": [ox, oy], "score": round(float(s), 3)}


def draw_check(img, quad):
    import cv2

    out = img.copy()
    src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)
    h = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    for i in range(9):
        for a, b in (((i, 0), (i, 8)), ((0, i), (8, i))):
            p = _apply(h, np.array([a, b], float)).astype(int)
            cv2.line(out, tuple(map(int, p[0])), tuple(map(int, p[1])), (0, 0, 255), 1, cv2.LINE_AA)
    for x, y in np.asarray(quad):
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), 2)
    return out


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, required=True, help="unzipped folder with train/ and test/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corners-override", type=Path)
    args = ap.parse_args(argv)
    override = json.loads(args.corners_override.read_text()) if args.corners_override else {}

    (args.out / "corner_check").mkdir(parents=True, exist_ok=True)
    report = {}
    for sub, dest in (("test", "photos"), ("train", "photos_chesscog_train")):
        out = args.out / dest
        out.mkdir(parents=True, exist_ok=True)
        corners, stm = {}, {}
        for js in sorted((args.raw / sub).glob("*.json")):
            img_path = js.with_suffix(".png")
            if not img_path.is_file():
                continue
            label = json.loads(js.read_text())
            fen = validate_label(label)
            ident = f"{sub}-{js.stem}"
            name = f"{placement_to_name(fen)}__{camera_side(label)}__{ident}.png"
            shutil.copyfile(img_path, out / name)
            stm[name] = "w" if label["white_turn"] else "b"
            img = cv2.imread(str(img_path))
            if img is None:
                raise SystemExit(f"unreadable {img_path}")
            if ident in override:
                quad, info = order_quad(override[ident]), {"override": True}
            else:
                quad, info = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            report[ident] = info
            if quad is None:
                continue
            h, w = img.shape[:2]
            corners[name] = [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in quad]
            cv2.imwrite(str(args.out / "corner_check" / f"{ident}.jpg"), draw_check(img, quad),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        (out / "corners.json").write_text(json.dumps(corners, indent=1))
        (out / "side_to_move.json").write_text(json.dumps(stm, indent=1))
        print(f"{dest}: {len(stm)} photos, corners for {len(corners)}")
    (args.out / "corner_report.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

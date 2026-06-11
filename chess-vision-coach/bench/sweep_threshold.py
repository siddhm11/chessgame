"""One-off experiment: sweep the model confidence threshold and also dump
warped board crops for visual alignment inspection. Not part of the test
suite -- run manually, read the table, then delete or keep under bench/.

  python bench/sweep_threshold.py --n 40
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_vision import vision
from chess_vision.model_backend import (
    _PIECE_NAME_TO_SYMBOL,
    _FALLBACK_CLASS_TO_SYMBOL,
)

DATASET = Path("/tmp/chess-dataset/labeled_originals")
PLACEMENT_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})\.")
NMS_THRESHOLD = 0.45
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def filename_to_grid(name: str):
    m = PLACEMENT_RE.match(name)
    if not m:
        return None
    placement = m.group(1)
    rows = placement.split("-")
    if len(rows) != 8:
        return None
    grid = []
    for r in rows:
        row = []
        for ch in r:
            if ch.isdigit():
                row.extend(["."] * int(ch))
            else:
                row.append(ch)
        if len(row) != 8:
            return None
        grid.append(row)
    return grid


def load_model(model_path: Path):
    import onnxruntime as ort
    import ast, json

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    meta = sess.get_modelmeta().custom_metadata_map
    try:
        imgsz = int(json.loads(meta.get("imgsz", "[640,640]"))[0])
    except Exception:
        imgsz = int(sess.get_inputs()[0].shape[2])
    try:
        names = ast.literal_eval(meta.get("names", "{}"))
        cls_map = {
            int(k): _PIECE_NAME_TO_SYMBOL[v]
            for k, v in names.items()
            if v in _PIECE_NAME_TO_SYMBOL
        }
    except Exception:
        cls_map = _FALLBACK_CLASS_TO_SYMBOL
    return sess, input_name, imgsz, cls_map


def infer_raw(sess, input_name, imgsz, board_bgr):
    """Return (boxes_xywh_norm, class_ids, confidences) with no threshold."""
    net = cv2.resize(board_bgr, (imgsz, imgsz))
    net = cv2.cvtColor(net, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net = np.transpose(net, (2, 0, 1))[None]
    out = sess.run(None, {input_name: net})[0]
    preds = out[0].T
    boxes = preds[:, :4].astype(np.float32)
    scores = preds[:, 4:].astype(np.float32)
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    if len(boxes) and boxes[:, :2].max() > 1.5:
        boxes = boxes / imgsz
    return boxes, class_ids, confs


def grid_from_dets(boxes, class_ids, confs, cls_map, thr):
    keep = confs >= thr
    b, ci, cf = boxes[keep], class_ids[keep], confs[keep]
    grid = [["." for _ in range(8)] for _ in range(8)]
    cell_conf = [[0.0] * 8 for _ in range(8)]
    if len(b) == 0:
        return grid
    tl = b.copy()
    tl[:, 0] -= tl[:, 2] / 2
    tl[:, 1] -= tl[:, 3] / 2
    nms = cv2.dnn.NMSBoxes(tl.tolist(), cf.tolist(), thr, NMS_THRESHOLD)
    if len(nms) == 0:
        return grid
    for i in np.asarray(nms).flatten():
        sym = cls_map.get(int(ci[i]))
        if sym is None:
            continue
        col = int(np.clip(b[i, 0] * 8, 0, 7))
        row = int(np.clip(b[i, 1] * 8, 0, 7))
        if cf[i] > cell_conf[row][col]:
            grid[row][col] = sym
            cell_conf[row][col] = float(cf[i])
    return grid


def score(pred, truth):
    m = defaultdict(int)
    for r in range(8):
        for c in range(8):
            p, t = pred[r][c], truth[r][c]
            m["total"] += 1
            if p == t:
                m["exact"] += 1
            if t != ".":
                m["truth_piece"] += 1
                if p != ".":
                    m["recall"] += 1
                    if p == t:
                        m["type_ok"] += 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dump", type=int, default=0, help="dump N warped boards to /tmp")
    args = ap.parse_args()

    model_path = Path(__file__).resolve().parent.parent / "chess_vision" / "models" / "yolov8m-chess.onnx"
    sess, input_name, imgsz, cls_map = load_model(model_path)
    finder = vision.ClassicalBackend()

    files = sorted(DATASET.glob("*.JPG"))[: args.n]
    print(f"{len(files)} photos, imgsz={imgsz}, model={model_path.name}\n")

    # Cache raw detections once per photo (inference is the expensive part).
    cache = []
    for i, fp in enumerate(files, 1):
        truth = filename_to_grid(fp.name)
        if truth is None:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        board = finder._find_board(img)
        if args.dump and i <= args.dump:
            cv2.imwrite(f"/tmp/warp_{i:02d}.png", cv2.resize(board, (480, 480)))
        boxes, cids, confs = infer_raw(sess, input_name, imgsz, board)
        cache.append((truth, boxes, cids, confs))
        print(f"  cached {i}/{len(files)}", end="\r")
    print()

    header = f"{'thr':>5} {'exact':>8} {'recall':>8} {'type|p':>8}"
    print(header)
    print("-" * len(header))
    for thr in THRESHOLDS:
        agg = defaultdict(int)
        for truth, boxes, cids, confs in cache:
            pred = grid_from_dets(boxes, cids, confs, cls_map, thr)
            for k, v in score(pred, truth).items():
                agg[k] += v
        exact = agg["exact"] / max(1, agg["total"])
        recall = agg["recall"] / max(1, agg["truth_piece"])
        type_ok = agg["type_ok"] / max(1, agg["recall"])
        print(f"{thr:>5.2f} {exact:>8.1%} {recall:>8.1%} {type_ok:>8.1%}")


if __name__ == "__main__":
    main()

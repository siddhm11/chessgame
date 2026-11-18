"""Learned board-corner detector: one more candidate for the board finder.

A small YOLO detector (one class, "corner", trained on ChessReD2K's annotated
board corners -- see bench/board_corners_prepare.py) proposes the 4 corners of
the playing area. ClassicalBackend.locate_board snaps that quad onto the
8x8 grid with _refine_quad and judges it with the same sharp grid fit as
every other candidate, so the detector can only ADD a candidate: it never
overrides the fit judge. If the ONNX file is absent (or CVC_CORNER_MODEL=0)
nothing changes.

Corners are detected unordered (any fixed order flips discontinuously as the
board rotates in the photo); the caller orders them geometrically.

Runtime: onnxruntime when installed, else OpenCV's DNN module (the in-browser
build has no onnxruntime) -- the same pattern as model_backend.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "board-corners.onnx"
IMGSZ = 640
CONF = 0.25
NMS_IOU = 0.5

_lock = threading.Lock()
_runner = None  # (callable blob -> raw output) once loaded; False if unavailable


def _model_path() -> Path | None:
    env = os.environ.get("CVC_CORNER_MODEL")
    if env == "0":
        return None
    p = Path(env) if env else MODEL_PATH
    return p if p.is_file() else None


def _load():
    global _runner
    if _runner is not None:
        return _runner
    with _lock:
        if _runner is not None:
            return _runner
        path = _model_path()
        if path is None:
            _runner = False
            return _runner
        runtime = os.environ.get("CVC_RUNTIME", "auto")
        ort = None
        if runtime != "opencv":
            try:
                import onnxruntime as ort
            except ImportError:
                ort = None
        if ort is not None:
            from chess_vision.vision import THREADS

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = THREADS
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(path), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            _runner = lambda blob: sess.run(None, {name: blob})[0]  # noqa: E731
        else:
            net = cv2.dnn.readNetFromONNX(str(path))
            net_lock = threading.Lock()  # a cv2.dnn.Net is not thread-safe

            def run(blob):
                with net_lock:
                    net.setInput(np.ascontiguousarray(blob, dtype=np.float32))
                    return net.forward()
            _runner = run
        return _runner


def available() -> bool:
    return bool(_load())


def _letterbox(img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = IMGSZ / max(h, w)
    nw, nh = round(w * r), round(h * r)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = small
    return canvas, r, px, py


def corner_points(raw: np.ndarray, r: float, px: int, py: int,
                  conf: float = CONF) -> tuple[np.ndarray, np.ndarray]:
    """Decode YOLO output [1, 5, N] (cx, cy, w, h, score; input pixels) to
    corner centres in original-image pixels, after NMS. Returns (points, scores),
    highest score first."""
    preds = raw[0].T if raw.ndim == 3 else raw.T
    scores = preds[:, 4]
    keep = scores >= conf
    preds, scores = preds[keep], scores[keep]
    if len(preds) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    boxes = preds[:, :4].copy()
    tl = boxes.copy()
    tl[:, 0] -= tl[:, 2] / 2
    tl[:, 1] -= tl[:, 3] / 2
    idx = cv2.dnn.NMSBoxes(tl.tolist(), scores.tolist(), conf, NMS_IOU)
    idx = np.asarray(idx).flatten()
    idx = idx[np.argsort(-scores[idx])]
    pts = np.stack([(boxes[idx, 0] - px) / r, (boxes[idx, 1] - py) / r], axis=1)
    return pts, scores[idx]


def plausible_quad(pts: np.ndarray, shape) -> bool:
    """4 points forming a convex quad of reasonable size."""
    if len(pts) != 4:
        return False
    hull = cv2.convexHull(pts.astype(np.float32))
    if len(hull) != 4:
        return False
    h, w = shape[:2]
    return cv2.contourArea(hull) >= 0.02 * w * h


def predict_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """4 board corners (pixels, unordered, shape (4, 1, 2)) or None."""
    run = _load()
    if not run:
        return None
    canvas, r, px, py = _letterbox(img_bgr)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    try:
        pts, _ = corner_points(run(blob), r, px, py)
    except cv2.error:
        return None
    pts = pts[:4]
    if not plausible_quad(pts, img_bgr.shape):
        return None
    return pts.reshape(4, 1, 2).astype("float32")

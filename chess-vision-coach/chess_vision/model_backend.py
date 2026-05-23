"""YOLO11n ONNX piece detector as a VisionBackend.

The model card (yamero999/chess-piece-detection-yolo11n) ships an ONNX
export of an Ultralytics YOLO11n detect head trained on 416x416 board
crops. We:

  1. Reuse the classical geometric board-finder to crop & warp the board
     to a top-down square (BOARD_PX). This is robust on synthetic + real
     photos and decouples board localization from piece classification.
  2. Resize the warped board to the model's expected 416x416 input.
  3. Run ONNX inference -- output shape [1, 16, N] where 16 = 4 box +
     12 class scores, N ~= 3549 anchors at this input size.
  4. Argmax class per anchor, confidence-filter, NMS via cv2.
  5. Each surviving detection's box center maps directly to one of the
     64 grid cells; per-cell we keep the highest-confidence piece.

Loading the ONNX session is lazy -- a synthetic test that never calls
analyze() never pays for it. The .onnx file is expected to live next
to this module at models/yolo11n-chess.onnx, or pointed to by the
CVC_MODEL_PATH env var.

Licensing note: the weights are Ultralytics YOLO11 derivatives and
inherit AGPL-3.0 (per the ONNX metadata), independent of the dataset
license. Fine for personal/learning use; for commercial distribution
you'd need an Ultralytics commercial license or a non-YOLO model.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from chess_vision.vision import BackendOutput, ClassicalBackend

# Class ID -> FEN piece symbol. Read from the ONNX metadata 'names' field
# of yamero999/chess-piece-detection-yolo11n on Hugging Face.
_CLASS_TO_SYMBOL: dict[int, str] = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
}

# Model expects 416x416 RGB input (from ONNX metadata 'imgsz').
_MODEL_IMGSZ = 416

# Minimum class score to consider a detection at all.
_CONF_THRESHOLD = 0.25
# IoU threshold for non-maximum suppression.
_NMS_THRESHOLD = 0.45

# Confidence reported for cells where we found no detection. The classical
# backend has a real "is empty" signal; the detector doesn't, so we default
# to a moderately high value -- not so high that misses hide, but high
# enough that empty cells don't all light up red in the confidence overlay.
_EMPTY_CONF = 0.90


class ModelBackend:
    """YOLO11n ONNX piece detector."""

    name = "yolo11n-onnx"

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        if model_path is None:
            model_path = os.environ.get("CVC_MODEL_PATH")
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "yolo11n-chess.onnx"
        self.model_path = Path(model_path)
        self._session = None  # onnxruntime.InferenceSession, lazy
        self._input_name: str | None = None
        # Board-finding (geometric) is shared logic; reuse the classical
        # backend's implementation. Templates won't be built unless used.
        self._board_finder = ClassicalBackend()

    def _load(self):
        if self._session is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {self.model_path}. "
                    "Vendor the .onnx file or set CVC_MODEL_PATH."
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for ModelBackend. "
                    "Install with: pip install onnxruntime"
                ) from exc
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        return self._session

    def analyze(self, image_bgr: np.ndarray) -> BackendOutput:
        # 1. Geometric board crop + warp.
        board_bgr = self._board_finder._find_board(image_bgr)

        # 2. Resize to model input, BGR -> RGB, normalize, NCHW.
        net = cv2.resize(board_bgr, (_MODEL_IMGSZ, _MODEL_IMGSZ))
        net = cv2.cvtColor(net, cv2.COLOR_BGR2RGB)
        net = net.astype(np.float32) / 255.0
        net = np.transpose(net, (2, 0, 1))[None, :, :, :]

        # 3. Inference.
        session = self._load()
        out = session.run(None, {self._input_name: net})[0]  # [1, 16, N]

        # 4. Decode: [1, 16, N] -> [N, 16]; 4 box + 12 class.
        preds = out[0].T
        boxes_xywh = preds[:, :4].astype(np.float32)
        class_scores = preds[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # 5. Threshold + NMS.
        keep_mask = confidences >= _CONF_THRESHOLD
        boxes_xywh = boxes_xywh[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        kept_indices: np.ndarray = np.array([], dtype=int)
        if len(boxes_xywh) > 0:
            # cv2.dnn.NMSBoxes expects (x_top, y_top, w, h).
            boxes_xywh_tl = boxes_xywh.copy()
            boxes_xywh_tl[:, 0] -= boxes_xywh_tl[:, 2] / 2.0
            boxes_xywh_tl[:, 1] -= boxes_xywh_tl[:, 3] / 2.0
            nms = cv2.dnn.NMSBoxes(
                boxes_xywh_tl.tolist(),
                confidences.tolist(),
                _CONF_THRESHOLD,
                _NMS_THRESHOLD,
            )
            if len(nms) > 0:
                kept_indices = np.asarray(nms).flatten()

        # 6. Map each surviving detection's center to a grid cell; keep the
        #    highest-confidence piece per cell.
        grid: list[list[str]] = [["." for _ in range(8)] for _ in range(8)]
        conf: list[list[float]] = [[0.0 for _ in range(8)] for _ in range(8)]
        for i in kept_indices:
            cx, cy = float(boxes_xywh[i, 0]), float(boxes_xywh[i, 1])
            col = int(np.clip(cx * 8.0 / _MODEL_IMGSZ, 0, 7))
            row = int(np.clip(cy * 8.0 / _MODEL_IMGSZ, 0, 7))
            c = float(confidences[i])
            if c > conf[row][col]:
                grid[row][col] = _CLASS_TO_SYMBOL.get(int(class_ids[i]), ".")
                conf[row][col] = c

        # 7. Empty squares: assign a default "we think this is empty" score
        #    so the confidence overlay doesn't paint the whole board red.
        for r in range(8):
            for c_ in range(8):
                if grid[r][c_] == ".":
                    conf[r][c_] = _EMPTY_CONF

        return BackendOutput(grid=grid, conf=conf, board_found=True)

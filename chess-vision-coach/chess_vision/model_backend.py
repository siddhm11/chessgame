"""YOLO ONNX piece detector as a VisionBackend.

Auto-configures input size and class mapping from the ONNX model metadata,
so any Ultralytics YOLO export (YOLO11n, YOLOv8m, …) can be used by
dropping the .onnx file into chess_vision/models/ without changing code.

Pipeline:
  1. Reuse the classical geometric board-finder to crop & warp the board.
  2. Resize the warped board to the model's required input size (read from
     the ONNX metadata 'imgsz' field, e.g. 416 or 640).
  3. Run ONNX inference — output [1, 4+C, N] where C = number of classes.
  4. Argmax class per anchor, confidence-filter, NMS.
  5. Map each box center to one of the 64 grid cells.

'board' detections (present in some models as class 0) are silently
ignored — only piece-class detections populate the grid.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import cv2
import numpy as np

from chess_vision.vision import BackendOutput, ClassicalBackend

# Canonical piece name -> FEN symbol (covers both model naming conventions).
_PIECE_NAME_TO_SYMBOL: dict[str, str] = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k",
}

# Fallback class map used when model metadata is missing (yolo11n order).
_FALLBACK_CLASS_TO_SYMBOL: dict[int, str] = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
}

_CONF_THRESHOLD = 0.25
_NMS_THRESHOLD = 0.45
_EMPTY_CONF = 0.90


class ModelBackend:
    """YOLO ONNX piece detector. Auto-configures from model metadata."""

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        if model_path is None:
            model_path = os.environ.get("CVC_MODEL_PATH")
        if model_path is None:
            models_dir = Path(__file__).resolve().parent / "models"
            # Prefer the fine-tuned model (trained on real tournament photos,
            # 98% per-square accuracy on held-out val); fall back to the
            # pre-trained models if it is absent.
            for candidate in (
                "yolov8n-chess-finetuned.onnx",
                "yolov8m-chess.onnx",
                "yolo11n-chess.onnx",
            ):
                p = models_dir / candidate
                if p.exists():
                    model_path = p
                    break
            else:
                model_path = models_dir / "yolov8n-chess-finetuned.onnx"
        self.model_path = Path(model_path)
        self.name = self.model_path.stem
        self._session = None
        self._input_name: str | None = None
        self._imgsz: int = 416
        self._class_to_symbol: dict[int, str] = _FALLBACK_CLASS_TO_SYMBOL
        self._board_finder = ClassicalBackend()

    def _load(self):
        if self._session is not None:
            return self._session
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

        meta = self._session.get_modelmeta().custom_metadata_map

        # Input size from metadata (falls back to ONNX input shape).
        try:
            imgsz = json.loads(meta.get("imgsz", "[416, 416]"))
            self._imgsz = int(imgsz[0])
        except Exception:
            self._imgsz = int(self._session.get_inputs()[0].shape[2])

        # Class mapping from metadata names dict.
        try:
            names: dict = ast.literal_eval(meta.get("names", "{}"))
            mapping = {
                int(k): _PIECE_NAME_TO_SYMBOL[v]
                for k, v in names.items()
                if v in _PIECE_NAME_TO_SYMBOL
            }
            if mapping:
                self._class_to_symbol = mapping
        except Exception:
            pass  # keep fallback

        return self._session

    def analyze(self, image_bgr: np.ndarray) -> BackendOutput:
        session = self._load()
        imgsz = self._imgsz

        # 1. Geometric board crop + warp.
        board_bgr = self._board_finder._find_board(image_bgr)

        # 2. Resize to model input, BGR -> RGB, normalize, NCHW.
        net = cv2.resize(board_bgr, (imgsz, imgsz))
        net = cv2.cvtColor(net, cv2.COLOR_BGR2RGB)
        net = net.astype(np.float32) / 255.0
        net = np.transpose(net, (2, 0, 1))[None]

        # 3. Inference.
        out = session.run(None, {self._input_name: net})[0]  # [1, 4+C, N]

        # 4. Decode: [1, 4+C, N] -> [N, 4+C].
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
            # Normalize to [0,1] before NMS so IoU works regardless of export format.
            if boxes_xywh[:, :2].max() > 1.5:
                boxes_xywh = boxes_xywh / imgsz
            boxes_tl = boxes_xywh.copy()
            boxes_tl[:, 0] -= boxes_tl[:, 2] / 2.0
            boxes_tl[:, 1] -= boxes_tl[:, 3] / 2.0
            nms = cv2.dnn.NMSBoxes(
                boxes_tl.tolist(), confidences.tolist(),
                _CONF_THRESHOLD, _NMS_THRESHOLD,
            )
            if len(nms) > 0:
                kept_indices = np.asarray(nms).flatten()

        # 6. Map box centers to 8×8 grid; keep highest-confidence per cell.
        grid: list[list[str]] = [["." for _ in range(8)] for _ in range(8)]
        conf: list[list[float]] = [[0.0 for _ in range(8)] for _ in range(8)]
        for i in kept_indices:
            symbol = self._class_to_symbol.get(int(class_ids[i]))
            if symbol is None:
                continue  # skip 'board' class and unknown IDs
            cx, cy = float(boxes_xywh[i, 0]), float(boxes_xywh[i, 1])
            col = int(np.clip(cx * 8.0, 0, 7))
            row = int(np.clip(cy * 8.0, 0, 7))
            c = float(confidences[i])
            if c > conf[row][col]:
                grid[row][col] = symbol
                conf[row][col] = c

        # 7. Empty squares get a default "we think this is empty" confidence.
        for r in range(8):
            for c_ in range(8):
                if grid[r][c_] == ".":
                    conf[r][c_] = _EMPTY_CONF

        return BackendOutput(grid=grid, conf=conf, board_found=True)

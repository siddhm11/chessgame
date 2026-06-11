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

Orientation recovery (geometric):
  After warping, the top-left corner of the board is a8 (always LIGHT in
  standard chess). If a8 appears DARK in the warped crop, the board was
  photographed 90° rotated. We correct for this by trying 90°CW and
  90°CCW rotations and selecting the one where both kings are found; this
  is a fast, YOLO-free pre-check that avoids incorrect rotation on dense
  positions where a king might be misclassified.

Sparse-position recovery:
  If fewer than 2 pieces survive the normal confidence threshold, we
  re-decode the same raw predictions at a lower threshold. This avoids a
  second ONNX inference while recovering genuine low-confidence detections
  in near-empty endgame positions.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import cv2
import numpy as np

from chess_vision.sanity import sanitize
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
_SPARSE_THRESHOLD = 0.12   # used only when < 2 pieces survive the normal threshold
_NMS_THRESHOLD = 0.45
_EMPTY_CONF = 0.90

# Pixel border width used when sampling cell corner brightness.
_CORNER_PX = 8


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

    @staticmethod
    def _cell_corner_brightness(gray: np.ndarray, row: int, col: int) -> float:
        """Median brightness of the 4 corner patches of cell (row, col)."""
        p = _CORNER_PX
        CELL = 80
        cell = gray[row * CELL : (row + 1) * CELL, col * CELL : (col + 1) * CELL]
        return float(np.median(np.concatenate([
            cell[:p, :p].ravel(),
            cell[:p, -p:].ravel(),
            cell[-p:, :p].ravel(),
            cell[-p:, -p:].ravel(),
        ])))

    @staticmethod
    def _board_needs_rotation(board_bgr: np.ndarray) -> bool:
        """Return True when the board crop appears 90°-rotated.

        In standard orientation (white at bottom) file-a squares alternate:
        a8 (row 0, col 0) is LIGHT and a1 (row 7, col 0) is DARK, so
        brightness(a8) > brightness(a1). If a8 is darker than a1 the board
        was photographed 90°-rotated and the warp preserved that rotation.
        Comparing the same file avoids the false positives that arise when
        both top-corner cells happen to be light (unusual board colours).
        """
        gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        tl = ModelBackend._cell_corner_brightness(gray, 0, 0)  # a8: should be light
        bl = ModelBackend._cell_corner_brightness(gray, 7, 0)  # a1: should be dark
        return tl < bl  # a8 darker than a1 → board is 90°-rotated

    def _decode_predictions(
        self, preds: np.ndarray, threshold: float
    ) -> tuple[list[list[str]], list[list[float]], list[list[list[tuple[str, float]] | None]]]:
        """Decode a raw [N, 4+C] prediction array into an 8×8 grid.

        Applies confidence thresholding and NMS; boxes normalised to [0,1].
        Also returns per-cell ranked alternative classes (from the winning
        anchor's class-score vector) for the sanity layer's auto-repair.
        """
        imgsz = self._imgsz
        boxes_xywh = preds[:, :4].astype(np.float32)
        class_scores = preds[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep_mask = confidences >= threshold
        boxes_xywh = boxes_xywh[keep_mask]
        class_scores = class_scores[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        kept_indices: np.ndarray = np.array([], dtype=int)
        if len(boxes_xywh) > 0:
            if boxes_xywh[:, :2].max() > 1.5:
                boxes_xywh = boxes_xywh / imgsz
            boxes_tl = boxes_xywh.copy()
            boxes_tl[:, 0] -= boxes_tl[:, 2] / 2.0
            boxes_tl[:, 1] -= boxes_tl[:, 3] / 2.0
            nms = cv2.dnn.NMSBoxes(
                boxes_tl.tolist(), confidences.tolist(),
                threshold, _NMS_THRESHOLD,
            )
            if len(nms) > 0:
                kept_indices = np.asarray(nms).flatten()

        grid: list[list[str]] = [["." for _ in range(8)] for _ in range(8)]
        conf: list[list[float]] = [[0.0 for _ in range(8)] for _ in range(8)]
        alts: list[list[list[tuple[str, float]] | None]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
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
                # Ranked runner-up classes of the winning anchor, for the
                # sanity layer's "this symbol is impossible" auto-repair.
                scores_i = class_scores[i]
                order = np.argsort(scores_i)[::-1]
                ranked = []
                for cls_idx in order:
                    alt_sym = self._class_to_symbol.get(int(cls_idx))
                    if alt_sym is None or alt_sym == symbol:
                        continue
                    if scores_i[cls_idx] < 0.01 or len(ranked) >= 3:
                        break
                    ranked.append((alt_sym, float(scores_i[cls_idx])))
                alts[row][col] = ranked or None

        for r in range(8):
            for c_ in range(8):
                if grid[r][c_] == ".":
                    conf[r][c_] = _EMPTY_CONF

        return grid, conf, alts

    def _infer_board(
        self, board_bgr: np.ndarray
    ) -> tuple[list[list[str]], list[list[float]], list]:
        """Run ONNX inference on a pre-cropped board image.

        Includes sparse-position recovery: if fewer than 2 pieces survive
        the normal threshold, reprocess the same predictions at a lower one.
        """
        imgsz = self._imgsz
        net = cv2.resize(board_bgr, (imgsz, imgsz))
        net = cv2.cvtColor(net, cv2.COLOR_BGR2RGB)
        net = net.astype(np.float32) / 255.0
        net = np.transpose(net, (2, 0, 1))[None]

        raw = self._session.run(None, {self._input_name: net})[0]  # [1, 4+C, N]
        preds = raw[0].T  # [N, 4+C]

        grid, conf, alts = self._decode_predictions(preds, _CONF_THRESHOLD)

        # Sparse-position recovery: revisit raw predictions at a lower bar.
        piece_count = sum(1 for row in grid for sq in row if sq != ".")
        if piece_count < 2:
            grid, conf, alts = self._decode_predictions(preds, _SPARSE_THRESHOLD)

        return grid, conf, alts

    def analyze(self, image_bgr: np.ndarray) -> BackendOutput:
        self._load()

        # 1. Geometric board crop + warp.
        board_bgr = self._board_finder._find_board(image_bgr)

        # 2. Geometric orientation check: if the board crop is 90°-rotated
        #    (detectable from a8/h8 corner brightness), correct it before
        #    running inference. This avoids adding rotation overhead to the
        #    ~97% of images that are already correctly oriented.
        if self._board_needs_rotation(board_bgr):
            best: tuple | None = None
            for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                r_board = cv2.rotate(board_bgr, rot_code)
                r_grid, r_conf, r_alts = self._infer_board(r_board)
                r_flat = [sq for row in r_grid for sq in row]
                if "K" in r_flat and "k" in r_flat:
                    # Both kings present — this rotation is correct.
                    return self._finalize(r_grid, r_conf, r_alts)
                if best is None:
                    best = (r_grid, r_conf, r_alts)
            # Neither rotation yielded both kings; use whichever was tried first.
            if best is not None:
                return self._finalize(*best)

        # 3. Standard inference on the (already correctly oriented) crop.
        grid, conf, alts = self._infer_board(board_bgr)
        return self._finalize(grid, conf, alts)

    @staticmethod
    def _finalize(grid, conf, alts) -> BackendOutput:
        """Apply the chess-logic sanity layer before handing the grid out.

        Impossible cells (back-rank pawns, duplicate kings, …) are repaired
        from the model's runner-up classes or confidence-flagged for the UI.
        """
        grid, conf, _notes = sanitize(grid, conf, alts)
        return BackendOutput(grid=grid, conf=conf, board_found=True)

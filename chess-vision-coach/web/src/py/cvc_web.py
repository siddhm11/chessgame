"""Glue between the browser page and the chess_vision pipeline (Pyodide).

The page talks to this module through a Web Worker. Every entry point takes
and returns plain JSON-able values. The vision code itself is the same
package the server runs; only the model runtime differs:

* With JSPI (Chrome/Edge 137+): the ONNX model runs in onnxruntime-web -- the
  same engine as the server, several times faster than the fallback -- and
  Python waits for it with `pyodide.ffi.run_sync`.
* Elsewhere: OpenCV's DNN module runs the same graph inside Python
  (identical results on the 112 held-out photos; slower).
"""

from __future__ import annotations

import json
import os

import chess
import chess.svg
import numpy as np

os.environ.setdefault("CVC_BACKEND", "model")
os.environ.setdefault("CVC_RUNTIME", "opencv")

from chess_vision import render, vision  # noqa: E402
from chess_vision.model_backend import ModelBackend  # noqa: E402
from chess_vision.position import confidence_fill, fen_problem  # noqa: E402

PIECE_CODES = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}
BOARD_SIZE = 480

_backend: ModelBackend | None = None
_image: np.ndarray | None = None   # the current upload, decoded
runtime_name = "opencv"


class _JsSession:
    """onnxruntime-web behind the InferenceSession.run() slice ModelBackend
    uses. `js_run(Float32Array) -> Promise<Float32Array>` is provided by the
    worker; run_sync suspends Python (JSPI) until the promise settles."""

    input_name = "images"

    def __init__(self, js_run, output_shape) -> None:
        self._js_run = js_run
        self._shape = tuple(output_shape)

    def run(self, _outputs, feeds: dict):
        from pyodide.ffi import run_sync, to_js

        (blob,) = feeds.values()
        flat = np.ascontiguousarray(blob, dtype=np.float32).ravel()
        result = run_sync(self._js_run(to_js(memoryview(flat))))
        out = np.frombuffer(result.to_py(), dtype=np.float32).copy()
        return [out.reshape(self._shape)]


def init(model_path: str, js_run=None, output_shape=None) -> str:
    """Load the model. With `js_run`, inference goes to onnxruntime-web."""
    global _backend, runtime_name
    backend = ModelBackend(model_path)
    backend._load()   # OpenCV session + imgsz/class map from the ONNX metadata
    if js_run is not None:
        backend._session = _JsSession(js_run, list(output_shape))
        runtime_name = "onnxruntime-web"
    _backend = backend
    return runtime_name


def _board_svg(fen: str, psc: dict | None, flipped: bool, arrows=()) -> str:
    board = chess.Board(fen) if len(fen.split()) > 1 else chess.Board(f"{fen} w - - 0 1")
    return chess.svg.board(
        board, size=BOARD_SIZE, coordinates=True, flipped=flipped,
        arrows=list(arrows), fill=confidence_fill(psc),
    )


def view(fen: str, psc_json: str | None = None, flipped: bool = False,
         arrow_uci: str | None = None) -> str:
    """Board SVG + legality check for the current position."""
    psc = json.loads(psc_json) if psc_json else None
    arrows = []
    if arrow_uci:
        mv = chess.Move.from_uci(arrow_uci)
        arrows = [chess.svg.Arrow(mv.from_square, mv.to_square, color="#2e7d32")]
    return json.dumps({
        "svg": _board_svg(fen, psc, flipped, arrows),
        "problem": fen_problem(fen),
        "low_conf_count": len(confidence_fill(psc)),
    })


def analyze(orientation: str, side_to_move: str, corners_json: str | None) -> str:
    """Read the photo loaded by load_image()."""
    if _image is None:
        return json.dumps({"error": "Please choose a chessboard photo to analyze."})
    img = _image
    corners = json.loads(corners_json) if corners_json else None
    stm = "b" if side_to_move == "black" else "w"
    result = vision.analyze_image(
        img, orientation="black" if orientation == "black" else "white",
        side_to_move=stm, backend=_backend, corners=corners,
    )
    return json.dumps({
        "fen": result.fen,
        "detection_failed": result.detection_failed,
        "confidence": result.confidence,
        "per_square_confidence": result.per_square_confidence,
        "notes": list(result.notes or []),
        "corners": result.board_corners or corners,
    })


def load_image(path: str) -> str:
    """Decode an upload once (EXIF-upright, size-capped) and keep it for
    analyze() and display_jpeg(). Bad uploads fail here with a clear message."""
    global _image
    _image = None
    try:
        img = vision.decode_image_bytes(open(path, "rb").read())
    except vision.ImageTooLarge as exc:
        return json.dumps({"error": str(exc)})
    if img is None:
        return json.dumps({"error": "That image could not be read -- it may be corrupt or not a real image."})
    _image = img
    return json.dumps({"ok": True, "size": [int(img.shape[1]), int(img.shape[0])]})


def display_jpeg() -> bytes:
    """The upright, size-capped image the pipeline analyses, as JPEG bytes
    (board corners are fractions of exactly this image)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", _image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else b""


def correct(fen: str, square: str, piece: str) -> str:
    parts = fen.split()
    target = chess.parse_square(square.strip().lower())
    base = chess.BaseBoard(parts[0])
    if piece == "empty":
        base.remove_piece_at(target)
    elif piece in PIECE_CODES:
        base.set_piece_at(target, chess.Piece.from_symbol(PIECE_CODES[piece]))
    parts[0] = base.board_fen()
    return " ".join(parts)


def set_side_to_move(fen: str, side: str) -> str:
    parts = fen.split()
    parts[1] = "b" if side == "black" else "w"
    return " ".join(parts)


def play(fen: str, uci: str) -> str | None:
    board = chess.Board(fen)
    mv = chess.Move.from_uci(uci)
    if mv not in board.legal_moves:
        return None
    board.push(mv)
    return board.fen()


def pv_to_san(fen: str, uci_moves: list) -> str:
    """'e2e4 e7e5 ...' -> '1. e4 e5 ...' (stops at the first illegal move)."""
    board = chess.Board(fen)
    moves = []
    for u in uci_moves:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break
        if mv not in board.legal_moves:
            break
        moves.append(mv)
        board.push(mv)
    return chess.Board(fen).variation_san(moves) if moves else ""


def game_over(fen: str) -> str | None:
    board = chess.Board(fen)
    if board.is_checkmate():
        return "Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw by insufficient material"
    return None

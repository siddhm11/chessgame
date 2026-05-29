"""Tests for the YOLO11n ONNX model backend.

Skipped automatically if the .onnx file has not been vendored at
chess_vision/models/yolo11n-chess.onnx or if onnxruntime is not
installed -- the classical backend remains the default and is the only
one required by CI.
"""

from pathlib import Path

import cv2
import pytest

from chess_vision import vision

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "chess_vision" / "models" / "yolo11n-chess.onnx"
)


def _have_model() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return MODEL_PATH.exists()


pytestmark = pytest.mark.skipif(
    not _have_model(),
    reason="yolo11n-chess.onnx or onnxruntime not available",
)


def _backend():
    # Pin to yolo11n explicitly: this suite verifies ONNX-backend mechanics
    # on the synthetic fixtures, and yolo11n parses rendered diagrams. The
    # default model is now yolov8n-chess-finetuned, which is specialized for
    # REAL tournament photos and does not recognize flat SVG-rendered pieces.
    from chess_vision.model_backend import ModelBackend
    return ModelBackend(model_path=MODEL_PATH)


def test_model_backend_loads_and_returns_grid_shape():
    backend = _backend()
    img = cv2.imread(str(FIXTURES / "starting.png"))
    out = backend.analyze(img)
    assert len(out.grid) == 8
    assert all(len(row) == 8 for row in out.grid)
    assert len(out.conf) == 8
    assert all(len(row) == 8 for row in out.conf)
    assert out.board_found is True


def test_model_backend_detects_kings_on_starting_position():
    """Sanity floor: on the starting fixture both kings must be detected.
    If even kings are missing, something is wired up wrong, not just noisy."""
    backend = _backend()
    img = cv2.imread(str(FIXTURES / "starting.png"))
    out = backend.analyze(img)
    flat = [sq for row in out.grid for sq in row]
    assert "K" in flat, f"white king not detected; grid={out.grid}"
    assert "k" in flat, f"black king not detected; grid={out.grid}"


def test_model_backend_via_analyze_image_env(monkeypatch):
    """End-to-end: setting CVC_BACKEND=model routes analyze_image through
    the ONNX backend and produces a valid (non-fallback) result."""
    monkeypatch.setenv("CVC_BACKEND", "model")
    # Route to yolo11n (parses synthetic renders); the finetuned default is
    # for real photos only. See _backend() above.
    monkeypatch.setenv("CVC_MODEL_PATH", str(MODEL_PATH))
    monkeypatch.setattr(vision, "_DEFAULT_BACKEND", None)

    result = vision.analyze_image(
        str(FIXTURES / "starting.png"),
        orientation="white",
        side_to_move="w",
    )
    assert not result.detection_failed, "model should not trigger fallback on starting position"
    placement = result.fen.split()[0]
    ranks = placement.split("/")
    assert "k" in ranks[0]  # black king on rank 8
    assert "K" in ranks[7]  # white king on rank 1

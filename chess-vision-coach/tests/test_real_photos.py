"""End-to-end tests against real tournament photos.

These tests use cropped-and-resized photos from the held-out validation
split of the samryan18/chess-dataset. The model is the fine-tuned
YOLOv8n (CVC_BACKEND=model); each test asserts a minimum per-square
accuracy rather than an exact FEN, so minor model updates that keep
accuracy high don't require fixture updates.

The tests are skipped automatically if:
  - onnxruntime is not installed, or
  - the fine-tuned .onnx model is not present.

All three fixtures cover different game phases and piece densities:
  opening_real.jpg   — 32 pieces, Sicilian-like opening
  midgame_real.jpg   — 17 pieces, complex middle game
  endgame_real.jpg   — 16 pieces, endgame with bishops and knights
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "real"
FINETUNED_MODEL = (
    Path(__file__).resolve().parent.parent
    / "chess_vision" / "models" / "yolov8n-chess-finetuned.onnx"
)


def _have_model() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return FINETUNED_MODEL.exists()


pytestmark = pytest.mark.skipif(
    not _have_model(),
    reason="yolov8n-chess-finetuned.onnx or onnxruntime not available",
)

# Ground-truth FEN placements encoded from the fixture filenames.
_FIXTURES: dict[str, str] = {
    "opening_real.jpg":  "r1bqkb1r/pp2pppp/2n2n2/2pp4/4P3/3P2P1/PPP2PBP/RNBQK1NR",
    "midgame_real.jpg":  "2b1k1r1/1p1p1pPB/5P2/6q1/8/6P1/1PP4P/1K1R3R",
    "endgame_real.jpg":  "8/1N1kb2p/2n2p2/2pB4/p4B2/2P3P1/1P5P/3b2K1",
}

# Minimum correct squares required (out of 64).
_MIN_CORRECT = 62


def _expand_placement(placement: str) -> list[str]:
    squares: list[str] = []
    for rank in placement.split("/"):
        for ch in rank:
            if ch.isdigit():
                squares += ["."] * int(ch)
            else:
                squares.append(ch)
    return squares


def _correct_squares(truth: str, pred: str) -> int:
    return sum(a == b for a, b in zip(_expand_placement(truth), _expand_placement(pred)))


@pytest.fixture(scope="module")
def model_backend():
    from chess_vision.model_backend import ModelBackend
    return ModelBackend(model_path=FINETUNED_MODEL)


class TestRealPhotoAccuracy:
    """Each real photo must be decoded with ≥ 62/64 correct squares."""

    @pytest.mark.parametrize("filename,truth_placement", list(_FIXTURES.items()))
    def test_per_square_accuracy(self, filename: str, truth_placement: str, model_backend):
        import cv2
        path = FIXTURES / filename
        img = cv2.imread(str(path))
        assert img is not None, f"fixture not found: {path}"

        out = model_backend.analyze(img)
        assert out.board_found, "board_found must be True on a real board photo"

        from chess_vision.vision import _grid_to_placement
        pred_placement = _grid_to_placement(out.grid)
        correct = _correct_squares(truth_placement, pred_placement)

        assert correct >= _MIN_CORRECT, (
            f"{filename}: only {correct}/64 squares correct.\n"
            f"  truth: {truth_placement}\n"
            f"  pred:  {pred_placement}"
        )

    @pytest.mark.parametrize("filename,truth_placement", list(_FIXTURES.items()))
    def test_both_kings_detected(self, filename: str, truth_placement: str, model_backend):
        """Both kings must be detected; missing a king indicates a serious failure."""
        import cv2
        path = FIXTURES / filename
        img = cv2.imread(str(path))

        out = model_backend.analyze(img)
        flat = [sq for row in out.grid for sq in row]
        assert "K" in flat, f"{filename}: white king (K) not detected"
        assert "k" in flat, f"{filename}: black king (k) not detected"

    def test_orientation_fix_applied_correctly(self, model_backend):
        """The geometric orientation check must not misfire on correctly oriented boards.

        All three fixtures are in standard orientation (white at bottom). The
        model should return the correct top-left square colour (a8 = light).
        """
        import cv2
        from chess_vision.vision import ClassicalBackend

        finder = ClassicalBackend()
        for filename in _FIXTURES:
            img = cv2.imread(str(FIXTURES / filename))
            board = finder._find_board(img)
            is_rotated = model_backend._board_needs_rotation(board)
            assert not is_rotated, (
                f"{filename}: _board_needs_rotation returned True on a "
                "correctly oriented board — orientation detection is misfiring."
            )

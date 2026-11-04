"""One ModelBackend instance serves every request (the app's singleton), and
/analyze runs it in a threadpool. Concurrent analyses must give exactly the
results they give one at a time -- no shared per-call state may leak
between requests.

The bug this guards against was real: the board fit that drives the doubt
gate used to be stashed on the shared board finder and read back after
inference. Measured under 8 threads, ~12% of calls read ANOTHER request's
fit -- so a misaligned photo could skip its warning, or a good one get
wrongly flagged.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import chess_vision.model_backend as mb
from chess_vision import vision
from chess_vision.vision import BackendOutput

ROOT = Path(__file__).resolve().parent


class _ScriptedFinder:
    """Board finder whose fit depends on the image; mimics the real class's
    shared `last_board_score` side effect so a racy reader would see it."""

    def __init__(self):
        self.last_board_score = 1.0

    def find_board_scored(self, img):
        return img, float(img[0, 0, 0]) / 100.0

    def _find_board(self, img):
        crop, score = self.find_board_scored(img)
        self.last_board_score = score
        return crop


def _backend_with_barrier(n_threads):
    """Backend whose inference step blocks until ALL threads have found
    their board -- the worst-case interleaving, forced every time."""
    backend = mb.ModelBackend.__new__(mb.ModelBackend)
    backend._board_finder = _ScriptedFinder()
    backend._load = lambda: None
    barrier = threading.Barrier(n_threads)

    def analyze_board(board):
        barrier.wait(timeout=5)
        grid = [["." for _ in range(8)] for _ in range(8)]
        grid[0][4], grid[7][4] = "k", "K"
        return BackendOutput(grid=grid, conf=[[0.95] * 8 for _ in range(8)],
                             board_found=True, notes=[])

    backend._analyze_board = analyze_board
    return backend


def _img(fit_percent):
    img = np.zeros((8, 8, 3), np.uint8)
    img[0, 0, 0] = fit_percent
    return img


def test_doubt_gate_uses_each_requests_own_board_fit():
    """Deterministic: a misaligned (fit 0.60) and an aligned (fit 0.95)
    photo analysed at the same moment must each get their OWN verdict."""
    backend = _backend_with_barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        low = pool.submit(backend.analyze, _img(60))
        high = pool.submit(backend.analyze, _img(95))
        low, high = low.result(timeout=10), high.result(timeout=10)

    assert low.notes and low.notes[0].startswith("Board alignment uncertain (grid fit 0.60)")
    assert all(c <= mb._BOARD_DOUBT_CONF for row in low.conf for c in row)
    assert high.notes == []
    assert all(c == 0.95 for row in high.conf for c in row)


def test_concurrent_first_load_is_safe():
    """Several requests arriving before the model has loaded must all get a
    fully configured session (input size from metadata, not the default)."""
    try:
        backend = mb.ModelBackend()
    except ImportError:
        pytest.skip("onnxruntime not available")
    if not backend.model_path.exists():
        pytest.skip("ONNX model not available")
    img = vision._load_bgr(ROOT / "fixtures" / "real" / "midgame_real.jpg")
    with ThreadPoolExecutor(max_workers=4) as pool:
        fens = set(pool.map(lambda _: vision.analyze_image(img, backend=backend).fen, range(4)))
    assert len(fens) == 1
    assert backend._imgsz == 640

"""Image -> FEN vision pipeline.

DECISION POINT 1 outcome
------------------------
The three approaches in the brief all depend on the network, and this
environment's network policy blocks them:

  (b) Pretrained YOLO from Hugging Face -- huggingface.co is hard-blocked
      ("Host not in allowlist"), so the trained chess weights cannot be
      downloaded. Not runnable here.
  (c) Roboflow hosted inference API -- detect.roboflow.com is blocked (403).
      Not runnable here.
  (a) chesscog -- a 2021 codebase with pinned old torch/opencv/numpy that
      conflict with current versions, plus a runtime model download from
      infrastructure not guaranteed reachable. High install risk.

So this module ships a CLASSICAL OpenCV backend that needs no model
download, behind a pluggable `VisionBackend` interface. A YOLO or Roboflow
backend can be added later as a single drop-in class implementing the same
protocol.

HONEST LIMITATION: the classical backend classifies piece *type* by
template-matching against python-chess's own piece glyph set. This
round-trips clean top-down board images (rendered diagrams, scans, flat
overhead photos) accurately. It will NOT reliably classify piece type from
oblique real-world photos -- that needs the model backend. Board detection
and perspective correction are general-purpose; piece-type matching is not.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import chess
import cv2
import numpy as np

from . import render
from .orientation import rotate_grid_180

CELL = 80
BOARD_PX = CELL * 8
PIECE_SYMBOLS = "PNBRQKpnbrqk"
FILES = "abcdefgh"

# Below this overall confidence the parse is treated as a failure and the
# pipeline falls back to the starting position with detection_failed=True.
CONF_THRESHOLD = 0.5
# Max mean-abs-difference (0-255) vs the empty-square template to call a
# square empty.
EMPTY_TOL = 18.0

STARTING_PLACEMENT = chess.STARTING_BOARD_FEN


@dataclass
class VisionResult:
    """Result of running the vision pipeline on one image."""

    fen: str
    confidence: float
    detection_failed: bool
    per_square_confidence: Optional[dict[str, float]] = None


@dataclass
class BackendOutput:
    """Raw output of a VisionBackend before orientation / FEN assembly."""

    # 8x8 grid of piece symbols ('.' for empty). Row 0 == rank 8 (top).
    grid: list[list[str]]
    # 8x8 grid of per-square confidences in [0, 1], same indexing as `grid`.
    conf: list[list[float]]
    board_found: bool = True


@runtime_checkable
class VisionBackend(Protocol):
    """Interface a vision backend must implement.

    Drop-in point for a future YOLO / Roboflow backend: implement `analyze`
    returning a BackendOutput and pass an instance to `analyze_image`.
    """

    name: str

    def analyze(self, image_bgr: np.ndarray) -> BackendOutput:
        ...


# --------------------------------------------------------------------------
# Classical OpenCV backend
# --------------------------------------------------------------------------


def _full_fen(placement: str) -> str:
    return f"{placement} w - - 0 1"


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _crop_cell(img_bgr: np.ndarray, row: int, col: int) -> np.ndarray:
    cell = img_bgr[row * CELL : (row + 1) * CELL, col * CELL : (col + 1) * CELL]
    return cv2.resize(cell, (CELL, CELL))


def _gray(cell_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)


class ClassicalBackend:
    """No-model board reader: detect board -> warp -> classify 64 squares."""

    name = "classical-opencv"

    def __init__(self) -> None:
        self._templates: dict[str, dict[str, np.ndarray]] | None = None
        self._bg_val: dict[str, float] = {}

    # -- reference templates ------------------------------------------------

    def _build_templates(self) -> None:
        """Render reference cells from python-chess's own piece set: each
        piece on a light square and on a dark square, plus empty squares.
        a1 is a dark square (image row 7, col 0); b1 is light (row 7, col 1).
        """
        templates: dict[str, dict[str, np.ndarray]] = {}

        empty_rgb = render.fen_to_image(
            _full_fen("8/8/8/8/8/8/8/8"), size=BOARD_PX, coordinates=False
        )
        empty_bgr = _to_bgr(empty_rgb)
        templates["."] = {
            "dark": _gray(_crop_cell(empty_bgr, 7, 0)),
            "light": _gray(_crop_cell(empty_bgr, 7, 1)),
        }
        self._bg_val["dark"] = float(templates["."]["dark"].mean())
        self._bg_val["light"] = float(templates["."]["light"].mean())

        for sym in PIECE_SYMBOLS:
            piece = chess.Piece.from_symbol(sym)

            bb_dark = chess.BaseBoard.empty()
            bb_dark.set_piece_at(chess.A1, piece)
            dark_bgr = _to_bgr(
                render.fen_to_image(
                    _full_fen(bb_dark.board_fen()), size=BOARD_PX, coordinates=False
                )
            )

            bb_light = chess.BaseBoard.empty()
            bb_light.set_piece_at(chess.B1, piece)
            light_bgr = _to_bgr(
                render.fen_to_image(
                    _full_fen(bb_light.board_fen()), size=BOARD_PX, coordinates=False
                )
            )

            templates[sym] = {
                "dark": _gray(_crop_cell(dark_bgr, 7, 0)),
                "light": _gray(_crop_cell(light_bgr, 7, 1)),
            }

        self._templates = templates

    @property
    def templates(self) -> dict[str, dict[str, np.ndarray]]:
        if self._templates is None:
            self._build_templates()
        assert self._templates is not None
        return self._templates

    # -- board detection ----------------------------------------------------

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2).astype("float32")
        ordered = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered[0] = pts[np.argmin(s)]  # top-left
        ordered[2] = pts[np.argmax(s)]  # bottom-right
        ordered[1] = pts[np.argmin(d)]  # top-right
        ordered[3] = pts[np.argmax(d)]  # bottom-left
        return ordered

    @staticmethod
    def _checkerboard_score(board: np.ndarray) -> float:
        """How well an already-cropped BOARD_PX image looks like an 8x8
        chessboard: sample each cell's corners (always background, never a
        glyph) and measure how cleanly they form an alternating pattern.
        Returns a value in [0.5, 1.0]."""
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        p = 8
        vals = np.zeros((8, 8), dtype=np.float64)
        for r in range(8):
            for c in range(8):
                cell = gray[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
                corners = np.concatenate(
                    [
                        cell[:p, :p].ravel(),
                        cell[:p, -p:].ravel(),
                        cell[-p:, :p].ravel(),
                        cell[-p:, -p:].ravel(),
                    ]
                )
                vals[r, c] = np.median(corners)
        thr = (vals.min() + vals.max()) / 2.0
        is_light = vals > thr
        pattern = (np.indices((8, 8)).sum(axis=0) % 2) == 0
        return float(max((is_light == pattern).mean(), (is_light != pattern).mean()))

    def _largest_quad(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Find a 4-corner contour likely to be the board. Iterates the
        approxPolyDP epsilon (real-world contours rarely fit at exactly 0.02)
        and falls back to minAreaRect of the largest big-enough contour --
        a rotated rectangle is fine, the warp will square it up."""
        h, w = img_bgr.shape[:2]
        img_area = float(h * w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        big = [c for c in contours if cv2.contourArea(c) >= 0.15 * img_area]

        # Strategy 1: largest contour that approximates to exactly 4 corners.
        # Try multiple epsilons; real-world board contours rarely fit 0.02.
        best: np.ndarray | None = None
        best_area = 0.0
        for c in big:
            peri = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            for eps in (0.005, 0.01, 0.02, 0.03, 0.04, 0.06):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    _, _, bw, bh = cv2.boundingRect(approx)
                    if 0.55 < bw / float(bh) < 1.8 and area > best_area:
                        best, best_area = approx, area
                    break
        if best is not None:
            return best

        # Strategy 2: minAreaRect on the largest contour -- a rotated rect.
        if big:
            biggest = max(big, key=cv2.contourArea)
            rect = cv2.minAreaRect(biggest)
            (_, _), (rw, rh), _ = rect
            if rw > 0 and rh > 0 and 0.55 < (rw / rh) < 1.8:
                box = cv2.boxPoints(rect).astype("float32")
                return box.reshape(4, 1, 2)
        return None

    def _warp_quad(self, img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = self._order_points(quad)
        dst = np.array(
            [[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]],
            dtype="float32",
        )
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img_bgr, m, (BOARD_PX, BOARD_PX))

    def _scan_for_board(self, img_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Sliding-square search for the board. Useful on top-down photos
        where the board doesn't fill the frame and `_largest_quad`'s 4-corner
        contour approximation fails (real-world clutter, watermarks, edge
        labels). Tries square crops at several scales/positions and returns
        the one with the highest checkerboard score."""
        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)
        best_score = 0.0
        best_crop: np.ndarray | None = None
        for side_frac in (0.95, 0.85, 0.75, 0.65, 0.55):
            side = int(min_dim * side_frac)
            if side < 200:
                continue
            step = max(20, side // 12)
            for y in range(0, max(1, h - side + 1), step):
                for x in range(0, max(1, w - side + 1), step):
                    crop = img_bgr[y : y + side, x : x + side]
                    resized = cv2.resize(crop, (BOARD_PX, BOARD_PX))
                    score = self._checkerboard_score(resized)
                    if score > best_score:
                        best_score, best_crop = score, resized
        return best_crop, best_score

    def _find_board(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return the board cropped/warped to BOARD_PX x BOARD_PX.

        Three candidate strategies, scored on checkerboard cleanness:
          (a) whole frame resized -- works on synthetic fixtures.
          (b) largest 4-corner quad warp -- works on board-fills-frame photos.
          (c) sliding-square scan -- works on top-down photos where the board
              is a sub-rectangle in white/textured background (real-world
              stock photos, phone shots with margin).
        Highest checkerboard score wins.
        """
        whole = cv2.resize(img_bgr, (BOARD_PX, BOARD_PX))
        best_score = self._checkerboard_score(whole)
        best = whole
        if best_score >= 0.97:
            return best

        quad = self._largest_quad(img_bgr)
        if quad is not None:
            warped = self._warp_quad(img_bgr, quad)
            s = self._checkerboard_score(warped)
            if s > best_score:
                best_score, best = s, warped

        scan_crop, scan_score = self._scan_for_board(img_bgr)
        if scan_crop is not None and scan_score > best_score:
            best_score, best = scan_score, scan_crop

        return best

    # -- per-square classification -----------------------------------------

    def _square_bg(self, cell_gray: np.ndarray) -> str:
        """Classify the square background as 'light' or 'dark' by sampling the
        cell corners (chess glyphs never reach the corners of a square)."""
        p = 10
        corners = np.concatenate(
            [
                cell_gray[:p, :p].ravel(),
                cell_gray[:p, -p:].ravel(),
                cell_gray[-p:, :p].ravel(),
                cell_gray[-p:, -p:].ravel(),
            ]
        )
        val = float(np.median(corners))
        d_light = abs(val - self._bg_val.get("light", 200.0))
        d_dark = abs(val - self._bg_val.get("dark", 130.0))
        return "light" if d_light <= d_dark else "dark"

    def _classify_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        cell = _gray(cell_bgr)
        bg = self._square_bg(cell)

        empty_t = self.templates["."][bg]
        mad_empty = float(np.mean(np.abs(cell.astype(np.int16) - empty_t.astype(np.int16))))
        if mad_empty < EMPTY_TOL:
            conf = max(0.0, 1.0 - mad_empty / (EMPTY_TOL * 2.0))
            return ".", conf

        best_sym = "."
        best_score = -1.0
        for sym in PIECE_SYMBOLS:
            templ = self.templates[sym][bg]
            score = float(
                cv2.matchTemplate(cell, templ, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            if score > best_score:
                best_score, best_sym = score, sym
        return best_sym, max(0.0, best_score)

    def analyze(self, image_bgr: np.ndarray) -> BackendOutput:
        board = self._find_board(image_bgr)
        grid: list[list[str]] = []
        conf: list[list[float]] = []
        for r in range(8):
            row_sym: list[str] = []
            row_conf: list[float] = []
            for c in range(8):
                cell = _crop_cell(board, r, c)
                sym, score = self._classify_cell(cell)
                row_sym.append(sym)
                row_conf.append(score)
            grid.append(row_sym)
            conf.append(row_conf)
        return BackendOutput(grid=grid, conf=conf, board_found=True)


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

_DEFAULT_BACKEND: "VisionBackend | None" = None


def _make_default_backend() -> "VisionBackend":
    """Pick a backend based on the CVC_BACKEND env var.

    "classical" (default) -> ClassicalBackend (template matching).
    "model"               -> ModelBackend (YOLO11n ONNX). Requires the
                             onnxruntime package and a vendored .onnx file.
    """
    name = os.environ.get("CVC_BACKEND", "classical").lower()
    if name == "model":
        from chess_vision.model_backend import ModelBackend
        return ModelBackend()
    return ClassicalBackend()


def _get_default_backend() -> "VisionBackend":
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = _make_default_backend()
    return _DEFAULT_BACKEND


def find_board(img_bgr: np.ndarray) -> np.ndarray:
    """Geometrically locate the chessboard in `img_bgr` and warp it to
    BOARD_PX x BOARD_PX. Shared by all backends -- board detection is
    classical CV, only piece classification differs across backends."""
    # ClassicalBackend's _find_board uses no piece-classification state, so
    # instantiating one here is cheap (templates are built lazily).
    finder = ClassicalBackend()
    return finder._find_board(img_bgr)


def _grid_to_placement(grid: list[list[str]]) -> str:
    ranks = []
    for row in grid:
        out = ""
        empties = 0
        for sym in row:
            if sym == ".":
                empties += 1
            else:
                if empties:
                    out += str(empties)
                    empties = 0
                out += sym
        if empties:
            out += str(empties)
        ranks.append(out)
    return "/".join(ranks)


def _load_bgr(source) -> np.ndarray:
    """Load `source` (path or BGR ndarray) into a BGR uint8 image."""
    if isinstance(source, np.ndarray):
        img = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {source}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {source}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def fallback_result(side_to_move: str = "w") -> VisionResult:
    """The detection-failed result: the standard starting position so the
    user has a board to correct manually."""
    return VisionResult(
        fen=_full_starting_fen(side_to_move),
        confidence=0.0,
        detection_failed=True,
        per_square_confidence=None,
    )


def _full_starting_fen(side_to_move: str) -> str:
    stm = "b" if side_to_move == "b" else "w"
    return f"{STARTING_PLACEMENT} {stm} - - 0 1"


def analyze_image(
    source,
    *,
    orientation: str = "white",
    side_to_move: str = "w",
    backend: VisionBackend | None = None,
) -> VisionResult:
    """Run the vision pipeline on an image.

    `source` is a file path or a BGR numpy image. On any failure -- bad
    image, no board, or low overall confidence -- this returns
    `fallback_result()` (starting position, detection_failed=True).
    """
    backend = backend or _get_default_backend()
    try:
        img = _load_bgr(source)
        if min(img.shape[:2]) < 64:
            return fallback_result(side_to_move)

        out = backend.analyze(img)
        if not out.board_found:
            return fallback_result(side_to_move)

        grid = out.grid
        conf = out.conf
        if orientation == "black":
            grid = rotate_grid_180(grid)
            conf = rotate_grid_180(conf)

        scores = [v for row in conf for v in row]
        overall = float(sum(scores) / len(scores)) if scores else 0.0
        if overall < CONF_THRESHOLD:
            return fallback_result(side_to_move)

        placement = _grid_to_placement(grid)
        stm = "b" if side_to_move == "b" else "w"
        fen = f"{placement} {stm} - - 0 1"

        per_square: dict[str, float] = {}
        for r in range(8):
            for c in range(8):
                square = f"{FILES[c]}{8 - r}"
                per_square[square] = round(float(conf[r][c]), 4)

        return VisionResult(
            fen=fen,
            confidence=round(overall, 4),
            detection_failed=False,
            per_square_confidence=per_square,
        )
    except Exception:
        return fallback_result(side_to_move)

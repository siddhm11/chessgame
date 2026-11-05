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
import math
import os
import re
import warnings
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
# Below this board-grid fit there is no chessboard in the photo (measured by
# bench/board_presence_eval.py: 508 real boards scored 0.732-1.01, 96
# ordinary non-chess photos 0.48-0.63; 0.68 is the midpoint of that gap).
# Never applied when the user marked the board's corners themselves.
NO_BOARD_FIT = 0.68

# Pixel budgets, checked from the image header BEFORE decoding. A 1 MB PNG
# can declare 32768x32768 pixels; decoding it peaked at 6.2 GB RSS (audit,
# measured). JPEGs are decoded at reduced scale (draft mode), so very large
# phone photos stay cheap; other formats decode at full size.
MAX_PIXELS_JPEG = 250_000_000
MAX_PIXELS_OTHER = 40_000_000


class ImageTooLarge(ValueError):
    """The image declares more pixels than we will decode."""


def effective_cpus() -> int:
    """CPUs this process may actually use: CVC_THREADS if set, else the
    cgroup CPU quota (a 2-vCPU container on a 16-core host must not size its
    thread pools to 16 -- ~4x slower, audit-measured), else affinity/count."""
    env = os.environ.get("CVC_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota = None
    try:  # cgroup v2
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota:
        n = min(n, max(1, math.ceil(quota)))
    return max(1, n)


THREADS = effective_cpus()
cv2.setNumThreads(THREADS)

# Longest side images are processed at. 24 MP phone/DSLR photos are decoded
# straight to ~this size; the board crop still gets 2-3x the 640px it needs.
MAX_WORK_SIDE = 2000
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
    # Human-readable record of auto-repairs made by the sanity layer
    # (e.g. "e8: no K found anywhere; reread Q as K"). Square names match
    # the returned FEN's orientation.
    notes: list[str] = field(default_factory=list)
    # Where the board was found: 4 (x, y) points in [0, 1] coordinates of the
    # (upright) photo, clockwise from top-left. None when no board was found.
    board_corners: Optional[list[list[float]]] = None


@dataclass
class BackendOutput:
    """Raw output of a VisionBackend before orientation / FEN assembly."""

    # 8x8 grid of piece symbols ('.' for empty). Row 0 == rank 8 (top).
    grid: list[list[str]]
    # 8x8 grid of per-square confidences in [0, 1], same indexing as `grid`.
    conf: list[list[float]]
    board_found: bool = True
    # Auto-repair notes from the backend's sanity layer, square-prefixed
    # ("e8: ..."), in the backend's (pre-orientation) frame.
    notes: list[str] = field(default_factory=list)
    # Board quad used, normalised to [0, 1] image coordinates (TL, TR, BR, BL).
    corners: Optional[list[list[float]]] = None


@dataclass
class BoardCrop:
    """Result of locating the board in a photo."""

    image: np.ndarray   # board warped to BOARD_PX x BOARD_PX
    fit: float          # sharp grid fit, ~0.9-1.0 when it sits on the squares
    quad: np.ndarray    # (4, 2) corners in source pixels: TL, TR, BR, BL


def no_board_output() -> BackendOutput:
    """What a backend returns when the photo holds no chessboard."""
    return BackendOutput(grid=[["."] * 8 for _ in range(8)],
                         conf=[[0.0] * 8 for _ in range(8)], board_found=False)


def normalise_quad(quad: np.ndarray, shape) -> list[list[float]]:
    """Pixel quad -> [[x, y], ...] in [0, 1] (rounded; for the UI/JSON)."""
    h, w = shape[:2]
    q = np.asarray(quad, dtype="float64").reshape(4, 2) / [w, h]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in q]


@runtime_checkable
class VisionBackend(Protocol):
    """Interface a vision backend must implement.

    Drop-in point for a future YOLO / Roboflow backend: implement `analyze`
    returning a BackendOutput and pass an instance to `analyze_image`.
    """

    name: str

    def analyze(self, image_bgr: np.ndarray, corners=None) -> BackendOutput:
        """`corners`: optional user-marked board corners, 4 (x, y) in [0, 1]."""
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
        # Grid-fit score of the most recent _find_board crop, in [0.5, ~1.0].
        self.last_board_score: float = 1.0

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

    def _largest_quad(self, img_bgr: np.ndarray, dilate_iter: int = 2) -> np.ndarray | None:
        """Find a 4-corner contour likely to be the board. Iterates the
        approxPolyDP epsilon (real-world contours rarely fit at exactly 0.02)
        and falls back to minAreaRect of the largest big-enough contour --
        a rotated rectangle is fine, the warp will square it up."""
        h, w = img_bgr.shape[:2]
        img_area = float(h * w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120)
        if dilate_iter:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iter)
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
        the (4,1,2) corner quad of the one with the highest checkerboard
        score. Coarse by design -- `_refine_quad` snaps it to the grid."""
        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)
        # Score on a downscaled copy: the coarse search only needs the rough
        # location, and this keeps 6000x4000 photos fast.
        k = min(1.0, 1200.0 / max(h, w))
        small = cv2.resize(img_bgr, (int(w * k), int(h * k))) if k < 1 else img_bgr
        sh, sw = small.shape[:2]
        best_score = 0.0
        best_quad: np.ndarray | None = None
        for side_frac in (0.95, 0.85, 0.75, 0.65, 0.55):
            side = int(min(sh, sw) * side_frac)
            if side < 200 * k or side < 64:
                continue
            step = max(8, side // 12)
            for y in range(0, max(1, sh - side + 1), step):
                for x in range(0, max(1, sw - side + 1), step):
                    crop = small[y : y + side, x : x + side]
                    resized = cv2.resize(crop, (BOARD_PX, BOARD_PX))
                    score = self._checkerboard_score(resized)
                    if score > best_score:
                        best_score = score
                        x0, y0, s = x / k, y / k, side / k
                        best_quad = np.array(
                            [[[x0, y0]], [[x0 + s, y0]], [[x0 + s, y0 + s]], [[x0, y0 + s]]],
                            dtype="float32",
                        )
        return best_quad, best_score

    # -- grid refinement -----------------------------------------------------
    #
    # Every strategy above proposes a ROUGH board quad. `_checkerboard_score`
    # takes the median of each cell's four corners, which hides a half-cell
    # misalignment (two corners in one square + two in the next still give a
    # plausible median). On real photos the rough quad is often the outer mat
    # including its coordinate-label margin, or a sliding-scan crop that is
    # half a square off -- every piece then straddles two cells.
    #
    # `_grid_fit` scores the four corner patches of every cell SEPARATELY, so
    # a half-cell offset (corners in different squares) scores ~0.5 and only
    # a grid sitting exactly on the squares scores ~1.0. `_refine_quad`
    # rectifies the rough quad with a margin around it and searches that
    # working image for the rectangle the 8x8 grid really occupies.

    _REFINE_WORK = 900   # working-image side (px)
    _REFINE_PAD = 100    # margin kept around the rough quad (~1.15 cells)
    # Corner patches hugging each cell corner: score peaks within a few px of
    # exact alignment. Used for fine-tuning and as the final judge.
    _SHARP_INSET = 0.025
    _SHARP_PATCH = 0.06
    # Sharp fit above which a candidate is trusted without trying the slow
    # sliding scan (aligned crops measure ~0.92-1.0 on real photos).
    _GOOD_FIT = 0.90

    @staticmethod
    def _grid_fit(
        ii: np.ndarray, x0: float, y0: float, x1: float, y1: float,
        inset: float = 0.10, patch: float = 0.16, strict: bool = True,
    ) -> float:
        """Alignment score in [0.5, 1.0] for an 8x8 grid on [x0,x1]x[y0,y1].

        `ii` is the integral image of a grayscale float image (cv2.integral).
        `inset`/`patch` (fractions of a cell) set where each corner patch is
        sampled. The default is forgiving: shifts up to ~10% of a cell score
        the same, giving a wide basin for coarse search. Small values (e.g.
        0.025/0.06) hug the cell corners so the score peaks sharply at exact
        alignment -- used for the final fine-tuning.
        """
        cw, ch = (x1 - x0) / 8.0, (y1 - y0) / 8.0
        if cw < 8 or ch < 8:
            return 0.0
        ix, iy = inset * cw, inset * ch                       # inset from corner
        pw, ph = max(2.0, patch * cw), max(2.0, patch * ch)   # patch size
        cx = x0 + np.arange(8) * cw
        cy = y0 + np.arange(8) * ch
        xs = np.stack([cx + ix, cx + cw - ix - pw])          # (2, 8) left/right
        ys = np.stack([cy + iy, cy + ch - iy - ph])          # (2, 8) top/bottom
        H, W = ii.shape[0] - 1, ii.shape[1] - 1
        xa = np.clip(np.rint(xs), 0, W - 1).astype(int)
        xb = np.clip(np.rint(xs + pw), 1, W).astype(int)
        ya = np.clip(np.rint(ys), 0, H - 1).astype(int)
        yb = np.clip(np.rint(ys + ph), 1, H).astype(int)
        # vals[ty, tx, r, c]: mean of patch (ty,tx) in cell (r,c).
        Y1, Y2 = ya[:, None, :, None], yb[:, None, :, None]
        X1, X2 = xa[None, :, None, :], xb[None, :, None, :]
        area = np.maximum(1, (Y2 - Y1) * (X2 - X1))
        vals = (ii[Y2, X2] - ii[Y1, X2] - ii[Y2, X1] + ii[Y1, X1]) / area
        flat = np.sort(vals.ravel())
        # Otsu threshold over the 256 patch means.
        n = flat.size
        csum = np.cumsum(flat)
        k = np.arange(1, n)
        m0 = csum[:-1] / k
        m1 = (csum[-1] - csum[:-1]) / (n - k)
        t_idx = int(np.argmax(k * (n - k) * (m0 - m1) ** 2))
        thr = (flat[t_idx] + flat[t_idx + 1]) / 2.0
        light = vals > thr
        pattern = (np.indices((8, 8)).sum(axis=0) % 2) == 0
        match = light == pattern[None, None]          # (2, 2, 8, 8)
        if match.mean() < 0.5:                        # parity is arbitrary
            match = ~match
        # Every row AND every column must look like chessboard. A grid shifted
        # by one whole square keeps the alternation everywhere except the one
        # row/column hanging off the board -- the mean barely notices (~6%),
        # the worst line drops to ~0.5.
        # `strict=False` keeps only the mean: smooth, so hill-climbing has a
        # slope to follow far from the answer -- but blind to whole-square
        # shifts. Use it to search; use strict to judge.
        if strict:
            per_col = match.mean(axis=(0, 1, 2))
            per_row = match.mean(axis=(0, 1, 3))
            worst = min(per_col.min(), per_row.min())
            score = 0.5 * match.mean() + 0.5 * worst
        else:
            score = match.mean()
        # Weak tie-breaker towards high light/dark contrast, so a grid on the
        # real squares beats a same-pattern grid spilling onto the mat.
        contrast = (m1[t_idx] - m0[t_idx]) / 255.0
        return float(score + 0.02 * contrast)

    @staticmethod
    def _lattice_1d(profile: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
        """Best 9 equally spaced peaks in `profile`: returns (offset, spacing).

        Spacing is searched over [lo, hi] px; every offset is tried. The 9
        points are the board's 7 interior grid lines plus its 2 outer edges.
        """
        n = profile.size
        best = (-1.0, 0.0, lo)
        k = np.arange(9)
        for c in np.arange(lo, hi, 0.5):
            span = 8 * c
            if span >= n - 1:
                break
            offs = np.arange(0, n - 1 - span)
            idx = np.rint(offs[:, None] + k[None, :] * c).astype(int)
            s = profile[idx].sum(axis=1)
            i = int(np.argmax(s))
            if s[i] > best[0]:
                best = (float(s[i]), float(offs[i]), float(c))
        return best[1], best[2]

    def _lattice_init(self, gray: np.ndarray) -> list[float]:
        """Estimate the 8x8 grid rectangle from straight, evenly spaced edges.

        Grid lines run the full board length, so the MEDIAN edge strength
        along each column (or row) is high only on grid lines -- pieces, which
        cover a minority of any line, barely move it. A 1-D search for 9
        evenly spaced peaks per axis then gives scale and position directly,
        avoiding the many local optima a hill-climb meets on a periodic board.
        """
        S = gray.shape[0]
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
        band = slice(int(0.3 * S), int(0.7 * S))   # surely inside the board
        px = np.median(gx[band, :], axis=0)
        py = np.median(gy[:, band], axis=1)
        px = np.convolve(px, np.ones(3) / 3, mode="same")
        py = np.convolve(py, np.ones(3) / 3, mode="same")
        lo, hi = 0.35 * S / 8, 1.0 * S / 8
        ox, cx = self._lattice_1d(px, lo, hi)
        oy, cy = self._lattice_1d(py, lo, hi)
        return [ox, oy, ox + 8 * cx, oy + 8 * cy]

    def _refine_quad(self, img_bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, float]:
        """Snap a rough board quad onto the actual 8x8 grid.

        Returns (refined quad in original-image coords, grid-fit score).
        """
        S, P = self._REFINE_WORK, self._REFINE_PAD
        src = self._order_points(quad)
        dst = np.array([[P, P], [S - P, P], [S - P, S - P], [P, S - P]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        work = cv2.warpPerspective(img_bgr, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float64)
        ii = cv2.integral(gray)

        rect = [float(P), float(P), float(S - P), float(S - P)]  # x0,y0,x1,y1
        best = self._grid_fit(ii, *rect)
        min_side = 0.35 * S

        def moves(step):
            for d in range(-4, 5):
                if d == 0:
                    continue
                o = d * step
                yield (o, 0, o, 0)      # translate x
                yield (0, o, 0, o)      # translate y
                yield (-o, -o, o, o)    # scale about centre
                yield (o, 0, 0, 0)
                yield (0, o, 0, 0)
                yield (0, 0, o, 0)
                yield (0, 0, 0, o)

        def climb(rect, steps, **fit_kw):
            best = self._grid_fit(ii, *rect, **fit_kw)
            for step in steps:
                for _ in range(8):
                    improved = False
                    for dx0, dy0, dx1, dy1 in moves(step):
                        c = [rect[0] + dx0, rect[1] + dy0, rect[2] + dx1, rect[3] + dy1]
                        if (c[0] < 0 or c[1] < 0 or c[2] > S or c[3] > S
                                or c[2] - c[0] < min_side or c[3] - c[1] < min_side):
                            continue
                        s = self._grid_fit(ii, *c, **fit_kw)
                        if s > best + 1e-9:
                            best, rect, improved = s, c, True
                    if not improved:
                        break
            return rect

        def judge(r):
            return self._grid_fit(ii, *r, inset=self._SHARP_INSET, patch=self._SHARP_PATCH)

        def fit_from(start):
            # Every stage is judged by the same strict score and the best is
            # kept, so refinement can never return a worse grid than it was
            # given (the smooth stage-1 score can be led astray by lighting
            # gradients). On near-ties the LATER stage wins: stage 3 pins
            # sub-square precision that the forgiving score cannot see.
            staged = [start]
            # Stage 1 -- smooth mean score, wide basin: find the squares.
            # May settle a whole square off (the mean can't see that).
            rect = climb(start, (24, 12, 6), strict=False)
            staged.append(rect)
            # Stage 2 -- explicitly try whole-square shifts, judged strictly
            # (every row and column must be chessboard).
            cw, ch = (rect[2] - rect[0]) / 8.0, (rect[3] - rect[1]) / 8.0
            options = [(self._grid_fit(ii, *rect), rect)]
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    c = [rect[0] + sx * cw, rect[1] + sy * ch,
                         rect[2] + sx * cw, rect[3] + sy * ch]
                    if c[0] >= 0 and c[1] >= 0 and c[2] <= S and c[3] <= S:
                        options.append((self._grid_fit(ii, *c), c))
            rect = max(options, key=lambda o: o[0])[1]
            staged.append(rect)
            # Stage 3 -- sharp corner patches: pin each edge exactly onto
            # the square boundary.
            rect = climb(rect, (6, 3, 2, 1), inset=self._SHARP_INSET, patch=self._SHARP_PATCH)
            staged.append(rect)
            # Judge with the SHARP strict score. The forgiving patches can't
            # tell a grid sitting exactly on the squares from one ~1/6 square
            # off (both ~0.9, measured); the sharp patches collapse within a
            # few pixels of misalignment, so the best-aligned stage wins.
            scored = [(judge(r), r) for r in staged]
            top = max(s for s, _ in scored)
            return next((s, r) for s, r in reversed(scored) if s >= top - 0.01)

        # Two starting points: the rough candidate itself, and an estimate
        # from evenly spaced straight edges (robust when the candidate's
        # scale is far off, e.g. a mat with a wide label margin).
        starts = [rect]
        try:
            lat = self._lattice_init(gray)
            if lat[2] - lat[0] >= min_side and lat[3] - lat[1] >= min_side:
                starts.append(lat)
        except (ValueError, IndexError):
            pass
        best, rect = max((fit_from(s) for s in starts), key=lambda o: o[0])

        x0, y0, x1, y1 = rect
        pts = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype="float32")
        refined = cv2.perspectiveTransform(pts, np.linalg.inv(M))
        return refined, best

    def _find_board(self, img_bgr: np.ndarray) -> np.ndarray:
        """Compatibility wrapper: the crop only, with the grid-fit score left
        on `self.last_board_score`. NOT safe when one backend instance serves
        concurrent requests -- use `locate_board` there."""
        found = self.locate_board(img_bgr)
        self.last_board_score = found.fit
        return found.image

    def find_board_scored(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """(board warped to BOARD_PX x BOARD_PX, grid-fit score)."""
        found = self.locate_board(img_bgr)
        return found.image, found.fit

    # A user-placed corner may be snapped onto the grid, but never moved
    # further than this fraction of a square: the user's placement wins over
    # a doubtful automatic fit.
    _USER_SNAP_MAX = 0.75

    def locate_board(self, img_bgr: np.ndarray, corners=None) -> "BoardCrop":
        """Find the board and warp it to BOARD_PX x BOARD_PX.

        `corners`: optional 4 (x, y) points in [0, 1] image coordinates, any
        order -- the user marking the board. They are snapped onto the exact
        8x8 grid (users click roughly; boards often have a label margin), but
        no corner moves more than `_USER_SNAP_MAX` of a square.

        Without corners, rough candidates are tried:
          (a) whole frame -- synthetic fixtures / board fills the frame.
          (b) largest 4-corner quad -- board edge clearly visible.
          (c) sliding-square scan -- board is a sub-rectangle of the photo.
        Each is snapped onto the 8x8 grid by `_refine_quad`; the best grid fit
        wins. The fit (sharp `_grid_fit`, ~0.9-1.0 when the crop sits on the
        squares) lets callers tell a trustworthy crop from a doubtful one.
        Pure function of the input: no shared state, thread-safe.
        """
        h, w = img_bgr.shape[:2]
        if corners is not None:
            return self._locate_from_user(img_bgr, corners)

        whole = cv2.resize(img_bgr, (BOARD_PX, BOARD_PX))
        frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        if self._checkerboard_score(whole) >= 0.97:
            # Rendered diagrams / perfectly framed boards: already aligned.
            ii = cv2.integral(cv2.cvtColor(whole, cv2.COLOR_BGR2GRAY).astype(np.float64))
            score = self._grid_fit(
                ii, 0, 0, BOARD_PX, BOARD_PX,
                inset=self._SHARP_INSET, patch=self._SHARP_PATCH)
            return BoardCrop(whole, float(score), frame)

        best_quad, best_score = None, -1.0

        def consider(cand):
            nonlocal best_quad, best_score
            try:
                refined, score = self._refine_quad(img_bgr, cand)
            except cv2.error:
                return
            if score > best_score:
                best_quad, best_score = refined, score

        consider(np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype="float32"))
        # Contour quads at several edge thicknesses. Which one isolates the
        # board edge depends on resolution and clutter (thick edges merge the
        # board with its label strip; thin ones break the outline) -- measured,
        # no single setting works on every photo. Most reliable first; stop
        # as soon as one fits confidently.
        for dilate_iter in (1, 2, 0):
            if best_score >= self._GOOD_FIT:
                break
            quad = self._largest_quad(img_bgr, dilate_iter=dilate_iter)
            if quad is not None:
                consider(quad.astype("float32"))
        # The sliding scan is the slowest step (~0.5 s); only pay for it when
        # the cheaper candidates didn't already land a confident grid fit.
        if best_score < self._GOOD_FIT:
            scan_quad, _ = self._scan_for_board(img_bgr)
            if scan_quad is not None:
                consider(scan_quad)

        if best_quad is None:
            return BoardCrop(whole, float(best_score), frame)
        quad = self._order_points(best_quad)
        return BoardCrop(self._warp_quad(img_bgr, quad), float(best_score), quad)

    def _locate_from_user(self, img_bgr: np.ndarray, corners) -> "BoardCrop":
        h, w = img_bgr.shape[:2]
        user = self._order_points(
            np.asarray(corners, dtype="float64").reshape(4, 2) * [w, h]
        ).astype("float32")
        quad = user
        try:
            refined, _ = self._refine_quad(img_bgr, user.reshape(4, 1, 2))
            refined = self._order_points(refined)
            side = max(np.linalg.norm(user[i] - user[(i + 1) % 4]) for i in range(4))
            if np.linalg.norm(refined - user, axis=1).max() <= self._USER_SNAP_MAX * side / 8:
                quad = refined
        except (cv2.error, np.linalg.LinAlgError):
            pass
        crop = self._warp_quad(img_bgr, quad)
        ii = cv2.integral(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64))
        fit = self._grid_fit(ii, 0, 0, BOARD_PX, BOARD_PX,
                             inset=self._SHARP_INSET, patch=self._SHARP_PATCH)
        return BoardCrop(crop, float(fit), quad)

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

    def analyze(self, image_bgr: np.ndarray, corners=None) -> BackendOutput:
        found = self.locate_board(image_bgr, corners)
        if corners is None and found.fit < NO_BOARD_FIT:
            return no_board_output()
        board = found.image
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
        return BackendOutput(grid=grid, conf=conf, board_found=True,
                             corners=normalise_quad(found.quad, image_bgr.shape))


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

_DEFAULT_BACKEND: "VisionBackend | None" = None


def _make_default_backend() -> "VisionBackend":
    """Pick a backend based on the CVC_BACKEND env var.

    "classical" (default) -> ClassicalBackend (template matching).
    "model"               -> ModelBackend (fine-tuned YOLO ONNX). Requires
                             the onnxruntime package and a vendored .onnx
                             file; falls back to ClassicalBackend (with a
                             warning) when either is missing, so a slim
                             deployment degrades gracefully instead of
                             failing every analysis.
    """
    name = os.environ.get("CVC_BACKEND", "classical").lower()
    if name == "model":
        try:
            from chess_vision.model_backend import ModelBackend

            backend = ModelBackend()
            if backend.model_path.exists():
                return backend
            print(
                f"WARNING: CVC_BACKEND=model but {backend.model_path} is "
                "missing; falling back to the classical backend.",
                flush=True,
            )
        except ImportError:
            print(
                "WARNING: CVC_BACKEND=model but onnxruntime is not "
                "installed; falling back to the classical backend.",
                flush=True,
            )
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
    """Load `source` (path or BGR ndarray) into a BGR uint8 image.

    Path inputs are routed through Pillow so EXIF orientation metadata is
    honoured -- phone cameras typically write the sensor data sideways and
    record "rotate 90° CW" in EXIF; OpenCV's imread ignores that tag and
    would hand the pipeline a portrait crop of a landscape board.
    """
    if isinstance(source, np.ndarray):
        img = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {source}")
        img = _decode_upright(path)
        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # local files PIL can't read
    if img is None:
        raise ValueError(f"could not decode image: {source}")
    return _normalise(img)


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode uploaded bytes exactly as `_load_bgr` decodes a file: EXIF
    orientation applied, size capped at MAX_WORK_SIDE. None if undecodable;
    raises ImageTooLarge past the pixel budget.

    Deliberately no OpenCV fallback for uploads: cv2.imdecode has no size
    guard, so it would decode the very bombs Pillow refuses.
    """
    from io import BytesIO

    img = _decode_upright(BytesIO(data))
    return None if img is None else _normalise(img)


def _decode_upright(fp) -> np.ndarray | None:
    """Pillow decode with EXIF orientation applied. JPEGs are decoded straight
    at reduced scale (DCT-domain draft mode, near free); draft keeps both
    sides >= the request, so the board keeps far more pixels than the 640px
    crop needs. Returns BGR, or None if Pillow can't read it."""
    from PIL import Image, ImageOps  # local: only the loaders need PIL

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS_JPEG   # our own budget check is below
    try:
        with warnings.catch_warnings():
            # Pillow warns while reading an oversized header; the size is
            # handled (and refused) explicitly just below.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            pil_img = Image.open(fp)
        with pil_img:
            w, h = pil_img.size                  # header only: nothing decoded yet
            limit = MAX_PIXELS_JPEG if pil_img.format == "JPEG" else MAX_PIXELS_OTHER
            if w * h > limit:
                raise ImageTooLarge(
                    f"That image is too large to process ({w}x{h} pixels). "
                    "Please upload a normal camera photo.")
            pil_img.draft("RGB", (MAX_WORK_SIDE, MAX_WORK_SIDE))
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.asarray(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImageTooLarge:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageTooLarge("That image is too large to process. "
                            "Please upload a normal camera photo.") from exc
    except Exception:
        return None


def _normalise(img: np.ndarray) -> np.ndarray:
    """3-channel BGR, longest side capped at MAX_WORK_SIDE. Board finding is
    the slow stage and scales with pixel count; the model only ever sees a
    640px crop."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if max(h, w) > MAX_WORK_SIDE:
        k = MAX_WORK_SIDE / float(max(h, w))
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
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


_NOTE_SQUARE_RE = re.compile(r"^([a-h])([1-8])(?=:)")


def _rotate_note_square(note: str) -> str:
    """Remap the leading square name of a sanity note by 180°.

    Backend notes are written in the backend's frame; when the caller asked
    for black orientation the grid is rotated, so e8 becomes d1, etc.
    """
    def sub(m: "re.Match[str]") -> str:
        file_idx = FILES.index(m.group(1))
        rank = int(m.group(2))
        return f"{FILES[7 - file_idx]}{9 - rank}"

    return _NOTE_SQUARE_RE.sub(sub, note)


def analyze_image(
    source,
    *,
    orientation: str = "white",
    side_to_move: str = "w",
    backend: VisionBackend | None = None,
    corners=None,
) -> VisionResult:
    """Run the vision pipeline on an image.

    `source` is a file path or a BGR numpy image. `corners`, if given, are
    the user's marking of the board: 4 (x, y) points in [0, 1] coordinates of
    the upright photo. On any failure -- bad image, no board, or low overall
    confidence -- this returns `fallback_result()` (starting position,
    detection_failed=True).
    """
    backend = backend or _get_default_backend()
    try:
        img = _load_bgr(source)
        if min(img.shape[:2]) < 64:
            return fallback_result(side_to_move)

        out = backend.analyze(img) if corners is None else backend.analyze(img, corners=corners)
        if not out.board_found:
            return fallback_result(side_to_move)

        grid = out.grid
        conf = out.conf
        notes = list(out.notes)
        if orientation == "black":
            grid = rotate_grid_180(grid)
            conf = rotate_grid_180(conf)
            notes = [_rotate_note_square(n) for n in notes]

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
            notes=notes,
            board_corners=out.corners,
        )
    except Exception:
        return fallback_result(side_to_move)

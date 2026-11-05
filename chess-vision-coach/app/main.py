"""FastAPI web app: upload a chessboard photo, correct the parsed position,
and get Stockfish's best move.

Run a SINGLE uvicorn worker -- session state lives in process memory
(see app/session.py).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import threading
import uuid
from pathlib import Path

import chess
import chess.svg
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import engine
from app.session import COOKIE_NAME, store
from chess_vision import render, vision

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
BOARD_SIZE = 480
# Uploads older than this are deleted by the background cleanup task.
_UPLOAD_MAX_AGE_HOURS = 24

# Per-square confidence thresholds for tinting the board.
CONF_CHECK = 0.55  # below this: red tint ("definitely check this square")
CONF_MAYBE = 0.80  # below this: amber tint ("uncertain")

# Engine strength presets -> (depth, time-limit seconds).
ENGINE_LEVELS = {
    "fast": (12, 0.4),
    "normal": (18, 1.0),
    "strong": (22, 2.0),
}

PIECE_CODES = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}

# Per-session minimum gap between /best-move calls. Stockfish at "strong"
# burns real CPU; HTMX can re-fire on each interaction. This bounds the
# damage one client can do without making the normal flow feel sluggish.
ENGINE_MIN_INTERVAL_S = 0.6

# How many vision analyses may run at once. Each one already uses every core
# (onnxruntime + OpenCV thread pools), so running more in parallel just
# oversubscribes the CPU -- measured with 4 concurrent users and no limit,
# /analyze p95 hit 21 s. Extra requests wait their turn (asyncio, so the
# event loop and every other endpoint stay responsive).
VISION_CONCURRENCY = max(1, int(os.environ.get("CVC_VISION_CONCURRENCY", "1")))
_vision_slots = asyncio.Semaphore(VISION_CONCURRENCY)

app = FastAPI(title="Chess Vision Coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _cleanup_old_uploads() -> None:
    """Delete uploads older than _UPLOAD_MAX_AGE_HOURS. Runs in a daemon thread."""
    cutoff = time.time() - _UPLOAD_MAX_AGE_HOURS * 3600
    for f in UPLOADS_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass


def _start_cleanup_thread() -> None:
    """Run cleanup once at startup, then every hour in the background."""
    def _loop():
        while True:
            _cleanup_old_uploads()
            time.sleep(3600)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


_start_cleanup_thread()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def fen_problem(fen: str | None) -> str | None:
    """Return a human-readable problem string if the FEN is not a legal
    position, or None if it is legal. Uses chess.Board's own validation."""
    if not fen:
        return "No position yet -- analyze a photo first."
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return f"The FEN is malformed: {exc}"

    status = board.status()
    if status == chess.STATUS_VALID:
        return None

    checks = [
        (chess.STATUS_EMPTY, "the board is empty"),
        (chess.STATUS_NO_WHITE_KING, "there is no white king"),
        (chess.STATUS_NO_BLACK_KING, "there is no black king"),
        (chess.STATUS_TOO_MANY_KINGS, "there is more than one king of a color"),
        (chess.STATUS_PAWNS_ON_BACKRANK, "a pawn is on the 1st or 8th rank"),
        (chess.STATUS_TOO_MANY_WHITE_PIECES, "too many white pieces"),
        (chess.STATUS_TOO_MANY_BLACK_PIECES, "too many black pieces"),
        (chess.STATUS_TOO_MANY_WHITE_PAWNS, "too many white pawns"),
        (chess.STATUS_TOO_MANY_BLACK_PAWNS, "too many black pawns"),
        (chess.STATUS_OPPOSITE_CHECK, "the side not to move is in check"),
    ]
    found = [msg for flag, msg in checks if status & flag]
    if not found:
        found = ["the position is illegal"]
    return "Invalid position: " + "; ".join(found) + "."


def confidence_fill(psc: dict | None) -> dict:
    """Map low-confidence squares to a tint color for the board SVG."""
    fill: dict = {}
    for name, conf in (psc or {}).items():
        try:
            square = chess.parse_square(name)
        except ValueError:
            continue
        if conf < CONF_CHECK:
            fill[square] = "#e0413188"
        elif conf < CONF_MAYBE:
            fill[square] = "#f0a93188"
    return fill


def board_view(session, *, arrows: list | None = None) -> dict:
    """Board-fragment context: the rendered SVG (with low-confidence squares
    tinted) plus how many squares are still flagged for the user to check."""
    fen = session.fen or ""
    fill = confidence_fill(session.per_square_confidence)
    svg = ""
    if fen:
        svg = render.board_svg(
            fen,
            size=BOARD_SIZE,
            coordinates=True,
            arrows=arrows or [],
            fill=fill,
        )
    return {"board_svg": svg, "low_conf_count": len(fill)}


def _session_from_request(request: Request):
    sid = request.cookies.get(COOKIE_NAME)
    return store.get_or_create(sid)


def _with_cookie(response: Response, sid: str) -> Response:
    response.set_cookie(COOKIE_NAME, sid, httponly=True, samesite="lax")
    return response


def _analyze_context(request: Request, session) -> dict:
    ctx = {
        "request": request,
        "fen": session.fen or "",
        "image_url": session.original_image_url or "",
        "orientation": session.orientation,
        "side_to_move": session.side_to_move,
        "detection_failed": session.detection_failed,
        "confidence": session.confidence,
        "fen_error": fen_problem(session.fen),
        "vision_notes": session.vision_notes or [],
        "board_corners_json": json.dumps(session.board_corners) if session.board_corners else "",
        "corners_by_user": session.corners_by_user,
        "corner_error": None,
    }
    ctx.update(board_view(session))
    return ctx


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def parse_corners(raw: str | None) -> list[list[float]] | None:
    """Validate user-dragged board corners from the form.

    Expects JSON: 4 [x, y] pairs in [0, 1] photo coordinates (any order).
    Returns them clamped to [0, 1], or None when absent. Raises ValueError
    for anything that can't be a board outline: wrong shape, non-numbers,
    far outside the photo, or a degenerate (tiny / self-crossing) quad.
    """
    if raw is None or not raw.strip():
        return None
    try:
        pts = json.loads(raw)
    except ValueError as exc:
        raise ValueError("corners are not valid JSON") from exc
    if (not isinstance(pts, list) or len(pts) != 4
            or not all(isinstance(p, list) and len(p) == 2 for p in pts)):
        raise ValueError("need exactly 4 corner points")
    out = []
    for x, y in pts:
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("corner coordinates must be numbers")
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("corner coordinates must be numbers") from exc
        if not (math.isfinite(x) and math.isfinite(y)) or not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
            raise ValueError("corners must lie on the photo")
        out.append([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
    # Order around the centroid, then require a convex, non-trivial area.
    cx = sum(p[0] for p in out) / 4
    cy = sum(p[1] for p in out) / 4
    ring = sorted(out, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    area = 0.5 * abs(sum(ring[i][0] * ring[(i + 1) % 4][1] - ring[(i + 1) % 4][0] * ring[i][1]
                         for i in range(4)))
    if area < 0.01:
        raise ValueError("the marked board is too small")
    signs = set()
    for i in range(4):
        a, b, c = ring[i], ring[(i + 1) % 4], ring[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            signs.add(cross > 0)
    if len(signs) > 1:
        raise ValueError("the marked corners don't form a board shape")
    return out


def _discard_upload(name: str | None) -> None:
    if name:
        try:
            (UPLOADS_DIR / name).unlink()
        except OSError:
            pass


@app.get("/photo")
def photo(request: Request) -> Response:
    """The current session's uploaded photo -- and only that.

    Uploads are not a public static folder: on a shared deployment anyone
    could otherwise fetch other people's photos by URL.
    """
    session = store.get(request.cookies.get(COOKIE_NAME))
    name = session.upload_name if session else None
    path = UPLOADS_DIR / name if name else None
    if path is None or not path.is_file():
        return PlainTextResponse("Not found", status_code=404)
    return FileResponse(
        path, media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    sid, _session = _session_from_request(request)
    response = templates.TemplateResponse(request, "index.html", {"error": None})
    return _with_cookie(response, sid)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile | None = File(None),
    orientation: str = Form("white"),
    side_to_move: str = Form("white"),
    corners: str | None = Form(None),
) -> Response:
    sid, session = _session_from_request(request)
    orientation = "black" if orientation == "black" else "white"
    stm = "b" if side_to_move == "black" else "w"
    try:
        user_corners = parse_corners(corners)
    except ValueError as exc:
        user_corners = None
        corner_error = f"Couldn't use those corners: {exc}. Drag the 4 dots onto the board's corners."
    else:
        corner_error = None

    def upload_error(message: str) -> Response:
        resp = templates.TemplateResponse(
            request, "_upload.html", {"error": message}
        )
        return _with_cookie(resp, sid)

    # New upload, or re-analyze the previously uploaded image.
    if image is not None and image.filename:
        content = await image.read()
        if len(content) > MAX_UPLOAD_BYTES:
            return upload_error("That image is larger than 10 MB. Please upload a smaller file.")
        if image.content_type and not image.content_type.startswith("image/"):
            return upload_error("That file is not an image. Please upload a JPG or PNG photo.")

        # Decode with EXIF orientation applied (phones store photos sideways
        # and record the rotation in EXIF) and size capped -- the same decoder
        # the pipeline uses for files. CPU-bound, so off the event loop:
        # running it inline would freeze every other user's requests.
        async with _vision_slots:
            decoded = await run_in_threadpool(vision.decode_image_bytes, content)
        if decoded is None:
            return upload_error("That image could not be read -- it may be corrupt or not a real image.")

        # Persist the upright, size-capped pixels so the thumbnail matches
        # what the pipeline analyses (and storage stays ~0.5 MB per photo).
        # The name is random and unrelated to the session id: it appears in
        # page HTML, so it must not double as a session credential.
        fname = f"{uuid.uuid4().hex}.jpg"
        await run_in_threadpool(
            cv2.imwrite, str(UPLOADS_DIR / fname), decoded,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        _discard_upload(session.upload_name)   # a new photo replaces the old one
        session.upload_name = fname
        session.original_image_url = f"/photo?v={fname[:12]}"
        session.board_corners, session.corners_by_user = None, False

    image_path = UPLOADS_DIR / session.upload_name if session.upload_name else None
    if image_path is None or not image_path.exists():
        return upload_error("Please choose a chessboard photo to analyze.")

    if corner_error:
        resp = templates.TemplateResponse(
            request, "_analyze.html",
            {**_analyze_context(request, session), "corner_error": corner_error})
        return _with_cookie(resp, sid)
    if user_corners is None and session.corners_by_user and not (image is not None and image.filename):
        user_corners = session.board_corners   # orientation change etc.: keep the user's outline

    # The vision pipeline is ~0.5 s of all-core CPU: run it off the event
    # loop (so other users' requests keep flowing) and behind the vision
    # admission limit (so concurrent uploads queue instead of thrashing).
    async with _vision_slots:
        result = await run_in_threadpool(
            vision.analyze_image, str(image_path),
            orientation=orientation, side_to_move=stm, corners=user_corners,
        )
    session.fen = result.fen
    session.orientation = orientation
    session.side_to_move = side_to_move
    session.detection_failed = result.detection_failed
    session.confidence = result.confidence
    session.per_square_confidence = result.per_square_confidence
    session.vision_notes = result.notes or None
    session.board_corners = result.board_corners or user_corners
    session.corners_by_user = user_corners is not None

    resp = templates.TemplateResponse(
        request, "_analyze.html", _analyze_context(request, session)
    )
    return _with_cookie(resp, sid)


@app.post("/correct", response_class=HTMLResponse)
def correct(
    request: Request,
    square: str = Form(...),
    piece: str = Form(...),
) -> Response:
    sid, session = _session_from_request(request)
    if not session.fen:
        resp = templates.TemplateResponse(
            request, "_analyze.html", _analyze_context(request, session)
        )
        return _with_cookie(resp, sid)

    parts = session.fen.split()
    try:
        target = chess.parse_square(square.strip().lower())
    except ValueError:
        target = None

    if target is not None:
        base = chess.BaseBoard(parts[0])
        if piece == "empty":
            base.remove_piece_at(target)
        elif piece in PIECE_CODES:
            base.set_piece_at(target, chess.Piece.from_symbol(PIECE_CODES[piece]))
        parts[0] = base.board_fen()
        session.fen = " ".join(parts)
        # The user has just vouched for this square -- it is no longer shaky.
        if session.per_square_confidence is not None:
            session.per_square_confidence[square.strip().lower()] = 1.0

    context = {
        "request": request,
        "fen": session.fen,
        "fen_error": fen_problem(session.fen),
    }
    context.update(board_view(session))
    resp = templates.TemplateResponse(request, "_correct.html", context)
    return _with_cookie(resp, sid)


@app.post("/best-move", response_class=HTMLResponse)
def best_move(request: Request, level: str = Form("normal")) -> Response:
    sid, session = _session_from_request(request)
    depth, time_limit = ENGINE_LEVELS.get(level, ENGINE_LEVELS["normal"])

    def panel(*, result=None, error=None, arrows=None) -> Response:
        ctx = {"request": request, "result": result, "error": error}
        ctx.update(board_view(session, arrows=arrows))
        resp = templates.TemplateResponse(request, "_bestmove.html", ctx)
        return _with_cookie(resp, sid)

    if not session.fen:
        return panel(error="Analyze a photo before asking for the best move.")

    now = time.monotonic()
    wait = ENGINE_MIN_INTERVAL_S - (now - session.last_engine_request)
    if wait > 0:
        return panel(error=f"Slow down a touch -- try again in {wait:.1f} s.")
    session.last_engine_request = now

    problem = fen_problem(session.fen)
    if problem:
        return panel(error=problem)

    try:
        result = engine.analyse(
            session.fen, depth=depth, time_limit=time_limit,
            multipv=3, timeout=6.0,
        )
    except engine.EngineError as exc:
        return panel(error=str(exc))

    arrows = []
    if result.best_move_uci:
        mv = chess.Move.from_uci(result.best_move_uci)
        arrows = [chess.svg.Arrow(mv.from_square, mv.to_square, color="#2e7d32")]

    return panel(result=result, arrows=arrows)


@app.post("/play-move", response_class=HTMLResponse)
def play_move(request: Request, move: str = Form(...)) -> Response:
    """Apply a UCI move (e.g. the engine's suggestion) to the position."""
    sid, session = _session_from_request(request)
    if session.fen:
        try:
            board = chess.Board(session.fen)
            mv = chess.Move.from_uci(move)
            if mv in board.legal_moves:
                board.push(mv)
                session.fen = board.fen()
                # The position changed; the old per-square confidence and
                # vision repair notes are stale.
                session.per_square_confidence = None
                session.vision_notes = None
        except ValueError:
            pass

    context = {
        "request": request,
        "fen": session.fen or "",
        "fen_error": fen_problem(session.fen),
    }
    context.update(board_view(session))
    resp = templates.TemplateResponse(request, "_correct.html", context)
    return _with_cookie(resp, sid)


@app.get("/download-fen", response_class=PlainTextResponse)
def download_fen(request: Request) -> Response:
    _sid, session = _session_from_request(request)
    fen = session.fen or vision.STARTING_PLACEMENT + " w - - 0 1"
    return PlainTextResponse(
        fen,
        headers={"Content-Disposition": 'attachment; filename="position.fen"'},
    )

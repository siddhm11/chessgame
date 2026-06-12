"""FastAPI web app: upload a chessboard photo, correct the parsed position,
and get Stockfish's best move.

Run a SINGLE uvicorn worker -- session state lives in process memory
(see app/session.py).
"""

from __future__ import annotations

import time
import threading
import uuid
from pathlib import Path

import chess
import chess.svg
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
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

app = FastAPI(title="Chess Vision Coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
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
    }
    ctx.update(board_view(session))
    return ctx


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


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
) -> Response:
    sid, session = _session_from_request(request)
    orientation = "black" if orientation == "black" else "white"
    stm = "b" if side_to_move == "black" else "w"

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

        # Decode + EXIF-rotate-into-pixels via Pillow. Phones encode
        # orientation in EXIF instead of rotating the sensor data; OpenCV
        # would feed the pipeline a sideways board.
        try:
            from PIL import Image, ImageOps
            from io import BytesIO

            with Image.open(BytesIO(content)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                rgb = np.asarray(pil_img.convert("RGB"))
            decoded = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return upload_error("That image could not be read -- it may be corrupt or not a real image.")

        # Persist the upright pixels so the thumbnail in the UI matches what
        # the pipeline actually analyses. JPEG keeps file size reasonable.
        suffix = ".jpg"
        fname = f"{sid}_{uuid.uuid4().hex[:8]}{suffix}"
        cv2.imwrite(str(UPLOADS_DIR / fname), decoded, [cv2.IMWRITE_JPEG_QUALITY, 90])
        session.original_image_url = f"/uploads/{fname}"

    image_path = None
    if session.original_image_url:
        image_path = UPLOADS_DIR / Path(session.original_image_url).name
    if image_path is None or not image_path.exists():
        return upload_error("Please choose a chessboard photo to analyze.")

    result = vision.analyze_image(
        str(image_path), orientation=orientation, side_to_move=stm
    )
    session.fen = result.fen
    session.orientation = orientation
    session.side_to_move = side_to_move
    session.detection_failed = result.detection_failed
    session.confidence = result.confidence
    session.per_square_confidence = result.per_square_confidence
    session.vision_notes = result.notes or None

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

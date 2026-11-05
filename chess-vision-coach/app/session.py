"""In-memory session store.

Decision Point 2(b): a plain in-memory dict keyed by a cookie UUID. This is
fine for SINGLE-USER LOCAL use, which is the intended deployment. Tradeoffs:
  - state is lost when the server restarts;
  - it is NOT shared across worker processes, so uvicorn must run with a
    single worker (run.sh does);
  - it would grow unbounded, so we cap it with simple LRU eviction.
For multi-user or multi-process deployment this would need Redis or similar.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

COOKIE_NAME = "cvc_session"
_SID = re.compile(r"[0-9a-f]{32}")


@dataclass
class Session:
    """Per-user analysis state."""

    fen: str | None = None
    original_image_url: str | None = None
    # File name of this session's photo inside UPLOADS_DIR. Random, never
    # derived from the session id, and only ever served back to this session.
    upload_name: str | None = None
    orientation: str = "white"
    side_to_move: str = "w"
    detection_failed: bool = False
    confidence: float = 0.0
    # Per-square confidence from the vision pass, used to tint shaky squares.
    # Set to None once the position is changed by playing a move.
    per_square_confidence: dict | None = None
    # Auto-repair notes from the vision sanity layer ("e8: ... reread Q as K").
    vision_notes: list | None = None
    # Board outline on the photo: 4 [x, y] in [0, 1], TL, TR, BR, BL.
    board_corners: list | None = None
    # True once the user dragged the corners; re-analyses then reuse them.
    corners_by_user: bool = False
    # Settings of the analysis currently shown; an identical re-analysis is free.
    analysis_key: str | None = None
    # An analysis for this session is running; a second one is refused.
    analyzing: bool = False
    # Monotonic time of the most recent /best-move call; used to throttle.
    last_engine_request: float = 0.0


class SessionStore:
    def __init__(self, max_sessions: int = 4096) -> None:
        # A session is a few hundred bytes (its photo lives on disk), so the
        # cap is about abuse, not memory. Only uploads create sessions.
        self._data: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions
        self.on_evict = None   # callback(session) -- e.g. delete its photo

    def create(self, sid: str | None = None) -> tuple[str, Session]:
        sid = sid or uuid.uuid4().hex
        session = Session()
        with self._lock:
            self._data[sid] = session
            self._evict()
        return sid, session

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self._lock:
            session = self._data.get(sid)
            if session is not None:
                self._data.move_to_end(sid)
            return session

    def get_or_create(self, sid: str | None) -> tuple[str, Session]:
        session = self.get(sid)
        if session is not None and sid is not None:
            return sid, session
        # A well-formed id we don't know (server restarted -- e.g. a Space
        # woke from sleep -- or the session was evicted): adopt it, so the
        # open page, which keeps sending that id, keeps working instead of
        # getting a fresh empty session on every click.
        if sid is not None and _SID.fullmatch(sid):
            return self.create(sid)
        return self.create()

    def _evict(self) -> None:
        while len(self._data) > self._max:
            _, gone = self._data.popitem(last=False)
            if self.on_evict is not None:
                try:
                    self.on_evict(gone)
                except Exception:
                    pass


store = SessionStore(int(os.environ.get("CVC_MAX_SESSIONS", "4096")))

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

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

COOKIE_NAME = "cvc_session"


@dataclass
class Session:
    """Per-user analysis state."""

    fen: str | None = None
    original_image_url: str | None = None
    orientation: str = "white"
    side_to_move: str = "w"
    detection_failed: bool = False
    confidence: float = 0.0
    # Per-square confidence from the vision pass, used to tint shaky squares.
    # Set to None once the position is changed by playing a move.
    per_square_confidence: dict | None = None
    # Auto-repair notes from the vision sanity layer ("e8: ... reread Q as K").
    vision_notes: list | None = None


class SessionStore:
    def __init__(self, max_sessions: int = 256) -> None:
        self._data: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self) -> tuple[str, Session]:
        sid = uuid.uuid4().hex
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
        return self.create()

    def _evict(self) -> None:
        while len(self._data) > self._max:
            self._data.popitem(last=False)


store = SessionStore()

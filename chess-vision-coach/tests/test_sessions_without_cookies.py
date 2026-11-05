"""Sessions work with the header the page sends, with no cookies at all.

Inside a Hugging Face Space page the app is a cross-site iframe, where
browsers don't send SameSite=Lax cookies. The page therefore sends its
session id in a header on every htmx request (hx-headers on <body>).
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import SESSION_HEADER, app
from app.session import store

FIXTURES = Path(__file__).parent / "fixtures"


def _cookieless():
    c = TestClient(app)
    c.cookies.clear()
    return c


def test_index_page_embeds_a_session_header_without_allocating():
    """GET / hands the page an id but stores nothing: only an upload creates
    a session, so crawlers and link previews can't evict real users."""
    page = TestClient(app).get("/").text
    m = re.search(r"hx-headers='\{\"X-CVC-Session\": \"([0-9a-f]{32})\"\}'", page)
    assert m, "body must carry hx-headers with the session id"
    assert store.get(m.group(1)) is None


def test_whole_flow_with_header_only_no_cookies():
    sid = re.search(r'"X-CVC-Session": "([0-9a-f]{32})"', TestClient(app).get("/").text).group(1)
    h = {SESSION_HEADER: sid}
    c = _cookieless()
    with open(FIXTURES / "starting.png", "rb") as fh:
        c.cookies.clear()
        r = c.post("/analyze", files={"image": ("s.png", fh, "image/png")},
                   data={"orientation": "white", "side_to_move": "white"}, headers=h)
    assert "rnbqkbnr/pppppppp" in r.text
    c.cookies.clear()
    r = c.post("/correct", data={"square": "e2", "piece": "empty"}, headers=h)
    c.cookies.clear()
    r = c.post("/analyze", data={"orientation": "white", "side_to_move": "white"}, headers=h)
    assert "Please choose a chessboard photo" not in r.text   # the photo was remembered


def test_unknown_but_wellformed_id_is_adopted_after_restart():
    """A Space that slept and restarted has forgotten all sessions; the open
    page keeps sending its old id. It must keep one session, not get a new
    empty one on every click."""
    sid = "ab" * 16
    assert store.get(sid) is None
    c = _cookieless()
    c.get("/", headers={SESSION_HEADER: sid})
    with open(FIXTURES / "starting.png", "rb") as fh:
        c.post("/analyze", files={"image": ("s.png", fh, "image/png")},
               data={"orientation": "white", "side_to_move": "white"},
               headers={SESSION_HEADER: sid})
    assert store.get(sid).fen and store.get(sid).fen.startswith("rnbqkbnr")


def test_malformed_session_ids_are_not_adopted():
    for bad in ["x", "../../etc", "A" * 32, "ab" * 17, ""]:
        c = _cookieless()
        r = c.get("/", headers={SESSION_HEADER: bad} if bad else {})
        new = re.search(r'"X-CVC-Session": "([0-9a-f]{32})"', r.text).group(1)
        assert new != bad and store.get(bad) is None

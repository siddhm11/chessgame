"""Uploaded photos stay private, and sessions work without cookies.

Two things used to leak on a public deployment:
- photos were a public static folder (/uploads/...);
- the file name started with the session id, so the thumbnail URL doubled
  as a session credential.
Now each photo has its own random 128-bit token (/photo/<token>), unrelated
to the session id; nothing can be listed or guessed. It is a capability link
-- not a cookie check -- because inside a Hugging Face Space page (a
cross-site iframe) cookies aren't sent at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import UPLOADS_DIR, app
from app.session import COOKIE_NAME

FIXTURES = Path(__file__).parent / "fixtures"


def _upload(client: TestClient, name: str = "starting.png"):
    with open(FIXTURES / name, "rb") as fh:
        return client.post(
            "/analyze",
            files={"image": (name, fh, "image/png")},
            data={"orientation": "white", "side_to_move": "white"},
        )


def test_photo_link_never_contains_the_session_id():
    client = TestClient(app)
    resp = _upload(client)
    sid = client.cookies.get(COOKIE_NAME)
    assert sid and resp.status_code == 200
    assert sid not in resp.text                      # the analysis fragment
    link = re.search(r'src="(/photo/[0-9a-f]{32})"', resp.text).group(1)
    assert sid not in link
    assert "/uploads/" not in resp.text


def test_photo_link_works_without_cookies_and_nothing_is_guessable():
    owner = TestClient(app)
    link = re.search(r'src="(/photo/[0-9a-f]{32})"', _upload(owner).text).group(1)
    resp = TestClient(app).get(link)                 # no cookie at all (iframe case)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:2] == b"\xff\xd8"
    assert "private" in resp.headers.get("cache-control", "")
    stranger = TestClient(app)
    for bad in ["/photo/" + "0" * 32, "/photo/abc", "/photo/..%2Fapp%2Fmain.py",
                "/photo/" + "A" * 32, "/photo", "/photo/"]:
        assert stranger.get(bad).status_code == 404, bad


def test_upload_folder_is_not_publicly_mounted():
    client = TestClient(app)
    _upload(client)
    files = sorted(UPLOADS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    newest = files[-1].name
    assert TestClient(app).get(f"/uploads/{newest}").status_code == 404
    assert client.get(f"/uploads/{newest}").status_code == 404


def test_new_upload_replaces_and_deletes_previous_photo():
    client = TestClient(app)
    _upload(client, "starting.png")
    before = set(UPLOADS_DIR.glob("*.jpg"))
    _upload(client, "midgame.png")
    after = set(UPLOADS_DIR.glob("*.jpg"))
    assert len(after - before) == 1, "exactly one new file"
    assert len(before - after) == 1, "the session's previous photo was deleted"


def test_reanalyze_without_new_upload_still_uses_stored_photo():
    client = TestClient(app)
    _upload(client)
    resp = client.post("/analyze", data={"orientation": "black", "side_to_move": "white"})
    assert resp.status_code == 200
    assert "Please choose a chessboard photo" not in resp.text
    # Same stored photo, re-read as photographed from Black's side: the
    # position is rotated 180 degrees.
    assert "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr" in resp.text

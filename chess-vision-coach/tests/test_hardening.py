"""Protections a public deployment needs, each pinned so removing it fails.

Found by an independent audit with measurements: oversized bodies were fully
buffered (1.5 GB -> 1.66 GB RSS), sessions were free so 60 of them ran 40
Stockfish searches at once, 256 bare page views evicted every real user,
and thread pools were sized to the host's cores instead of the container's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.session import Session, SessionStore
from chess_vision import vision

FIXTURES = Path(__file__).parent / "fixtures"


def _upload(client):
    with open(FIXTURES / "starting.png", "rb") as fh:
        return client.post("/analyze", files={"image": ("s.png", fh, "image/png")},
                           data={"orientation": "white", "side_to_move": "white"})


def test_chunked_oversized_body_is_cut_off_not_buffered():
    """No Content-Length (chunked): the middleware counts bytes and stops."""
    limit = main.MAX_UPLOAD_BYTES + main._BODY_SLACK

    def gen():
        chunk = b"0" * (1024 * 1024)
        for _ in range(limit // len(chunk) + 3):
            yield chunk

    resp = TestClient(app=main.app).post(
        "/analyze", content=gen(),
        headers={"content-type": "multipart/form-data; boundary=x"})
    assert resp.status_code in (400, 413)


def test_engine_is_limited_globally_not_per_session(monkeypatch):
    """With every engine slot taken, a new request is told the engine is
    busy instead of starting yet another Stockfish process."""
    monkeypatch.setattr(main, "ENGINE_QUEUE_TIMEOUT_S", 0.2)
    client = TestClient(main.app)
    _upload(client)
    sem = main._engine_slots
    taken = 0
    while sem._value > 0:            # occupy every slot (as other users would)
        sem._value -= 1
        taken += 1
    try:
        resp = client.post("/best-move", data={"level": "fast"})
    finally:
        sem._value += taken
    assert "Stockfish is busy" in resp.text


def test_second_concurrent_analysis_of_same_session_is_refused():
    client = TestClient(main.app)
    _upload(client)
    sid = client.cookies.get("cvc_session")
    main.store.get(sid).analyzing = True
    try:
        resp = client.post("/analyze", data={"orientation": "black", "side_to_move": "white"})
    finally:
        main.store.get(sid).analyzing = False
    assert resp.status_code == 204


def test_identical_reanalysis_does_not_rerun_vision(monkeypatch):
    client = TestClient(main.app)
    _upload(client)
    calls = []
    real = vision.analyze_image
    monkeypatch.setattr(vision, "analyze_image", lambda *a, **k: calls.append(1) or real(*a, **k))
    client.post("/analyze", data={"orientation": "white", "side_to_move": "white"})
    assert calls == [], "same photo + same settings must reuse the analysis"
    client.post("/analyze", data={"orientation": "black", "side_to_move": "white"})
    assert calls == [1], "a real change re-runs it"


@pytest.mark.parametrize("route, data", [
    ("/correct", {"square": "e2", "piece": "empty"}),
    ("/play-move", {"move": "e2e4"}),
    ("/best-move", {"level": "fast"}),
])
def test_expired_session_asks_to_reupload(route, data):
    resp = TestClient(main.app).post(route, data=data, headers={main.SESSION_HEADER: "cd" * 16})
    assert resp.status_code == 200
    assert "upload your photo again" in resp.text
    assert resp.headers.get("HX-Retarget") == "#app"


def test_download_fen_never_invents_a_position():
    assert TestClient(main.app).get("/download-fen").status_code == 404


def test_bare_page_views_do_not_evict_real_sessions(monkeypatch):
    monkeypatch.setattr(main.store, "_max", 20)       # small cap: eviction would show
    real = TestClient(main.app)
    _upload(real)
    sid = real.cookies.get("cvc_session")
    for _ in range(60):
        TestClient(main.app).get("/")
    assert main.store.get(sid) is not None and main.store.get(sid).fen


def test_evicted_session_photo_is_deleted(tmp_path):
    store = SessionStore(max_sessions=2)
    gone = []
    store.on_evict = lambda s: gone.append(s.upload_name)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _, s = store.create()
        s.upload_name = name
    assert gone == ["a.jpg"]


def test_thread_count_respects_override(monkeypatch):
    monkeypatch.setenv("CVC_THREADS", "2")
    assert vision.effective_cpus() == 2
    monkeypatch.delenv("CVC_THREADS")
    assert 1 <= vision.effective_cpus() <= (__import__("os").cpu_count() or 1)


def test_jpeg_budget_allows_big_phone_photos_but_png_budget_is_tight():
    assert vision.MAX_PIXELS_JPEG >= 200_000_000      # 200 MP phone modes decode via draft
    assert vision.MAX_PIXELS_OTHER <= 50_000_000      # full-size decodes stay bounded

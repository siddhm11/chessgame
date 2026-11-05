"""End-to-end route tests with FastAPI's TestClient."""

from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from app import engine
from app.main import app, confidence_fill

FIXTURES = Path(__file__).parent / "fixtures"
STARTING_OPENINGS = {"e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "g2g3"}


def _client() -> TestClient:
    return TestClient(app)


def _upload_starting(client: TestClient):
    with open(FIXTURES / "starting.png", "rb") as fh:
        return client.post(
            "/analyze",
            files={"image": ("starting.png", fh, "image/png")},
            data={"orientation": "white", "side_to_move": "white"},
        )


def test_index_serves_upload_form():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert 'hx-post="/analyze"' in resp.text
    assert 'name="image"' in resp.text


def test_analyze_parses_starting_position():
    client = _client()
    resp = _upload_starting(client)
    assert resp.status_code == 200
    assert "board-container" in resp.text
    assert "rnbqkbnr/pppppppp" in resp.text


def test_analyze_rejects_non_image():
    resp = _client().post(
        "/analyze",
        files={"image": ("notes.txt", b"hello world", "text/plain")},
        data={"orientation": "white", "side_to_move": "white"},
    )
    assert resp.status_code == 200
    assert "not an image" in resp.text or "could not be read" in resp.text


def test_analyze_rejects_corrupt_image():
    resp = _client().post(
        "/analyze",
        files={"image": ("x.png", b"\x89PNG not really", "image/png")},
        data={"orientation": "white", "side_to_move": "white"},
    )
    assert resp.status_code == 200
    assert "could not be read" in resp.text


def test_analyze_rejects_oversized_image():
    """Refused from the declared size, before the body is read; htmx gets a
    readable message (it only swaps 2xx), other clients a 413."""
    from app.main import MAX_UPLOAD_BYTES
    big = b"\x89PNG" + b"0" * (MAX_UPLOAD_BYTES + 300 * 1024)
    resp = _client().post(
        "/analyze",
        files={"image": ("big.png", big, "image/png")},
        data={"orientation": "white", "side_to_move": "white"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "larger than 25 MB" in resp.text
    resp = _client().post("/analyze", files={"image": ("big.png", big, "image/png")})
    assert resp.status_code == 413


def test_analyze_rejects_image_bomb():
    """A small file declaring a huge pixel count is refused from its header;
    it must not reach a decoder (a 1 MB PNG bomb used to decode to 6 GB)."""
    import struct, zlib
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    w = h = 20000                                            # 400 MP, header only
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\0" * 1000)) + chunk(b"IEND", b""))
    resp = _client().post(
        "/analyze", files={"image": ("bomb.png", png, "image/png")},
        data={"orientation": "white", "side_to_move": "white"})
    assert resp.status_code == 200
    assert "too large to process" in resp.text


def test_correct_updates_fen():
    client = _client()
    _upload_starting(client)
    resp = client.post("/correct", data={"square": "e2", "piece": "empty"})
    assert resp.status_code == 200
    # e2 pawn removed -> rank 2 becomes PPPP1PPP
    assert "PPPP1PPP" in resp.text
    assert 'hx-swap-oob="true"' in resp.text


def test_correct_to_illegal_position_warns():
    client = _client()
    _upload_starting(client)
    client.post("/correct", data={"square": "e1", "piece": "empty"})
    resp = client.post("/correct", data={"square": "d1", "piece": "empty"})
    assert "no white king" in resp.text


def test_best_move_without_a_session_asks_to_reupload():
    """No live session (never uploaded, or the server restarted): replace the
    whole view with the upload form and say why -- never a made-up board."""
    resp = _client().post("/best-move")
    assert resp.status_code == 200
    assert "upload your photo again" in resp.text
    assert resp.headers.get("HX-Retarget") == "#app"


@pytest.mark.skipif(
    engine.find_stockfish() is None, reason="Stockfish not installed"
)
def test_best_move_on_starting_position():
    client = _client()
    _upload_starting(client)
    resp = client.post("/best-move")
    assert resp.status_code == 200
    assert "Best move" in resp.text
    # The board is swapped back out-of-band with the move arrow.
    assert 'id="board-container"' in resp.text


@pytest.mark.skipif(
    engine.find_stockfish() is None, reason="Stockfish not installed"
)
def test_best_move_suggests_sane_opening():
    client = _client()
    _upload_starting(client)
    result = engine.analyse(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    )
    assert result.best_move_uci in STARTING_OPENINGS


def test_best_move_blocked_on_illegal_fen():
    client = _client()
    _upload_starting(client)
    client.post("/correct", data={"square": "e1", "piece": "empty"})
    resp = client.post("/best-move")
    assert resp.status_code == 200
    assert "Invalid position" in resp.text


def test_download_fen_is_plain_text():
    client = _client()
    _upload_starting(client)
    resp = client.get("/download-fen")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.startswith("rnbqkbnr/pppppppp")


def test_play_move_applies_a_legal_move():
    client = _client()
    _upload_starting(client)
    resp = client.post("/play-move", data={"move": "e2e4"})
    assert resp.status_code == 200
    # After 1.e4 the pawn is on e4 and it is Black to move.
    assert "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b" in resp.text


def test_play_move_ignores_an_illegal_move():
    client = _client()
    _upload_starting(client)
    resp = client.post("/play-move", data={"move": "e2e5"})  # not legal
    assert resp.status_code == 200
    assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in resp.text


@pytest.mark.skipif(
    engine.find_stockfish() is None, reason="Stockfish not installed"
)
def test_best_move_accepts_strength_level():
    client = _client()
    _upload_starting(client)
    resp = client.post("/best-move", data={"level": "fast"})
    assert resp.status_code == 200
    assert "Best move" in resp.text


def test_confidence_fill_tiers_squares():
    fill = confidence_fill({"e4": 0.30, "d4": 0.70, "a1": 0.99})
    assert chess.parse_square("e4") in fill  # low confidence -> tinted
    assert chess.parse_square("d4") in fill  # uncertain -> tinted
    assert chess.parse_square("a1") not in fill  # confident -> not tinted
    assert confidence_fill(None) == {}


def test_upload_cleanup_removes_old_files(tmp_path, monkeypatch):
    """Cleanup task must delete files older than the age limit."""
    import time
    from app.main import _cleanup_old_uploads, UPLOADS_DIR, _UPLOAD_MAX_AGE_HOURS

    # Create two fake uploads: one old, one recent.
    old_file = UPLOADS_DIR / "old_upload_test.jpg"
    new_file = UPLOADS_DIR / "new_upload_test.jpg"
    old_file.write_bytes(b"x")
    new_file.write_bytes(b"x")

    # Back-date the old file past the retention window.
    old_mtime = time.time() - (_UPLOAD_MAX_AGE_HOURS + 1) * 3600
    import os
    os.utime(str(old_file), (old_mtime, old_mtime))

    _cleanup_old_uploads()

    assert not old_file.exists(), "cleanup should remove files past the age limit"
    assert new_file.exists(), "cleanup should keep recent files"
    new_file.unlink(missing_ok=True)  # tidy up


@pytest.mark.skipif(
    engine.find_stockfish() is None, reason="Stockfish not installed"
)
def test_best_move_is_rate_limited(monkeypatch):
    """Two /best-move calls in quick succession: the second is throttled."""
    from app import main as app_main
    monkeypatch.setattr(app_main, "ENGINE_MIN_INTERVAL_S", 5.0)

    client = _client()
    _upload_starting(client)
    first = client.post("/best-move")
    assert first.status_code == 200
    assert "Best move" in first.text

    second = client.post("/best-move")
    assert second.status_code == 200
    assert "Slow down" in second.text


def test_reload_shows_your_board_not_a_blank_form():
    client = _client()
    _upload_starting(client)
    client.post("/correct", data={"square": "e2", "piece": "empty"})
    page = client.get("/").text
    assert "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR" in page    # the corrected board
    assert 'id="photo-wrap"' in page
    assert 'hx-headers=' in page                                     # same session continues


def test_first_visit_still_shows_the_upload_form():
    assert 'name="image"' in _client().get("/").text

"""End-to-end route tests with FastAPI's TestClient."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import engine
from app.main import app

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
    big = b"\x89PNG" + b"0" * (10 * 1024 * 1024 + 16)
    resp = _client().post(
        "/analyze",
        files={"image": ("big.png", big, "image/png")},
        data={"orientation": "white", "side_to_move": "white"},
    )
    assert resp.status_code == 200
    assert "10 MB" in resp.text


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


def test_best_move_before_analyze_is_friendly():
    resp = _client().post("/best-move")
    assert resp.status_code == 200
    assert "Analyze a photo" in resp.text


@pytest.mark.skipif(
    engine.find_stockfish() is None, reason="Stockfish not installed"
)
def test_best_move_on_starting_position():
    client = _client()
    _upload_starting(client)
    resp = client.post("/best-move")
    assert resp.status_code == 200
    assert "Best move:" in resp.text
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

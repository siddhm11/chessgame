"""The in-browser build's Python side: ONNX metadata without the onnx
package, the OpenCV DNN runtime (what the browser falls back to), and the
web/src/py/cvc_web.py glue the page calls. Runs natively -- the same code
runs under Pyodide in the browser (web/tools/e2e.mjs covers that)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import chess_vision.model_backend as mb

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "chess_vision" / "models" / "yolov8n640-chess-finetuned.onnx"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="ONNX model not vendored")


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num: int, payload: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def test_onnx_metadata_parser_on_synthetic_model():
    entry = lambda k, v: _field(14, _field(1, k.encode()) + _field(2, v.encode()))  # noqa: E731
    model = (
        _varint(1 << 3 | 0) + _varint(8)          # ir_version = 8 (varint field)
        + _field(7, b"\x00" * 300)                # a "graph" blob to skip
        + entry("imgsz", "[640, 640]")
        + entry("names", "{0: 'white_king'}")
    )
    assert mb._onnx_metadata(model) == {"imgsz": "[640, 640]", "names": "{0: 'white_king'}"}


@needs_model
def test_onnx_metadata_matches_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    expected = dict(ort.InferenceSession(str(MODEL)).get_modelmeta().custom_metadata_map)
    assert mb._onnx_metadata(MODEL.read_bytes()) == expected


@needs_model
def test_opencv_runtime_matches_onnxruntime(monkeypatch):
    pytest.importorskip("onnxruntime")
    ref = mb.ModelBackend(MODEL)
    ref._load()
    monkeypatch.setenv("CVC_RUNTIME", "opencv")
    cv = mb.ModelBackend(MODEL)
    cv._load()
    assert isinstance(cv._session, mb._OpenCVSession)
    assert (cv._imgsz, cv._class_to_symbol) == (ref._imgsz, ref._class_to_symbol)

    blob = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    a = ref._session.run(None, {ref._input_name: blob})[0]
    b = cv._session.run(None, {cv._input_name: blob})[0]
    assert a.shape == b.shape
    assert np.abs(a[0, 4:] - b[0, 4:]).max() < 1e-3      # class scores
    assert np.abs(a[0, :4] - b[0, :4]).max() < 0.05      # box coords, pixels


@pytest.fixture
def cvc_web(monkeypatch):
    if not MODEL.exists():
        pytest.skip("ONNX model not vendored")
    monkeypatch.setenv("CVC_RUNTIME", "opencv")
    spec = importlib.util.spec_from_file_location("cvc_web", ROOT / "web" / "src" / "py" / "cvc_web.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cvc_web", mod)
    spec.loader.exec_module(mod)
    assert mod.init(str(MODEL)) == "opencv"
    return mod


def test_web_glue_rejects_garbage(cvc_web, tmp_path):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not an image")
    assert "error" in json.loads(cvc_web.load_image(str(bad)))
    assert "error" in json.loads(cvc_web.analyze("white", "white", None))


def test_web_glue_full_flow(cvc_web, tmp_path):
    import cv2

    # A plain grey square on a darker margin: no real board, but with user
    # corners the pipeline runs end to end.
    board = np.full((480, 480, 3), 200, np.uint8)
    photo = tmp_path / "p.png"
    cv2.imwrite(str(photo), cv2.copyMakeBorder(board, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(90, 90, 90)))
    assert json.loads(cvc_web.load_image(str(photo)))["ok"]
    res = json.loads(cvc_web.analyze("white", "black", json.dumps([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])))
    assert res["fen"].split()[1] == "b"
    assert len(res["corners"]) == 4
    assert cvc_web.display_jpeg()[:2] == b"\xff\xd8"

    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    fen2 = cvc_web.correct(fen, "d4", "wQ")
    assert fen2.split()[0] == "4k3/8/8/8/3Q4/8/4P3/4K3"
    assert cvc_web.correct(fen2, "d4", "empty").split()[0] == fen.split()[0]
    assert cvc_web.set_side_to_move(fen, "black").split()[1] == "b"
    assert cvc_web.play(fen, "e2e4").split()[0] == "4k3/8/8/8/4P3/8/8/4K3"
    assert cvc_web.play(fen, "e2e5") is None
    assert cvc_web.pv_to_san(fen, ["e2e4", "e8d7", "zzzz"]) == "1. e4 Kd7"
    assert cvc_web.game_over("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1") == "Checkmate"

    v = json.loads(cvc_web.view(fen, json.dumps({"e2": 0.3, "e1": 0.7}), False, "e2e4"))
    assert v["svg"].startswith("<svg") and v["problem"] is None and v["low_conf_count"] == 2
    assert "Invalid position" in json.loads(cvc_web.view("QQQQQQQQ/QQQQQQQQ/8/8/8/8/8/K6k w - - 0 1"))["problem"]

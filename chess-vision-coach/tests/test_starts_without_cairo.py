"""The web app must start and read photos without Cairo.

Cairo is a native library that is often missing on Windows. The web app
only draws SVG, so only PNG rendering (CLI, classical backend) may need it.
Run in a subprocess so the import block can't leak into other tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import sys, os
sys.modules["cairosvg"] = None          # any `import cairosvg` now raises ImportError
os.environ["CVC_BACKEND"] = "model"
sys.path.insert(0, sys.argv[1])
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)
assert c.get("/").status_code == 200
with open(sys.argv[2], "rb") as fh:
    r = c.post("/analyze", files={"image": ("p.jpg", fh, "image/jpeg")},
               data={"orientation": "white", "side_to_move": "white"})
assert r.status_code == 200 and 'data-failed="0"' in r.text, r.text[:300]
print("OK")
"""


def test_app_runs_without_cairo():
    if not (ROOT / "chess_vision/models/yolov8n640-chess-finetuned.onnx").exists():
        pytest.skip("model not available")
    photo = ROOT / "tests/fixtures/real/midgame_real.jpg"
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), str(photo)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]

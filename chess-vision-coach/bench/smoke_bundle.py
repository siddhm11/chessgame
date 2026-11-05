"""Smoke-test the exact file bundle deploy_hf_space.py would upload.

Stages the bundle into an empty temp folder, installs requirements.txt into a
FRESH virtualenv there, starts the server the way the Dockerfile does
(CVC_BACKEND=model, MALLOC_ARENA_MAX=2, one worker, port 8000-style CMD), and
drives the real product flow over HTTP on a photo with a known position.
Catches missing files, missing dependencies, and anything that only worked
because of leftovers in the dev checkout.

    python bench/smoke_bundle.py [--photo PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deploy_hf_space import stage  # noqa: E402

DEFAULT_PHOTO = Path("/tmp/chess-dataset/labeled_originals/"
                     "rnbqkb1r-1p2pppp-p2p1n2-8-3NP3-2N5-PPP2PPP-R1BQKB1R.JPG")
FEN_RE = re.compile(r'id="fen-text"[^>]*>([^<]+)<')
UCI_RE = re.compile(r'hx-vals=\'\{"move": "([a-h][1-8][a-h][1-8][qrbn]?)"\}\'')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, default=DEFAULT_PHOTO)
    ap.add_argument("--port", type=int, default=8055)
    args = ap.parse_args()
    truth = None
    m = re.match(r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})", args.photo.name)
    if m:
        truth = m.group(1).replace("-", "/")

    with tempfile.TemporaryDirectory(prefix="cvc-bundle-") as tmp:
        app_dir = Path(tmp) / "app_root"
        app_dir.mkdir()
        files = stage(ROOT, app_dir)
        print(f"staged {len(files)} files into an empty folder")
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        t0 = time.time()
        subprocess.run([str(venv / "bin/pip"), "install", "-q", "-r", str(app_dir / "requirements.txt")],
                       check=True)
        print(f"fresh pip install of requirements.txt OK ({time.time() - t0:.0f}s)")

        env = {k: v for k, v in os.environ.items() if not k.startswith("CVC_")}
        env.update(CVC_BACKEND="model", MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1")
        server = subprocess.Popen(
            [str(venv / "bin/uvicorn"), "app.main:app", "--host", "127.0.0.1",
             "--port", str(args.port), "--workers", "1"],
            cwd=app_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        base = f"http://127.0.0.1:{args.port}"
        ok = True
        try:
            for _ in range(120):
                try:
                    httpx.get(base + "/", timeout=1)
                    break
                except httpx.HTTPError:
                    time.sleep(0.5)
            c = httpx.Client(base_url=base, timeout=120)
            checks = []
            r = c.get("/")
            checks.append(("GET / renders upload form", r.status_code == 200 and 'name="image"' in r.text))
            with open(args.photo, "rb") as fh:
                r = c.post("/analyze", files={"image": (args.photo.name, fh, "image/jpeg")},
                           data={"orientation": "white", "side_to_move": "white"})
            fen = FEN_RE.search(r.text).group(1).strip() if FEN_RE.search(r.text) else ""
            checks.append(("POST /analyze returns a board", r.status_code == 200 and bool(fen)))
            checks.append(("board found (not the fallback)", 'data-failed="0"' in r.text))
            if truth:
                checks.append((f"FEN matches the photo's true position", fen.split()[0] == truth))
            link = re.search(r'src="(/photo/[0-9a-f]{32})"', r.text)
            r = httpx.get(base + link.group(1), timeout=30) if link else None   # no cookie: iframe case
            checks.append(("photo link serves the upload (no cookie needed)",
                           r is not None and r.status_code == 200 and r.content[:2] == b"\xff\xd8"))
            r = c.post("/best-move", data={"level": "fast"})
            mv = UCI_RE.search(r.text)
            checks.append(("POST /best-move: Stockfish answers", r.status_code == 200 and bool(mv)))
            if mv:
                r = c.post("/play-move", data={"move": mv.group(1)})
                r2 = c.get("/download-fen")
                checks.append(("POST /play-move advances the board", r2.text.split()[0] != fen.split()[0]))
            for name, passed in checks:
                print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
                ok &= passed
        finally:
            server.send_signal(signal.SIGINT)
            try:
                out, _ = server.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                out, _ = server.communicate()
            errors = [ln for ln in out.splitlines() if "Traceback" in ln or "ERROR" in ln]
            if errors:
                ok = False
                print("server log errors:\n  " + "\n  ".join(errors[:10]))
    print("BUNDLE SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

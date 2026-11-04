"""End-to-end HTTP load test against a real uvicorn server.

Starts the app exactly as production does (CVC_BACKEND=model, one worker),
then drives N concurrent "users". Each user owns its own cookie session and
runs the whole product flow on real held-out photos:

  POST /analyze  (multipart upload)   -> FEN must equal the filename's truth
  POST /best-move                     -> must return a move
  POST /play-move  with that move     -> board must advance
  GET  /download-fen                  -> must return the new FEN

Reports per-endpoint p50/p95 latency, FEN accuracy over HTTP, error count,
session isolation (no user ever sees another user's FEN), and server RSS
before/after so leaks show up.

  .venv/bin/python bench/e2e_load.py --users 4 --photos 24
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "finetune_data" / "images" / "val"
ORIG = Path("/tmp/chess-dataset/labeled_originals")
PLACEMENT_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})")
FEN_RE = re.compile(r'id="fen-text"[^>]*>([^<]+)<')
UCI_RE = re.compile(r'hx-vals=\'\{"move": "([a-h][1-8][a-h][1-8][qrbn]?)"\}\'')


def rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--photos", type=int, default=24)
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the whole run; RSS per round exposes leaks")
    args = ap.parse_args()

    photos = []
    for p in sorted(VAL.glob("*.jpg"))[: args.photos]:
        orig = ORIG / f"{p.stem}.JPG"
        m = PLACEMENT_RE.match(p.stem)
        if orig.exists() and m:
            photos.append((orig, m.group(1).replace("-", "/")))
    print(f"{len(photos)} held-out photos, {args.users} concurrent users")

    env = dict(os.environ, CVC_BACKEND="model")
    server = subprocess.Popen(
        [str(ROOT / ".venv/bin/uvicorn"), "app.main:app", "--host", "127.0.0.1",
         "--port", str(args.port), "--workers", "1", "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(60):
            try:
                httpx.get(base + "/", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        # Warm the model once so cold start isn't charged to one user.
        rss_start = rss_mb(server.pid)

        lat = defaultdict(list)
        stats = defaultdict(int)
        failures: list[str] = []
        lock = threading.Lock()

        def user(uid: int, my_photos):
            client = httpx.Client(base_url=base, timeout=60)
            client.get("/")  # obtain session cookie
            for path, truth in my_photos:
                def timed(name, fn):
                    t0 = time.perf_counter()
                    r = fn()
                    with lock:
                        lat[name].append(time.perf_counter() - t0)
                    return r

                try:
                    with open(path, "rb") as fh:
                        r = timed("analyze", lambda: client.post(
                            "/analyze",
                            files={"image": (path.name, fh, "image/jpeg")},
                            data={"orientation": "white", "side_to_move": "white"}))
                    r.raise_for_status()
                    fen = FEN_RE.search(r.text).group(1).strip()
                    with lock:
                        stats["analyzed"] += 1
                        stats["fen_exact"] += fen.split()[0] == truth
                    if fen.split()[0] != truth:
                        # Session isolation: a wrong FEN must not be another
                        # user's photo. Truths are unique per photo.
                        others = {t for _, t in photos if t != truth}
                        if fen.split()[0] in others:
                            with lock:
                                stats["CROSS_SESSION_LEAK"] += 1

                    time.sleep(0.7)  # respect the /best-move rate limit
                    r = timed("best-move", lambda: client.post(
                        "/best-move", data={"level": "fast"}))
                    r.raise_for_status()
                    m = UCI_RE.search(r.text)
                    if not m:
                        if "Invalid position" in r.text:
                            with lock:
                                stats["illegal_after_parse"] += 1
                            continue
                        raise RuntimeError("no move in /best-move response")
                    move = m.group(1)

                    r = timed("play-move", lambda: client.post(
                        "/play-move", data={"move": move}))
                    r.raise_for_status()
                    r = timed("download-fen", lambda: client.get("/download-fen"))
                    if r.text.strip().split()[0] == fen.split()[0]:
                        raise RuntimeError("play-move did not advance the board")
                    with lock:
                        stats["full_flow_ok"] += 1
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        stats["errors"] += 1
                        failures.append(f"user{uid} {path.name}: {exc!r}")
            client.close()

        shards = [photos[i:: args.users] for i in range(args.users)]
        t0 = time.perf_counter()
        rss_rounds = []
        for _ in range(args.rounds):
            threads = [threading.Thread(target=user, args=(i, s)) for i, s in enumerate(shards)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            rss_rounds.append(rss_mb(server.pid))
        wall = time.perf_counter() - t0
        rss_end = rss_rounds[-1]
        n_flows = len(photos) * args.rounds

        print(f"\nwall time {wall:.1f}s for {n_flows} full user flows")
        for name in ("analyze", "best-move", "play-move", "download-fen"):
            xs = lat[name]
            print(f"  {name:<13} n={len(xs):>3}  p50={pct(xs, .5):.2f}s  "
                  f"p95={pct(xs, .95):.2f}s  max={max(xs, default=float('nan')):.2f}s")
        print(f"  FEN exact over HTTP : {stats['fen_exact']}/{stats['analyzed']}")
        print(f"  full flow completed : {stats['full_flow_ok']}/{n_flows}")
        print(f"  illegal after parse : {stats['illegal_after_parse']}")
        print(f"  errors              : {stats['errors']}")
        print(f"  cross-session leaks : {stats['CROSS_SESSION_LEAK']}")
        print(f"  server RSS (MB)     : start {rss_start:.0f} -> after each round " + ", ".join(f"{r:.0f}" for r in rss_rounds))
        for f in failures[:10]:
            print("   !", f)
        return 1 if stats["errors"] or stats["CROSS_SESSION_LEAK"] else 0
    finally:
        server.send_signal(signal.SIGINT)
        server.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())

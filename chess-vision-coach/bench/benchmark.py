"""Benchmark chess vision backends against the samryan18/chess-dataset.

Each photo's filename encodes the ground-truth FEN placement (with `-`
instead of `/`). For every backend (classical + every .onnx dropped into
chess_vision/models/) we compute:

  exact_acc   -- per-square exact match (the real "is the parse correct" number)
  empty_acc   -- per-square: did we agree on whether the square is empty?
  piece_acc   -- of squares the truth says have a piece, how often did we
                 predict a piece at all? (recall-like)
  type_acc    -- when both predicted-and-truth have a piece, how often does
                 the piece type match? (precision-like, conditional)

We sample N photos (default 20) so a run finishes in minutes; bump --n to
hammer harder.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

# Run as `python bench/benchmark.py` without needing PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_vision import vision
from chess_vision.model_backend import ModelBackend

DATASET_URL = "https://github.com/samryan18/chess-dataset.git"
DATASET_PATH = Path("/tmp/chess-dataset")
PLACEMENT_RE = re.compile(
    r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})\.",
)


def ensure_dataset() -> None:
    if DATASET_PATH.exists():
        return
    print(f"Cloning {DATASET_URL} -> {DATASET_PATH} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", DATASET_URL, str(DATASET_PATH)],
        check=True,
    )


def filename_to_placement(name: str) -> str | None:
    m = PLACEMENT_RE.match(name)
    if not m:
        return None
    return m.group(1).replace("-", "/")


def expand_row(row: str) -> list[str]:
    out: list[str] = []
    for ch in row:
        if ch.isdigit():
            out.extend(["."] * int(ch))
        else:
            out.append(ch)
    return out


def placement_to_grid(placement: str) -> list[list[str]] | None:
    rows = placement.split("/")
    if len(rows) != 8:
        return None
    grid = [expand_row(r) for r in rows]
    if not all(len(r) == 8 for r in grid):
        return None
    return grid


def compare(pred_grid, truth_grid) -> dict:
    m = defaultdict(int)
    for r in range(8):
        for c in range(8):
            p, t = pred_grid[r][c], truth_grid[r][c]
            m["total"] += 1
            if p == t:
                m["exact"] += 1
            if (p == ".") == (t == "."):
                m["empty_agree"] += 1
            if t == ".":
                m["truth_empty"] += 1
            else:
                m["truth_piece"] += 1
                if p != ".":
                    m["pred_any_piece"] += 1
                    if p == t:
                        m["pred_type_correct"] += 1
    return m


def discover_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("classical", vision.ClassicalBackend())]
    models_dir = Path(__file__).resolve().parent.parent / "chess_vision" / "models"
    for onnx in sorted(models_dir.glob("*.onnx")):
        backends.append((onnx.stem, ModelBackend(model_path=onnx)))
    return backends


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many photos to sample")
    ap.add_argument("--source", default=None, help="override photo directory")
    args = ap.parse_args()

    ensure_dataset()
    source = Path(args.source) if args.source else DATASET_PATH / "labeled_originals"
    files = sorted(source.glob("*.JPG"))[: args.n]
    if not files:
        files = sorted(source.glob("*.jpg"))[: args.n]
    print(f"running on {len(files)} photos from {source}")

    backends = discover_backends()
    print(f"backends: {[b[0] for b in backends]}")

    totals: dict = {bn: defaultdict(int) for bn, _ in backends}
    parse_fail: dict = {bn: 0 for bn, _ in backends}
    elapsed: dict = {bn: 0.0 for bn, _ in backends}

    for i, fp in enumerate(files, 1):
        truth_p = filename_to_placement(fp.name)
        truth_g = placement_to_grid(truth_p) if truth_p else None
        if truth_g is None:
            print(f"  [{i}/{len(files)}] {fp.name}  SKIP (bad filename)")
            continue
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [{i}/{len(files)}] {fp.name}  SKIP (unreadable)")
            continue
        print(f"  [{i}/{len(files)}] {fp.name}")
        for bname, backend in backends:
            t0 = time.perf_counter()
            out = backend.analyze(img)
            elapsed[bname] += time.perf_counter() - t0
            pred_p = vision._grid_to_placement(out.grid)
            pred_g = placement_to_grid(pred_p)
            if pred_g is None:
                parse_fail[bname] += 1
                continue
            for k, v in compare(pred_g, truth_g).items():
                totals[bname][k] += v

    print()
    header = f"{'backend':<22} {'exact':>7} {'empty':>7} {'recall':>7} {'type|p':>7} {'sec/img':>8} {'fail':>5}"
    print(header)
    print("-" * len(header))
    for bn, _ in backends:
        t = totals[bn]
        n = max(1, t["total"])
        exact = t["exact"] / n
        empty = t["empty_agree"] / n
        recall = t["pred_any_piece"] / max(1, t["truth_piece"])
        type_acc = t["pred_type_correct"] / max(1, t["pred_any_piece"])
        per = elapsed[bn] / max(1, len(files))
        print(f"{bn:<22} {exact:>7.1%} {empty:>7.1%} {recall:>7.1%} {type_acc:>7.1%} {per:>8.2f} {parse_fail[bn]:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

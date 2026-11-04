"""Full evaluation of the production vision pipeline.

Every photo in samryan18/chess-dataset is assigned to exactly one bucket,
based on what the model saw during fine-tuning:

  val       68 photos held out from training (finetune_data/images/val)
  rejected  photos prepare_finetune_dataset.py dropped (low board-detection
            score). NEVER trained on -- a second, harder held-out set.
  train     388 photos the model trained on. Reported for completeness only;
            these numbers are optimistic by construction.

Plus two stress variants of the held-out photos and a cross-domain smoke run:

  val+EXIF  val photos re-saved with EXIF orientation=6 (stored sideways,
            "rotate 90 CW to display") -- proves the EXIF fix end-to-end.
  val+dark  val photos at 55% brightness -- dim-room robustness.
  other     photos NOT from samryan18 (test_photos/, tests/fixtures/real/)
            -- no ground truth; checks no crash + legal-shaped FEN only.

Everything runs through the public entry point vision.analyze_image() with
CVC_BACKEND=model, i.e. exactly what the web app calls.

  CVC_BACKEND=model .venv/bin/python bench/full_eval.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CVC_BACKEND", "model")

from chess_vision import vision  # noqa: E402

DATASET = Path("/tmp/chess-dataset/labeled_originals")
FT = ROOT / "finetune_data" / "images"
PLACEMENT_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})")
TINT = 0.80  # app/main.py CONF_MAYBE: below this a square is tinted


def truth_grid(name: str):
    m = PLACEMENT_RE.match(name)
    if not m:
        return None
    return fen_to_grid(m.group(1).replace("-", "/"))


def fen_to_grid(placement: str):
    grid = []
    for row in placement.split("/"):
        cells = []
        for ch in row:
            cells.extend(["."] * int(ch) if ch.isdigit() else [ch])
        grid.append(cells)
    if len(grid) != 8 or any(len(r) != 8 for r in grid):
        return None
    return grid


def conf_to_grid(psc: dict) -> list[list[float]]:
    g = [[1.0] * 8 for _ in range(8)]
    for sq, c in psc.items():
        g[8 - int(sq[1])]["abcdefgh".index(sq[0])] = c
    return g


def evaluate(items, label):
    """items: list of (path, truth_grid). Returns a metrics dict."""
    sq = exact = truth_pc = found = type_ok = 0
    silent = flagged_wrong = false_tint = 0
    perfect = within2 = fallback = crashes = 0
    errors_per_board: list[int] = []
    per_photo: list[dict] = []
    confusions: Counter = Counter()
    lat: list[float] = []
    finder = vision._get_default_backend()  # exposes last_board_score

    for path, truth in items:
        try:
            t0 = time.perf_counter()
            res = vision.analyze_image(str(path), orientation="white", side_to_move="w")
            lat.append(time.perf_counter() - t0)
        except Exception as exc:  # analyze_image should never raise
            crashes += 1
            print(f"    CRASH {path.name}: {exc!r}")
            continue
        if res.detection_failed:
            fallback += 1
            errors_per_board.append(64)
            per_photo.append({"photo": path.name, "wrong": 64, "fallback": True})
            sq += 64
            continue
        pred = fen_to_grid(res.fen.split()[0])
        conf = conf_to_grid(res.per_square_confidence or {})
        wrong = 0
        for r in range(8):
            for c in range(8):
                p, t = pred[r][c], truth[r][c]
                sq += 1
                if t != ".":
                    truth_pc += 1
                    if p != ".":
                        found += 1
                        type_ok += p == t
                if p == t:
                    exact += 1
                    false_tint += conf[r][c] < TINT
                else:
                    wrong += 1
                    confusions[f"{t}->{p}"] += 1
                    if conf[r][c] >= TINT:
                        silent += 1
                    else:
                        flagged_wrong += 1
        errors_per_board.append(wrong)
        per_photo.append({
            "photo": path.name,
            "wrong": wrong,
            "board_fit": round(getattr(finder, "last_board_score", float("nan")), 4),
            "doubtful_flag": any("alignment" in n for n in res.notes),
            "min_conf": round(min(min(r) for r in conf), 3),
        })
        perfect += wrong == 0
        within2 += wrong <= 2

    n = len(errors_per_board)
    if n == 0:
        return None
    lat.sort()
    out = {
        "set": label,
        "photos": n,
        "exact": exact / max(1, sq),
        "recall": found / max(1, truth_pc),
        "type_given_found": type_ok / max(1, found),
        "perfect_boards": perfect,
        "within_2": within2,
        "fallbacks": fallback,
        "crashes": crashes,
        "wrong_squares": sum(min(e, 64) for e in errors_per_board) - 64 * fallback,
        "wrong_flagged": flagged_wrong,
        "wrong_silent": silent,
        "false_tints_per_board": false_tint / n,
        "p50_s": lat[len(lat) // 2] if lat else 0,
        "p95_s": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0,
        "top_confusions": confusions.most_common(6),
        "worst_boards": sorted(
            ((p["wrong"], p["photo"]) for p in per_photo), reverse=True
        )[:3],
        "per_photo": per_photo,
    }
    print(
        f"  {label:<10} n={n:>3}  exact={out['exact']:.2%}  "
        f"perfect={perfect}/{n} ({perfect / n:.0%})  within2={within2}/{n}  "
        f"wrong={out['wrong_squares']} (flagged {flagged_wrong}, silent {silent})  "
        f"fallback={fallback} crash={crashes}  p50={out['p50_s']:.2f}s p95={out['p95_s']:.2f}s"
    )
    return out


def variant(items, tmp: Path, kind: str):
    """Write stressed copies of the photos; return new (path, truth) items."""
    out = []
    for path, truth in items:
        dst = tmp / f"{kind}_{path.name}"
        with Image.open(path) as im:
            im = im.convert("RGB")
            if kind == "exif":
                # Store pixels rotated 90 CCW and tag orientation=6 so a
                # correct EXIF-aware loader rotates them back upright.
                im = im.rotate(90, expand=True)
                exif = Image.Exif()
                exif[0x0112] = 6
                im.save(dst, "JPEG", quality=92, exif=exif.tobytes())
            elif kind == "dark":
                arr = (np.asarray(im).astype(np.float32) * 0.55).clip(0, 255)
                Image.fromarray(arr.astype(np.uint8)).save(dst, "JPEG", quality=92)
        out.append((dst, truth))
    return out


def smoke(paths, label):
    ok = 0
    for p in paths:
        try:
            res = vision.analyze_image(str(p), orientation="white", side_to_move="w")
            chess.Board(res.fen)
            ok += 1
            pieces = sum(ch.isalpha() for ch in res.fen.split()[0])
            print(f"    {p.name:<34} fallback={res.detection_failed!s:<5} "
                  f"pieces={pieces:>2} conf={res.confidence:.2f} notes={len(res.notes)}")
        except Exception as exc:
            print(f"    {p.name}: FAILED {exc!r}")
    print(f"  {label:<10} {ok}/{len(paths)} returned a parseable FEN without crashing")
    return {"set": label, "photos": len(paths), "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write all metrics here")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    backend = vision._get_default_backend()
    print(f"backend={type(backend).__name__} model={getattr(backend, 'model_path', '-')}")

    val_stems = {p.stem for p in (FT / "val").glob("*.jpg")}
    train_stems = {p.stem for p in (FT / "train").glob("*.jpg")}
    buckets = {"val": [], "rejected": [], "train": []}
    for p in sorted(DATASET.glob("*.JPG")):
        t = truth_grid(p.name)
        if t is None:
            continue
        key = "val" if p.stem in val_stems else "train" if p.stem in train_stems else "rejected"
        buckets[key].append((p, t))
    print({k: len(v) for k, v in buckets.items()})

    results = []
    print("\nHeld-out (never trained on):")
    results.append(evaluate(buckets["val"], "val"))
    results.append(evaluate(buckets["rejected"], "rejected"))
    with tempfile.TemporaryDirectory() as td:
        print("\nStress variants of the val photos:")
        results.append(evaluate(variant(buckets["val"], Path(td), "exif"), "val+EXIF"))
        results.append(evaluate(variant(buckets["val"], Path(td), "dark"), "val+dark"))
    if not args.skip_train:
        print("\nTraining photos (optimistic by construction):")
        results.append(evaluate(buckets["train"], "train"))
    print("\nCross-domain (not samryan18; no ground truth):")
    others = sorted((ROOT / "test_photos").glob("*.jpg")) + sorted(
        (ROOT / "tests" / "fixtures" / "real").glob("*.jpg"))
    results.append(smoke(others, "other"))

    if args.json:
        args.json.write_text(json.dumps([r for r in results if r], indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

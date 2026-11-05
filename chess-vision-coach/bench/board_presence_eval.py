"""Is there a chessboard in this photo? Measure the signals that can tell.

Positives: every samryan18 photo (500 real boards), the repo's other real
board photos, and the synthetic fixtures. Negatives: ordinary photos and
graphics from the OpenCV and scikit-image sample-image folders -- including
deliberately hard ones (camera-calibration checkerboards, bricks, tiles).

For each image records the board finder's grid fit and what the piece model
sees on that crop (pieces found, their mean confidence, whether both kings
appear). Prints the distributions and the best rejection thresholds.

  python bench/board_presence_eval.py --negatives DIR [DIR ...] --json out.json

Negative image folders are read as data only (pixels), never executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_vision import vision  # noqa: E402
from chess_vision.model_backend import ModelBackend  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HARD = ("chessboard", "checker", "left0", "left1", "right0", "right1", "brick",
        "block", "board", "grid", "tile")


def signals(backend: ModelBackend, path: Path) -> dict | None:
    try:
        img = vision._load_bgr(str(path))
    except Exception:
        return None
    if min(img.shape[:2]) < 64:
        return None
    crop, fit = backend._board_finder.find_board_scored(img)
    out = backend._analyze_board(crop)
    confs = [c for row, crow in zip(out.grid, out.conf) for s, c in zip(row, crow) if s != "."]
    flat = [s for row in out.grid for s in row]
    return {
        "fit": round(fit, 4),
        "pieces": len(confs),
        "piece_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
        "kings": ("K" in flat) + ("k" in flat),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", nargs="+", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/tmp/chess-dataset/labeled_originals"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    backend = ModelBackend()
    backend._load()
    rows = []
    positives = sorted(args.dataset.glob("*.JPG")) + sorted((ROOT / "test_photos").glob("*.jpg")) \
        + sorted((ROOT / "tests/fixtures/real").glob("*.jpg")) + sorted((ROOT / "tests/fixtures").glob("*.png"))
    for p in positives:
        s = signals(backend, p)
        if s:
            rows.append({"set": "board", "photo": p.name, **s})
    for d in args.negatives:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG and p.is_file():
                s = signals(backend, p)
                if s:
                    kind = "hard-negative" if any(h in p.name.lower() for h in HARD) else "negative"
                    rows.append({"set": kind, "photo": str(p.relative_to(d)), **s})
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))

    def q(xs, p):
        return float(np.percentile(xs, p)) if xs else float("nan")

    for kind in ("board", "negative", "hard-negative"):
        sel = [r for r in rows if r["set"] == kind]
        fits = [r["fit"] for r in sel]
        print(f"{kind:<14} n={len(sel):>3}  fit min={min(fits):.3f} p5={q(fits, 5):.3f} "
              f"p50={q(fits, 50):.3f} p95={q(fits, 95):.3f} max={max(fits):.3f}  "
              f"pieces p50={q([r['pieces'] for r in sel], 50):.0f}  "
              f"both kings={sum(r['kings'] == 2 for r in sel)}/{len(sel)}")

    boards = [r for r in rows if r["set"] == "board"]
    negs = [r for r in rows if r["set"] != "board"]
    print("\nrule: reject when fit < T")
    for t in (0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75):
        fr = sum(r["fit"] < t for r in boards)
        tr = sum(r["fit"] < t for r in negs)
        print(f"  T={t:.2f}: boards wrongly rejected {fr}/{len(boards)}, "
              f"non-boards rejected {tr}/{len(negs)}")
    print("\nrule: reject when fit < T OR no king found")
    for t in (0.60, 0.65, 0.70):
        fr = sum(r["fit"] < t or r["kings"] == 0 for r in boards)
        tr = sum(r["fit"] < t or r["kings"] == 0 for r in negs)
        print(f"  T={t:.2f}: boards wrongly rejected {fr}/{len(boards)}, "
              f"non-boards rejected {tr}/{len(negs)}")
    print("\nnon-boards that would still pass at T=0.65 (fit, pieces, kings):")
    for r in sorted(negs, key=lambda r: -r["fit"])[:15]:
        if r["fit"] >= 0.65:
            print(f"  {r['set']:<13} {r['photo']:<40} fit={r['fit']:.3f} pieces={r['pieces']} kings={r['kings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

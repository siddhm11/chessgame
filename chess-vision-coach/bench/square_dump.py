"""Dump every square's (truth, prediction, confidence) for a labelled photo folder.

For calibration work: run inference once, then study silent errors offline.

  CVC_BACKEND=model python bench/square_dump.py --photos DIR [--use-corners] --out squares.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import full_eval  # noqa: E402
from chess_vision import vision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--use-corners", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path, truth, orientation, corners, give in full_eval.photo_folder(args.photos, args.use_corners):
        res = vision.analyze_image(str(path), orientation=orientation, side_to_move="w",
                                   corners=corners if give else None)
        if res.detection_failed:
            rows.append({"photo": path.name, "failed": True})
            continue
        pred = full_eval.fen_to_grid(res.fen.split()[0])
        conf = full_eval.conf_to_grid(res.per_square_confidence or {})
        rows.append({"photo": path.name, "truth": truth, "pred": pred, "conf": conf,
                     "notes": res.notes})
    args.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} photos to {args.out}")


if __name__ == "__main__":
    main()

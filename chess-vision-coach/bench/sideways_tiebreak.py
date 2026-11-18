"""Sideways boards: which of the two 90-degree turns is the right one?

Every samryan18 photo is upright with White at the bottom. Turn its board
crop 90 degrees either way and ask ModelBackend for both candidate turns;
exactly one restores the true position. Scores candidate tie-break rules by
how often they pick the turn that matches the label (and how many squares a
wrong pick costs). Held-out (val + rejected) and train are reported apart.

    CVC_BACKEND=model .venv/bin/python bench/sideways_tiebreak.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_vision import vision  # noqa: E402
from chess_vision.model_backend import ModelBackend, _white_below_black  # noqa: E402

DATASET = Path("/tmp/chess-dataset/labeled_originals")
SPLIT = Path(__file__).resolve().parent.parent / "finetune_data"


def truth_grid(name: str):
    rows = Path(name).stem.split("-")
    grid = []
    for row in rows:
        cells = []
        for ch in row:
            cells.extend(["."] * int(ch) if ch.isdigit() else [ch])
        grid.append(cells)
    return grid


def held_out_names() -> set[str]:
    trained = {p.stem.split(".")[0] for p in (SPLIT / "images" / "train").glob("*")} if SPLIT.exists() else set()
    return {p.name for p in DATASET.iterdir() if Path(p.name).stem not in trained}


def features(grid, conf):
    flat = [sq for row in grid for sq in row]
    pieces = [(grid[r][c], conf[r][c]) for r in range(8) for c in range(8) if grid[r][c] != "."]
    mean_conf = sum(c for _, c in pieces) / len(pieces) if pieces else 0.0
    back_rank_pawns = sum(1 for c in range(8) for r in (0, 7) if grid[r][c] in "Pp")
    return {
        "kings": ("K" in flat) + ("k" in flat),
        "wbb": _white_below_black(grid),
        "mean_conf": mean_conf,
        "back_rank_pawns": back_rank_pawns,
        "n": len(pieces),
    }


RULES = {
    "kings,wbb (current)": lambda f: (f["kings"], f["wbb"]),
    "kings,conf": lambda f: (f["kings"], f["mean_conf"]),
    "kings,-backpawns,conf": lambda f: (f["kings"], -f["back_rank_pawns"], f["mean_conf"]),
    "kings,-backpawns,wbb": lambda f: (f["kings"], -f["back_rank_pawns"], f["wbb"]),
}


def main() -> None:
    backend = ModelBackend()
    backend._load()
    held = held_out_names()
    rows = []
    for photo in sorted(DATASET.iterdir()):
        truth = truth_grid(photo.name)
        crop = backend._board_finder.locate_board(vision._load_bgr(str(photo)))
        upright = crop.image
        for turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            sideways = cv2.rotate(upright, turn)
            cands = []
            for fix in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                g, c, _ = backend._infer_board(cv2.rotate(sideways, fix))
                wrong = sum(g[r][k] != truth[r][k] for r in range(8) for k in range(8))
                cands.append({"features": features(g, c), "wrong": wrong})
            rows.append({"photo": photo.name, "held_out": photo.name in held, "cands": cands})
    Path("/tmp/sideways_tiebreak.json").write_text(json.dumps(rows))
    for name, key in RULES.items():
        for label, sel in (("held-out", True), ("train", False)):
            subset = [r for r in rows if r["held_out"] == sel]
            picks = [max(r["cands"], key=lambda c: key(c["features"])) for r in subset]
            best = [min(c["wrong"] for c in r["cands"]) for r in subset]
            right = sum(p["wrong"] == b for p, b in zip(picks, best))
            wrong_sq = sum(p["wrong"] for p in picks)
            print(f"{name:24s} {label:8s} right turn {right}/{len(subset)}  wrong squares {wrong_sq}")


if __name__ == "__main__":
    main()

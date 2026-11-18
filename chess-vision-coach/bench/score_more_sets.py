"""Score the current model on the extra piece sets (set 3, set 4).

Adds best-move agreement to the per-square numbers bench/full_eval.py
reports, and can score a board crop directly (bypassing the board finder).

  # set 3: prepared by bench/set3_prepare.py (clean held-out frames only)
  python bench/score_more_sets.py --set3 /data/set3_prepared --mode crop
  python bench/score_more_sets.py --set3 /data/set3_prepared --mode pipeline

  # set 4 (or any folder in full_eval.py --photos format):
  python bench/score_more_sets.py --photos /data/set4_prepared/photos [--use-corners]

Modes
  crop      the 640x640 board crop goes straight to the model backend's
            board reader (orientation check, TTA, sanity layer) -- what the
            app does after it has found the board.
  pipeline  vision.analyze_image on the whole image, exactly what the app
            calls. Set-3 frames are already cropped to the board edge, so they
            get a neutral grey border (--pad squares wide, default 1) for the
            board finder to see the edge.

Metrics (per board, then summed)
  exact     share of the 64 squares read correctly
  perfect   boards with all 64 squares right
  silent    wrong squares shown with confidence >= 0.80 (not tinted in the app)
  best-move Stockfish's best move on the predicted position equals its best
            move on the true position (depth --depth, one thread, no castling
            rights). A predicted position python-chess calls invalid counts as
            disagree. Boards whose TRUE position is invalid (e.g. the side not
            to move in check) are left out of this metric and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CVC_BACKEND", "model")

import chess  # noqa: E402
import chess.engine  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

TINT = 0.80
NAME_RE = re.compile(r"^([rnbqkpRNBQKP1-8]+(?:-[rnbqkpRNBQKP1-8]+){7})(?:__([wb])__)?")


def grid_from_placement(placement: str) -> list[list[str]]:
    grid = []
    for row in placement.split("/"):
        cells = []
        for ch in row:
            cells.extend(["."] * int(ch) if ch.isdigit() else [ch])
        grid.append(cells)
    if len(grid) != 8 or any(len(r) != 8 for r in grid):
        raise ValueError(f"bad placement {placement!r}")
    return grid


def placement_from_grid(grid) -> str:
    rows = []
    for row in grid:
        s, run = "", 0
        for ch in row:
            if ch == ".":
                run += 1
            else:
                s += (str(run) if run else "") + ch
                run = 0
        rows.append(s + (str(run) if run else ""))
    return "/".join(rows)


def compare(pred, conf, truth) -> dict:
    wrong = silent = 0
    confusions = Counter()
    for r in range(8):
        for c in range(8):
            if pred[r][c] != truth[r][c]:
                wrong += 1
                silent += conf[r][c] >= TINT
                confusions[f"{truth[r][c]}->{pred[r][c]}"] += 1
    return {"wrong": wrong, "silent": silent, "confusions": confusions}


class BestMove:
    def __init__(self, depth: int):
        path = shutil.which("stockfish") or "/usr/games/stockfish"
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 64})
        self.depth = depth
        self._cache: dict[str, str | None] = {}

    def __call__(self, fen: str) -> str | None:
        """Best move in UCI, '' for a game-over position, None if invalid."""
        if fen in self._cache:
            return self._cache[fen]
        board = chess.Board(fen)
        if not board.is_valid():
            move = None
        elif board.is_game_over():
            move = ""
        else:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth),
                                       game=fen)  # new game per position: no hash carry-over
            move = info["pv"][0].uci() if info.get("pv") else ""
        self._cache[fen] = move
        return move

    def close(self):
        self.engine.quit()


def pad_board(img: np.ndarray, squares: float) -> np.ndarray:
    n = int(round(img.shape[0] / 8 * squares))
    return cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_CONSTANT, value=(128, 128, 128))


def run(items, mode: str, pad: float, depth: int, label: str) -> dict:
    """items: dicts with path, truth (grid), stm ('w'/'b'), orientation, corners."""
    from chess_vision import vision

    backend = vision._get_default_backend()
    if mode == "crop" and not hasattr(backend, "_analyze_board"):
        raise SystemExit("--mode crop needs the model backend (CVC_BACKEND=model)")
    if mode == "crop":
        backend._load()  # analyze() does this; _analyze_board alone does not
    bm = BestMove(depth)
    tot = Counter()
    confusions = Counter()
    per_board = []
    lat = []
    try:
        for it in items:
            img = cv2.imread(str(it["path"]))
            t0 = time.perf_counter()
            failed = False
            if mode == "crop":
                out = backend._analyze_board(img)
                pred, conf = out.grid, out.conf
                if it["orientation"] == "black":
                    pred = [row[::-1] for row in pred[::-1]]
                    conf = [row[::-1] for row in conf[::-1]]
            else:
                src = pad_board(img, pad) if pad else img
                res = vision.analyze_image(src, orientation=it["orientation"],
                                           side_to_move=it["stm"], corners=it.get("corners"))
                failed = res.detection_failed
                pred = grid_from_placement(res.fen.split()[0])
                psc = res.per_square_confidence or {}
                conf = [[1.0] * 8 for _ in range(8)]
                for sq, c in psc.items():
                    conf[8 - int(sq[1])]["abcdefgh".index(sq[0])] = c
            lat.append(time.perf_counter() - t0)
            cmp = compare(pred, conf, it["truth"])
            if failed:  # starting-position fallback: every square counts as wrong
                cmp["wrong"], cmp["silent"] = 64, 0
            confusions.update(cmp["confusions"])
            true_fen = f"{placement_from_grid(it['truth'])} {it['stm']} - - 0 1"
            pred_fen = f"{placement_from_grid(pred)} {it['stm']} - - 0 1"
            t_move = bm(true_fen)
            if t_move is None:
                agree = None
            else:
                p_move = None if failed else bm(pred_fen)
                agree = p_move is not None and p_move == t_move
            tot["boards"] += 1
            tot["squares"] += 64
            tot["wrong"] += cmp["wrong"]
            tot["silent"] += cmp["silent"]
            tot["perfect"] += cmp["wrong"] == 0
            tot["fallback"] += failed
            tot["truth_invalid"] += agree is None
            tot["pred_invalid"] += (not failed) and not chess.Board(pred_fen).is_valid()
            tot["bm_scored"] += agree is not None
            tot["bm_agree"] += bool(agree)
            per_board.append({"image": Path(it["path"]).name, "wrong": cmp["wrong"],
                              "silent": cmp["silent"], "fallback": failed,
                              "true": true_fen, "pred": pred_fen, "best_move_agree": agree})
    finally:
        bm.close()
    n = max(1, tot["boards"])
    lat.sort()
    out = {
        "set": label, "mode": mode, "boards": tot["boards"],
        "exact": 1 - tot["wrong"] / max(1, tot["squares"]),
        "perfect": tot["perfect"], "wrong_squares": tot["wrong"], "silent": tot["silent"],
        "silent_per_board": tot["silent"] / n, "fallbacks": tot["fallback"],
        "pred_invalid": tot["pred_invalid"], "truth_invalid": tot["truth_invalid"],
        "best_move_agree": tot["bm_agree"], "best_move_scored": tot["bm_scored"],
        "p50_s": lat[len(lat) // 2] if lat else 0.0,
        "top_confusions": confusions.most_common(8), "per_board": per_board,
    }
    print(f"  {label} [{mode}] boards={out['boards']} exact={out['exact']:.2%} "
          f"perfect={out['perfect']}/{out['boards']} wrong={out['wrong_squares']} "
          f"silent={out['silent']} ({out['silent_per_board']:.2f}/board) "
          f"best-move {out['best_move_agree']}/{out['best_move_scored']} "
          f"(true-invalid {out['truth_invalid']}, pred-invalid {out['pred_invalid']}) "
          f"fallback={out['fallbacks']} p50={out['p50_s']:.2f}s")
    print(f"    top confusions: {out['top_confusions']}")
    return out


def set3_items(prepared: Path, splits: list[str]) -> list[dict]:
    items = []
    for line in (prepared / "positions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["eval"] and r["split"] in splits:
            items.append({"path": prepared / r["image"], "truth": grid_from_placement(r["placement"]),
                          "stm": "w", "orientation": "white", "split": r["split"]})
    return items


def photo_items(folder: Path, use_corners: bool) -> list[dict]:
    corners = {}
    if (folder / "corners.json").exists():
        corners = json.loads((folder / "corners.json").read_text())
    stm = {}
    if (folder / "side_to_move.json").exists():
        stm = json.loads((folder / "side_to_move.json").read_text())
    items = []
    for p in sorted(folder.iterdir()):
        m = NAME_RE.match(p.name)
        if not m or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        items.append({
            "path": p, "truth": grid_from_placement(m.group(1).replace("-", "/")),
            "stm": stm.get(p.name, "w"),
            "orientation": "black" if m.group(2) == "b" else "white",
            "corners": corners.get(p.name) if use_corners else None,
        })
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set3", type=Path, help="folder written by bench/set3_prepare.py")
    ap.add_argument("--splits", default="test,val", help="set-3 splits to score, comma separated")
    ap.add_argument("--photos", type=Path, help="folder in full_eval.py --photos format")
    ap.add_argument("--use-corners", action="store_true")
    ap.add_argument("--mode", choices=["crop", "pipeline"], default="pipeline")
    ap.add_argument("--pad", type=float, default=1.0, help="set 3 pipeline: border, in squares")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    results = []
    if args.set3:
        all_items = set3_items(args.set3, args.splits.split(","))
        for split in args.splits.split(","):
            items = [i for i in all_items if i["split"] == split]
            if items:
                results.append(run(items, args.mode, args.pad if args.mode == "pipeline" else 0,
                                   args.depth, f"set3-{split}"))
    elif args.photos:
        if args.mode == "crop":
            raise SystemExit("--mode crop is for set 3 (images already cropped to the board)")
        items = photo_items(args.photos, args.use_corners)
        results.append(run(items, "pipeline", 0, args.depth,
                           args.photos.name + (" +corners" if args.use_corners else "")))
    else:
        ap.error("give --set3 or --photos")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

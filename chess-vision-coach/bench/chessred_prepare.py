"""Turn ChessReD into a folder bench/full_eval.py --photos can score.

ChessReD (Masouris & van Gemert, 2023): 10,800 smartphone photos of a
different board and piece set, shot from a player's seat (angled), with the
position of every piece. The 2,078-image "ChessReD2K" subset also has the
board's four corners. License CC BY-NC-SA 4.0 (non-commercial use).
https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2

Download annotations.json (~22 MB) and chessred2k.zip (~4.7 GB) from that
page, unzip, then:

    python bench/chessred_prepare.py --annotations annotations.json \
        --images-root <folder you unzipped into> --split test --out chessred_test
    python bench/full_eval.py --photos chessred_test             # auto board finding
    python bench/full_eval.py --photos chessred_test --use-corners  # given the true corners

Each image is copied as <placement>__<w|b>__<id>.jpg (w/b = which side is
closer to the camera, derived from the annotated corners), and the true
corners go to corners.json. Images without corner annotations are skipped:
without them the camera side is unknown.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FILES, RANKS = "abcdefgh", "87654321"
PIECE = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}
CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def symbol(category_name: str) -> str | None:
    """'white-pawn' -> 'P', 'black-king' -> 'k', 'empty' -> None."""
    name = category_name.lower().replace("_", "-")
    if name == "empty" or "-" not in name:
        return None
    colour, kind = name.split("-", 1)
    s = PIECE[kind]
    return s.upper() if colour == "white" else s


def placement(squares: dict[str, str]) -> str:
    rows = []
    for rank in RANKS:
        row, run = "", 0
        for f in FILES:
            s = squares.get(f + rank)
            if s is None:
                run += 1
            else:
                row += (str(run) if run else "") + s
                run = 0
        rows.append(row + (str(run) if run else ""))
    return "/".join(rows)


def split_ids(splits: dict, split: str) -> list[int]:
    """Find the image ids of `split`, preferring the corner-annotated 2K subset."""
    for key in ("chessred2k", "chessred_2k", "2k"):
        sub = splits.get(key)
        if isinstance(sub, dict) and split in sub:
            return list(sub[split]["image_ids"])
    if split in splits:
        return list(splits[split]["image_ids"])
    raise SystemExit(f"split '{split}' not found; splits has keys: {sorted(splits)}")


def corner_table(annotations: dict) -> dict[int, dict[str, list[float]]]:
    raw = annotations.get("corners") or []
    table = {}
    for item in raw:
        c = {k.replace("-", "_"): v for k, v in (item.get("corners") or {}).items()}
        if all(k in c for k in CORNER_KEYS):
            table[item["image_id"]] = c
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    args = ap.parse_args(argv)

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    cats = {c["id"]: symbol(c["name"]) for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    ids = split_ids(data.get("splits", {}), args.split)
    corners = corner_table(data.get("annotations", {}))
    if not corners:
        print("annotations.json has no board-corner annotations (need the ChessReD2K "
              "version, v2 of the dataset).", file=sys.stderr)
        return 1

    pieces: dict[int, dict[str, str]] = {}
    for p in data["annotations"]["pieces"]:
        s = cats.get(p["category_id"])
        if s:
            pieces.setdefault(p["image_id"], {})[p["chessboard_position"]] = s

    args.out.mkdir(parents=True, exist_ok=True)
    sidecar, written, skipped = {}, 0, 0
    for image_id in ids:
        if args.limit and written >= args.limit:
            break
        im, c = images.get(image_id), corners.get(image_id)
        src = args.images_root / im["path"] if im else None
        if c is None or src is None or not src.is_file():
            skipped += 1
            continue
        # Corners are named from White's point of view: bottom_left is a1.
        # If a1/h1 sit lower in the photo than a8/h8, White is nearer the camera.
        white_near = (c["bottom_left"][1] + c["bottom_right"][1]) > (c["top_left"][1] + c["top_right"][1])
        name = f"{placement(pieces.get(image_id, {})).replace('/', '-')}__{'w' if white_near else 'b'}__{image_id}{src.suffix.lower()}"
        shutil.copy2(src, args.out / name)
        w, h = im.get("width"), im.get("height")
        if not (w and h):
            from PIL import Image
            with Image.open(src) as pil:
                w, h = pil.size
        sidecar[name] = [[c[k][0] / w, c[k][1] / h] for k in CORNER_KEYS]
        written += 1
    (args.out / "corners.json").write_text(json.dumps(sidecar, indent=1))
    print(f"wrote {written} images to {args.out} ({skipped} skipped: no corners or file missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate synthetic top-down chessboard PNGs from FEN.

IMPORTANT: these fixtures are rendered by python-chess itself. They exercise
the rendering layer, the board-segmentation logic and the FEN-handling /
orientation layer of the pipeline -- they do NOT test real-world vision
(camera angle, lighting, perspective, real piece sets). Real vision testing
requires real photos, which the user will supply separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_vision import render  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# (name, placement-only FEN)
FIXTURES: list[tuple[str, str]] = [
    ("starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"),
    # Italian Game, a typical middlegame.
    ("midgame", "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"),
    # King + pawn vs king endgame (sparse position).
    ("endgame", "8/8/8/3k4/8/4P3/3K4/8"),
]

# Rendered at 640px with coordinates disabled so the whole image is the
# 8x8 board -- this matches what the vision pipeline expects as input.
SIZE = 640


def generate(force: bool = True) -> list[Path]:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, placement in FIXTURES:
        out = FIXTURES_DIR / f"{name}.png"
        paths.append(out)
        if out.exists() and not force:
            continue
        png = render.fen_to_png_bytes(
            f"{placement} w - - 0 1", size=SIZE, coordinates=False
        )
        out.write_bytes(png)
    return paths


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p}")

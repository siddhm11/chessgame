"""CLI for the chess-vision pipeline.

Usage:
    python -m chess_vision.cli photo.jpg [--orientation white|black]
                                         [--side-to-move white|black]
                                         [--output output.png]
"""

from __future__ import annotations

import argparse
import sys

from . import render, vision

FILES = "abcdefgh"


def _confidence_report(result: vision.VisionResult) -> str:
    lines: list[str] = []
    lines.append(f"Overall confidence: {result.confidence:.2%}")
    psc = result.per_square_confidence
    if psc:
        # 8x8 grid, rank 8 at the top.
        lines.append("Per-square confidence:")
        for rank in range(8, 0, -1):
            cells = []
            for f in FILES:
                v = psc.get(f"{f}{rank}", 0.0)
                cells.append(f"{v:4.2f}")
            lines.append(f"  {rank} " + " ".join(cells))
        lines.append("    " + "    ".join(FILES))
        low = sorted(sq for sq, v in psc.items() if v < 0.70)
        if low:
            lines.append(f"Low-confidence squares (<0.70): {', '.join(low)}")
        else:
            lines.append("Low-confidence squares (<0.70): none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chess_vision.cli",
        description="Extract a FEN from a photo of a chessboard.",
    )
    parser.add_argument("image", help="path to the chessboard photo")
    parser.add_argument(
        "--orientation",
        choices=["white", "black"],
        default="white",
        help="which side is closer to the camera (default: white)",
    )
    parser.add_argument(
        "--side-to-move",
        choices=["white", "black"],
        default="white",
        help="side to move, written into the FEN (default: white)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.png",
        help="side-by-side visualization output path (default: output.png)",
    )
    args = parser.parse_args(argv)

    stm = "b" if args.side_to_move == "black" else "w"
    result = vision.analyze_image(
        args.image, orientation=args.orientation, side_to_move=stm
    )

    print(f"FEN: {result.fen}")
    if result.detection_failed:
        print(
            "WARNING: board detection failed or confidence too low. "
            "Falling back to the starting position -- set up the board "
            "manually."
        )
    print(_confidence_report(result))

    try:
        render.side_by_side(args.image, result.fen, args.output)
        print(f"Visualization written to: {args.output}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write visualization: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

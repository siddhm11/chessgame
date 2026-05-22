"""Render a FEN to an image with python-chess.

python-chess produces SVG; we rasterize to PNG with CairoSVG.
"""

from __future__ import annotations

import io

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image


def _normalize_fen(fen: str) -> str:
    """Accept either a full FEN or a placement-only field; return a full FEN."""
    if len(fen.split()) == 1:
        return f"{fen} w - - 0 1"
    return fen


def board_svg(
    fen: str,
    *,
    size: int = 400,
    lastmove: chess.Move | None = None,
    arrows: list | None = None,
    check: int | None = None,
    coordinates: bool = True,
    fill: dict | None = None,
) -> str:
    """Return an SVG string for the given FEN (placement-only FEN accepted).

    `fill` maps square ints to a CSS color, used to tint squares (e.g. to
    flag low-confidence detections).
    """
    board = chess.Board(_normalize_fen(fen))
    return chess.svg.board(
        board,
        size=size,
        lastmove=lastmove,
        arrows=arrows or [],
        check=check,
        coordinates=coordinates,
        fill=fill or {},
    )


def svg_to_png_bytes(svg: str, *, width: int, height: int) -> bytes:
    return cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=width,
        output_height=height,
    )


def fen_to_png_bytes(fen: str, *, size: int = 400, coordinates: bool = True) -> bytes:
    """Render a FEN directly to PNG bytes."""
    svg = board_svg(fen, size=size, coordinates=coordinates)
    return svg_to_png_bytes(svg, width=size, height=size)


def fen_to_image(fen: str, *, size: int = 640, coordinates: bool = False) -> np.ndarray:
    """Render a FEN to an RGB numpy image (used for synthetic fixtures and for
    building the vision template set)."""
    png = fen_to_png_bytes(fen, size=size, coordinates=coordinates)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    return np.array(img)


def side_by_side(
    original_path: str,
    fen: str,
    out_path: str,
    *,
    board_size: int = 480,
) -> None:
    """Write a side-by-side PNG: original photo on the left, parsed board on
    the right."""
    original = Image.open(original_path).convert("RGB")

    board_png = fen_to_png_bytes(fen, size=board_size, coordinates=True)
    board_img = Image.open(io.BytesIO(board_png)).convert("RGB")

    # Scale the original to match the board height, preserving aspect ratio.
    target_h = board_size
    scale = target_h / original.height
    orig_resized = original.resize(
        (max(1, int(original.width * scale)), target_h), Image.LANCZOS
    )

    pad = 16
    canvas_w = orig_resized.width + board_img.width + pad * 3
    canvas_h = target_h + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    canvas.paste(orig_resized, (pad, pad))
    canvas.paste(board_img, (orig_resized.width + pad * 2, pad))
    canvas.save(out_path)

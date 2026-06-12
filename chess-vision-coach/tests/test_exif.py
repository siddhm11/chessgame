"""Phone uploads encode rotation in EXIF rather than the pixel data; OpenCV
ignores EXIF, so we route loading through Pillow to keep the analysed
image and the displayed thumbnail upright."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from chess_vision import vision


# A small synthetic image with an obvious top-row vs bottom-row signal:
# top row pure red, bottom row pure blue (in BGR after Pillow rgb->BGR).
def _make_test_jpeg(tmp: Path, exif_orientation: int) -> Path:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :] = [255, 0, 0]   # top half red (RGB)
    arr[20:, :] = [0, 0, 255]   # bottom half blue (RGB)
    pil = Image.fromarray(arr)

    # JPEG EXIF requires APP1 segment; Pillow's getexif().tobytes() works.
    exif = pil.getexif()
    exif[274] = exif_orientation  # 274 = Orientation tag
    out = tmp / f"orient_{exif_orientation}.jpg"
    pil.save(out, "JPEG", exif=exif.tobytes())
    return out


def test_load_bgr_respects_exif_rotate_90cw(tmp_path):
    """EXIF orientation=6 means 'rotate 90° CW for display'. After
    transpose the original top half (red) ends up on the RIGHT, and the
    image was originally 40x40 — without the transpose the columns would
    still be uniform top-red / bottom-blue and red would be split 50/50."""
    src = _make_test_jpeg(tmp_path, exif_orientation=6)
    img = vision._load_bgr(str(src))
    h, w = img.shape[:2]
    left_red = img[:, : w // 2, 2].mean()   # BGR: index 2 = Red
    right_red = img[:, w // 2 :, 2].mean()
    assert right_red > left_red + 50, (
        f"expected red on the right after EXIF transpose; "
        f"got left={left_red:.0f} right={right_red:.0f}"
    )


def test_load_bgr_no_exif_passthrough(tmp_path):
    """orientation=1 is the identity — the image should come back as
    written, with red on top."""
    src = _make_test_jpeg(tmp_path, exif_orientation=1)
    img = vision._load_bgr(str(src))
    h, _ = img.shape[:2]
    top_red = img[: h // 2, :, 2].mean()
    bottom_red = img[h // 2 :, :, 2].mean()
    assert top_red > bottom_red + 50

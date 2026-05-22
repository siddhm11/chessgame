"""Render the analyze page to docs/screenshot.png.

No headless browser is available in this environment (and Playwright's
browser-download CDN is blocked), so we drive the running server over HTTP,
splice the HTMX analyze fragment into the full page exactly as the browser
would after the swap, and rasterize it with WeasyPrint + PyMuPDF.

This is a dev/docs utility, not part of the app or test suite. It needs two
extra packages that the app itself does not:

    pip install weasyprint pymupdf

Usage: start the server, then `python tests/make_screenshot.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://localhost:8000"
OUT = ROOT / "docs" / "screenshot.png"

# Fixed canvas so the two-column desktop layout is captured in one page.
PAGE_CSS = CSS(string="@page { size: 1180px 1320px; margin: 0; }")


def main() -> int:
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    index_html = client.get("/").text
    with open(ROOT / "tests" / "fixtures" / "starting.png", "rb") as fh:
        fragment = client.post(
            "/analyze",
            files={"image": ("starting.png", fh, "image/png")},
            data={"orientation": "white", "side_to_move": "white"},
        ).text

    # Also run the engine so the screenshot shows the populated best-move
    # panel. The /best-move response is [panel content] + [OOB board]; keep
    # just the panel content and splice it into the empty panel div.
    bestmove = client.post("/best-move", data={"level": "normal"}).text
    panel = bestmove.split('<div id="board-container"', 1)[0]
    fragment = fragment.replace(
        '<div id="best-move-panel" class="best-move-panel"></div>',
        f'<div id="best-move-panel" class="best-move-panel">{panel}</div>',
    )

    full = re.sub(
        r'(<main id="app">).*?(</main>)',
        lambda m: m.group(1) + fragment + m.group(2),
        index_html,
        flags=re.S,
    )

    pdf = HTML(string=full, base_url=BASE_URL + "/").write_pdf(stylesheets=[PAGE_CSS])

    doc = fitz.open(stream=pdf, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for crispness
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(OUT))
    print(f"wrote {OUT} ({pix.width}x{pix.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

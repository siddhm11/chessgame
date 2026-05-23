# Chess Vision Coach

Upload a photo of a physical chessboard. The app extracts the position to a
FEN, lets you fix any misclassified pieces by clicking squares, and uses
Stockfish to suggest the best move with an evaluation.

```
photo  ->  board detection  ->  FEN  ->  your corrections  ->  Stockfish
```

---

## Prerequisites

- **Python 3.10+** (developed and tested on 3.11)
- **Stockfish** chess engine
- A C library or two for image rendering (`libcairo`, `libpango`) — already
  present on most desktop Linux and macOS installs.

### Install Stockfish

| Platform | Command |
|----------|---------|
| **Linux (Debian/Ubuntu)** | `sudo apt-get install stockfish` |
| **macOS** | `brew install stockfish` |
| **Windows** | Download from <https://stockfishchess.org/download/> and either put it on your `PATH` or set `STOCKFISH_PATH` |

The app auto-discovers Stockfish at `/usr/games/stockfish`, `/usr/bin`,
`/usr/local/bin`, `/opt/homebrew/bin`, or anywhere on `PATH`. You can also
point it explicitly:

```bash
export STOCKFISH_PATH=/full/path/to/stockfish
```

---

## Quick start

```bash
./run.sh
```

`run.sh` checks your Python version, creates a `.venv`, installs dependencies,
verifies Stockfish, and starts the server. Then open:

**<http://localhost:8000>**

To run it manually instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --workers 1
```

---

## Usage

### Web app

1. Open <http://localhost:8000>.
2. Upload a chessboard photo. Choose **which side is closer to the camera**
   and **whose move it is**.
3. The parsed board appears next to your photo with an overall confidence
   score. Squares the vision pass was unsure about are **tinted amber/red**
   so you can see exactly what to check.
4. **Click any square** to correct a misread piece (or empty it). The clicked
   square is highlighted, and once you fix it the tint clears.
5. Pick an engine **strength** (Fast / Normal / Strong) and click **Get best
   move** — Stockfish returns the best move (drawn as an arrow), an
   evaluation, the **principal-variation line** for each of the top 3
   candidates, and a button to **play the suggested move on the board**.
6. Copy or download the FEN at any time.

### Command-line vision pipeline

```bash
python -m chess_vision.cli photo.jpg --orientation white --side-to-move white
```

Outputs the parsed FEN, a per-square confidence grid, and a side-by-side
visualization (`output.png`: original photo | rendered board).

### Vision backend — classical (default) or YOLO11n model

Two interchangeable backends live behind the `VisionBackend` Protocol in
`chess_vision/vision.py`:

| Backend | What it does | When it shines |
|---|---|---|
| `classical` (default) | Template-match each warped square against python-chess's own piece glyphs | Synthetic boards, lichess/chess.com diagrams (parses at ~100%) |
| `model` | Ultralytics **YOLO11n** ONNX detector trained on real chess photos | Real-world boards, occluded pieces, varied piece sets |

Switch backends by setting `CVC_BACKEND=model` before launching the app or
the CLI. The ONNX file is vendored at
`chess_vision/models/yolo11n-chess.onnx` (~10 MB) and is loaded lazily —
no cost unless you actually use it. `onnxruntime` is included in
`requirements.txt`.

> **License note:** the model weights are AGPL-3.0 (Ultralytics YOLO11
> framework license, regardless of the dataset). Fine for personal /
> learning use. For commercial use you'd need an Ultralytics commercial
> license or a non-YOLO model.

---

## How the pipeline works

**Phase 1 — vision (`chess_vision/`)**

1. **Board detection** (`vision.py`) — locate the board: the whole frame and
   the largest perspective-correctable quad are both scored on how cleanly
   their cells form a checkerboard; the better one wins.
2. **Segmentation** — the board is warped to 640×640 and split into 64 cells.
3. **Per-square classification** — each cell is matched against a reference
   set of piece glyphs (rendered by python-chess) on light and dark
   backgrounds, using normalized cross-correlation. Empty squares are found
   by comparison to an empty-cell template.
4. **FEN assembly + orientation** (`orientation.py`) — squares become a FEN;
   if Black's side was closer to the camera the board is rotated 180°.
5. A `VisionResult` carries the FEN, an overall + per-square confidence, and
   a `detection_failed` flag. On failure it falls back to the starting
   position so you can set the board up manually.

**Phase 2 — web app (`app/`)**

- FastAPI + Jinja2 + HTMX. `app/engine.py` drives Stockfish over UCI
  (`multipv=3`, depth 18, 1.0 s, 5 s hard timeout). `app/session.py` keeps
  per-user state in an in-memory dict keyed by a cookie UUID.

### The vision backend is pluggable

`vision.py` defines a `VisionBackend` protocol. The shipped `ClassicalBackend`
needs no model download. A YOLO or hosted-API backend can be dropped in as a
single new class — see *Known limitations* and *Roadmap*.

---

## Known limitations

- **Vision quality depends heavily on the photo.** The classical backend
  works well on clean, top-down (overhead) board images — rendered diagrams,
  scans, flat phone photos shot straight down. **Oblique angles, shadows,
  glare, and unusual piece sets will reduce accuracy or fail outright.**
  When detection fails, the app falls back to the starting position and asks
  you to set the board up by clicking squares. Always sanity-check the parsed
  board before trusting the engine.
- **Piece-type recognition is template-based.** It is tuned to a standard
  2D piece set. A real-world model backend (YOLO / hosted API) would
  generalize far better — it was the intended approach but both Hugging Face
  and Roboflow are blocked by this environment's network policy (see below).
- **90° / mirrored orientations** are not auto-corrected — only the 180°
  "which side is closer" case is handled, via the orientation control.
- **Single user, single process.** Session state is in-memory, so run one
  uvicorn worker and expect state to reset when the server restarts.
- **Build-environment note:** the brief specified Tailwind + HTMX via CDN,
  but this environment's network policy blocks those CDNs. HTMX is therefore
  vendored in `static/`, and styling is a hand-written `static/app.css` —
  still no build step, and the app works fully offline.

---

## Roadmap

- **Swap in a real vision backend.** Implement `VisionBackend` with a YOLO
  chess-piece detector (e.g. a fine-tuned YOLO11) or a hosted inference API
  for robust real-photo recognition at oblique angles.
- **Add a YOLO option** selectable at runtime, falling back to the classical
  backend when no model/network is available.
- **Better board detection** — corner refinement and automatic 90°/mirror
  orientation correction.
- **Multi-position game replay** — analyze a sequence of photos as a game,
  step through moves, and show an evaluation graph.
- **Engine controls** — expose depth/time/skill level in the UI.
- **Persistent sessions** — move session state to Redis for multi-user use.

---

## Tests

```bash
pytest -v
```

- `test_vision_synthetic.py` / `test_vision_fallback.py` / `test_orientation.py`
  — vision pipeline on synthetic fixtures, the failure fallback, and FEN
  rotation. *Synthetic fixtures test the rendering / segmentation / FEN
  layers only — not real-world vision, which needs real photos.*
- `test_engine.py` — Stockfish on a mate-in-1, an equal middlegame, and a
  drawn K-vs-K endgame.
- `test_routes.py` — every HTTP route, plus the edge cases (bad uploads,
  illegal positions, engine-before-analyze).

---

## Project layout

```
chess_vision/      Phase 1 — vision pipeline + CLI
  vision.py          image -> FEN, pluggable VisionBackend
  orientation.py     180° FEN/grid rotation
  render.py          FEN -> SVG/PNG, side-by-side visualization
  cli.py             python -m chess_vision.cli
app/               Phase 2 — web app
  main.py            FastAPI routes
  engine.py          Stockfish (UCI) wrapper
  session.py         in-memory session store
templates/         Jinja2 + HTMX templates
static/            CSS + vendored htmx.min.js
tests/             test suite + synthetic fixtures
```

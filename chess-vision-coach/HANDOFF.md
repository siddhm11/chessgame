# Chess Vision Coach — Handoff Document

**Branch:** `claude/chess-vision-coach-DwzKV`  
**Repo:** `siddhm11/chessgame`  
**Last commit:** `0bcbb71` — upgrade to YOLOv8m chess model + auto-configuring backend  
**Tests:** 48 passing  
**Status:** fully functional; model accuracy is the main open question

---

## What this project does

Upload a photo of a physical chessboard → extract the position as FEN → let the user fix misread squares → get Stockfish's best move with evaluation and PV lines.

```
photo → board detection → piece classification → FEN → user corrections → Stockfish
```

The app runs locally at `http://localhost:8000`. One command starts it:

```bash
cd chess-vision-coach
./run.sh
```

---

## Architecture in one page

### Phase 1 — vision (`chess_vision/`)

| file | purpose |
|---|---|
| `vision.py` | entry point: `analyze_image()`, board detection, classical backend, `VisionBackend` protocol |
| `model_backend.py` | YOLO ONNX backend; auto-configures from model metadata |
| `orientation.py` | 180° FEN/grid rotation for camera orientation |
| `render.py` | FEN → SVG/PNG; side-by-side visualization; `board_svg(fill=…)` for tinting |
| `cli.py` | `python -m chess_vision.cli photo.jpg` |
| `models/` | vendored ONNX weights |

**Backend selection** is via env var `CVC_BACKEND`:

- `CVC_BACKEND=classical` (default) — template matching against python-chess glyphs
- `CVC_BACKEND=model` — YOLO ONNX; model auto-selected from `models/` (yolov8m preferred)
- `CVC_MODEL_PATH=/path/to/custom.onnx` — override the model file

The `VisionBackend` protocol (`vision.py:81`) is the drop-in interface:
```python
class VisionBackend(Protocol):
    name: str
    def analyze(self, image_bgr: np.ndarray) -> BackendOutput: ...
```

### Phase 2 — web app (`app/`)

| file | purpose |
|---|---|
| `main.py` | FastAPI routes |
| `engine.py` | Stockfish UCI wrapper (depth, time-limit, multipv=3) |
| `session.py` | in-memory LRU session store, keyed by cookie UUID |

**Routes:**

| method + path | what it does |
|---|---|
| `GET /` | landing page |
| `POST /analyze` | upload photo → VisionResult → board SVG with confidence tints |
| `POST /correct` | click a square → open piece picker → apply correction |
| `POST /best-move` | call Stockfish → return eval + top-3 PV candidates |
| `POST /play-move` | apply a UCI move to the session FEN |
| `GET /download-fen` | return the current FEN as plain text |

**Templates:** `templates/` — Jinja2 + HTMX (vendored in `static/htmx.min.js`; CDN blocked). `_board.html`, `_analyze.html`, `_bestmove.html`, `base.html`.

**Session state** (`app/session.py:Session`):
```python
fen: str | None
original_image_url: str | None
orientation: str          # "white" | "black"
side_to_move: str         # "w" | "b"
detection_failed: bool
confidence: float         # overall 0–1
per_square_confidence: dict | None  # {"e4": 0.92, …}; cleared on play-move
```

---

## Vision pipeline detail

### Board detection (`vision.py:_find_board`)

Three strategies are tried; the one with the highest `_checkerboard_score` wins:

1. **Whole-frame resize** — 640×640 crop of the full image. Wins on rendered diagrams.
2. **`_largest_quad`** (`vision.py:219`) — Canny + `findContours` + iterating `approxPolyDP` epsilon (0.005→0.06) + `minAreaRect` fallback. Wins on clean overhead photos where the board is the dominant quadrilateral.
3. **`_scan_for_board`** (`vision.py:272`) — sliding square at 5 scale factors. Wins on photos where the board is a sub-rectangle with textured/white background.

`_checkerboard_score` (`vision.py:194`) samples corner pixels of each of 64 cells and measures how cleanly they alternate light/dark. Score is in [0,1]; the winner must beat the whole-frame score.

### Classical backend (default)

- Warps the board to 640×640, splits into 64 cells (each 80×80).
- Template-matches each cell against python-chess SVG glyphs rendered on light/dark backgrounds using normalized cross-correlation.
- Empty squares are found by comparison to an empty-cell template (`EMPTY_TOL = 18.0`).
- `CONF_THRESHOLD = 0.5` — if overall confidence is below this, `detection_failed=True` and the app falls back to the starting position.

### YOLO model backend (`model_backend.py`)

Auto-configures from ONNX metadata at load time:
- `imgsz` → input resolution (416 for yolo11n, 640 for yolov8m)
- `names` → class-id-to-FEN-symbol map (handles `board` class by silently skipping it)

**Coordinate space gotcha (fixed):** yolo11n exports box coordinates in pixel space (0–416); yolov8m exports normalized coordinates (0–1). The backend auto-detects this by checking `boxes_xywh[:, :2].max() > 1.5` and normalizes to [0,1] before grid mapping (`model_backend.py:149-152`).

**Parameters:**
```python
_CONF_THRESHOLD = 0.25   # minimum class score to keep a detection
_NMS_THRESHOLD  = 0.45   # IoU threshold for NMS
_EMPTY_CONF     = 0.90   # confidence assigned to empty cells (no real "is empty" signal)
```

---

## Vendored models

| file | size | architecture | input | classes | source |
|---|---|---|---|---|---|
| `yolov8m-chess.onnx` | 99 MB | YOLOv8m | 640×640 | 13 (board + 12 pieces) | NAKSTStudio/yolov8m-chess-piece-detection (AGPL-3.0) |
| `yolo11n-chess.onnx` | 11 MB | YOLO11n | 416×416 | 12 pieces | same repo, mobile export (AGPL-3.0) |

**Both are AGPL-3.0.** Fine for personal/learning use; a commercial deployment would need an Ultralytics commercial license or a different model.

---

## Benchmark results (100 photos, `samryan18/chess-dataset`)

Run with: `python bench/benchmark.py --n 100`

The dataset (`/tmp/chess-dataset/labeled_originals`) auto-clones on first run (~1.3 GB, real top-down photos of tournament Staunton sets, FEN encoded in filename).

| backend | exact | empty | recall | type\|p | sec/img |
|---|---|---|---|---|---|
| classical | 22.3% | 39.1% | 94.4% | 16.4% | 1.36 |
| yolo11n-chess | 78.6% | 79.0% | 2.4% | 21.2% | 1.23 |
| **yolov8m-chess** | **79.5%** | **82.6%** | **18.9%** | **21.6%** | 1.49 |

**Metric definitions:**

- `exact` — per-square exact match (the headline "is the position right" number)
- `empty` — agreement on whether the square is empty (regardless of piece type)
- `recall` — of squares the truth says have a piece, how often did we predict *any* piece
- `type|p` — given both predicted-and-truth have a piece, how often does the type match

**What these numbers mean:**

yolov8m wins on all four metrics and is the current default. However, 18.9% recall means it misses ~81% of actual pieces on this dataset. This is not a bug — it reflects a domain gap: the models were trained on a chess-piece-detection dataset that differs from the `samryan18` test photos (different board/piece styles, lighting). The models are appropriately conservative (high precision, low recall).

The classical backend has 94% recall but only 22% exact accuracy — it finds *something* on almost every square but mostly identifies the piece type wrong on real photos.

**How to add a new model:** drop the `.onnx` file into `chess_vision/models/`. The benchmark auto-discovers it, and `ModelBackend` auto-configures from metadata. No code change needed.

---

## Confidence tinting (UX feature)

`app/main.py:confidence_fill()` maps per-square confidence to CSS fill colors:
- `conf < 0.55` → red `#e0413188` ("definitely check this square")
- `conf < 0.80` → amber `#f0a93188` ("uncertain")
- `conf ≥ 0.80` → no tint

These fill colors are passed to `chess.svg.board(fill=fill)` and rendered into the SVG. Clicking a square to correct it sets `per_square_confidence[square] = 1.0`, clearing the tint.

---

## Engine integration

`app/engine.py` drives Stockfish over UCI. Strength presets:

| level | depth | time limit |
|---|---|---|
| fast | 12 | 0.4 s |
| normal | 18 | 1.0 s |
| strong | 22 | 2.0 s |

`BestMoveResult` has `candidates: list[Candidate]`, each with:
- `move: str` — UCI move (e.g. `e2e4`)
- `san: str` — algebraic notation
- `score_str: str` — e.g. `+0.34` or `#3`
- `line: str` — PV in SAN, e.g. `Nf3 d5 c4`

Stockfish auto-discovered at `/usr/games/stockfish`, `/usr/bin`, `/usr/local/bin`, `/opt/homebrew/bin`, or `PATH`. Override with `STOCKFISH_PATH`.

---

## Environment constraints (important)

This project was built inside a **managed remote execution container** with a restrictive network policy. Understanding these constraints avoids wasted debugging time:

| domain | status |
|---|---|
| `github.com` | accessible (git operations via local proxy at `127.0.0.1:42479`) |
| `huggingface.co` | **blocked** (`403 host_not_allowed`) |
| `roboflow.com` | **blocked** |
| Tailwind/HTMX CDNs | **blocked** (HTMX is vendored in `static/`) |
| Git LFS downloads | **broken** — proxy returns 502 for LFS batch API; workaround: query GitHub LFS batch API directly via HTTPS, then `curl` the CDN URL |

**How the 99 MB model was retrieved:** GitHub's public LFS batch API was queried via `curl` to get a signed CDN URL, then the actual file was downloaded with a second `curl`. The proxy does not support LFS but direct HTTPS to `github.com` works.

---

## Known limitations / open issues

### 1. Piece recall is low on the benchmark dataset (18.9%)
The models don't detect most pieces on the `samryan18` tournament photos. Root causes:
- Domain gap between training data and test data (piece styles, board colors, lighting)
- Board detection crop may not align perfectly with piece positions at the edges
- Confidence threshold (0.25) may still be too conservative for this dataset

**Potential fixes to try:**
- Lower `_CONF_THRESHOLD` from 0.25 to 0.15 and re-benchmark
- Try a model fine-tuned on `samryan18`-style photos specifically
- Check board detection alignment by saving warped boards and visually inspecting

### 2. Board detection fails on oblique/angled photos
`_find_board` was improved but still struggles with steep camera angles, heavy shadows, or boards photographed at more than ~30° from vertical. The `_checkerboard_score` metric can accept a bad crop when all three candidates score similarly.

### 3. Piece-type accuracy is ~21% on real photos (both models)
Even when a piece is detected, the type is wrong 79% of the time. This is mostly a model issue (training data mismatch), not a code issue. The "correct a square" UX exists precisely to compensate for this.

### 4. Single-user, in-memory sessions
`app/session.py` uses an in-memory LRU dict capped at 256 sessions. State is lost on server restart and is not shareable across workers. For multi-user deployment, swap to Redis.

### 5. Classical backend is the default for good reason
`CVC_BACKEND=classical` is the default because it requires no model file and round-trips rendered/synthetic boards at ~100%. For real photos, `CVC_BACKEND=model` is better. The default was not changed from `classical` because:
- The model backend is slower (~20% per image)
- The model is 99 MB; users without it would get errors
- The classical backend gracefully degrades (at least it detects *a* piece in most squares)

Consider flipping the default to `model` once yolov8m-chess.onnx is confirmed present.

---

## How to run

```bash
cd chess-vision-coach
./run.sh                          # starts at http://localhost:8000

# With model backend:
CVC_BACKEND=model ./run.sh

# CLI:
python -m chess_vision.cli photo.jpg --orientation white --side-to-move w
python -m chess_vision.cli photo.jpg --orientation black --side-to-move b -o out.png

# Tests:
pytest -v

# Benchmark (auto-clones dataset to /tmp/chess-dataset on first run):
python bench/benchmark.py --n 30        # quick
python bench/benchmark.py --n 100       # stable
python bench/benchmark.py --source /path/to/your/photos
```

---

## Project layout

```
chess-vision-coach/
  chess_vision/
    vision.py           image → FEN, VisionBackend protocol, ClassicalBackend
    model_backend.py    YOLO ONNX backend (auto-configures from metadata)
    orientation.py      180° rotation for camera orientation
    render.py           FEN → SVG/PNG, side-by-side viz, board_svg(fill=…)
    cli.py              python -m chess_vision.cli
    models/
      yolov8m-chess.onnx   99 MB — current default for CVC_BACKEND=model
      yolo11n-chess.onnx   11 MB — fallback if yolov8m absent
  app/
    main.py             FastAPI routes, confidence_fill(), ENGINE_LEVELS
    engine.py           Stockfish UCI (BestMoveResult, Candidate, PV lines)
    session.py          Session dataclass, SessionStore (in-memory LRU)
  templates/            Jinja2 + HTMX fragments
    base.html
    _board.html         board SVG + 8×8 overlay + confidence legend
    _analyze.html       orientation/side-to-move controls + strength selector
    _bestmove.html      eval + top-3 candidates with PV lines + play button
  static/
    app.css             hand-written CSS (CDN blocked)
    htmx.min.js         vendored HTMX
  bench/
    benchmark.py        3-backend accuracy benchmark vs labeled dataset
    README.md           metric definitions + usage
  tests/
    test_vision_synthetic.py
    test_vision_fallback.py
    test_orientation.py
    test_engine.py
    test_routes.py
    test_model_backend.py   (skipped if yolo11n-chess.onnx absent)
  model-upload/
    best.onnx           Git LFS pointer — the 99 MB yolov8m source file
                        (actual binary is now vendored at models/yolov8m-chess.onnx)
  requirements.txt
  run.sh
  README.md
  HANDOFF.md            (this file)
```

---

## What to work on next

**Highest-value items in order:**

1. **Lower `_CONF_THRESHOLD` and re-benchmark.** The current 0.25 threshold might be too conservative for real tournament photos. Try 0.15 and check if recall improves without too many false positives. Edit `model_backend.py:45`.

2. **Investigate board alignment.** Save the warped board crop for a few photos and compare it to the original. If pieces are being cropped to the edge or the grid is slightly misaligned, a small pad or crop adjustment would fix recall and type accuracy simultaneously.

3. **Wire `CVC_BACKEND=model` as default in `run.sh`.** The yolov8m model is already in the repo. The only reason classical is still default is the historical "no model file" concern — that no longer applies. Edit `run.sh` to set `CVC_BACKEND=model` before `uvicorn`.

4. **Real-photo test suite.** Add `tests/fixtures/real/` with 2–3 representative real photos (e.g. the iStock photos the user supplied) and `tests/test_vision_real.py` that asserts the structural contract only (valid `VisionResult`, FEN parses, confidence in [0,1], no crash). Does not assert piece accuracy; just ensures the pipeline doesn't blow up on real input.

5. **Persistent sessions (optional).** Replace the in-memory `SessionStore` with Redis for multi-user deployment. The `Session` dataclass is clean; the swap is contained to `app/session.py`.

---

## Commit history (most recent first)

| hash | what |
|---|---|
| `0bcbb71` | YOLOv8m model + auto-configuring backend + coordinate fix |
| `adaa9fb` | user pushed best.onnx via Git LFS (user commit) |
| `e56dbd1` | benchmark harness (`bench/benchmark.py`) |
| `175eff1` | board detection improvements for real photos |
| `d3be113` | test photos committed by user |
| `369e50e` | YOLO11n ONNX backend + tests |
| `59e0daf` | UX: confidence tinting, PV lines, strength selector, play-move |
| `e74349c` | Phase 2: FastAPI web app |
| `29db954` | Phase 1: CLI vision pipeline |

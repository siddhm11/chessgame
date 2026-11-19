# Chess Vision Coach — Handoff Document

**Branch:** `claude/chess-vision-coach-DwzKV`  
**Repo:** `siddhm11/chessgame`  
**Last commit:** piece model retrained on ChessReD2K + samryan18  
**Tests:** 211 passed (CI green on every push)  
**Status:** reads two board styles (samryan18, ChessReD) at ~99% per square; weights non-commercial.
**Read `bench/RESULTS.md` and `README.md` first.**

- Piece model retrained on ChessReD2K + samryan18 (`prepare_mixed_dataset.py`).
  samryan18 held-out val 99.66% per square (65/68 perfect); ChessReD2K test
  (other set, angled phone photos, never trained on) **99.15%** with the
  annotated corners, **99.13%** with automatic board finding (~4 in 5
  boards perfect). Silent errors 0.2-0.4 per board. See `bench/RESULTS.md`.
- **Weights are non-commercial**: ChessReD is CC BY-NC-SA 4.0, on top of
  the AGPL-3.0 of the Ultralytics fine-tune.
- Other piece sets (iStock photos): still read wrong, but every square is
  tinted (impossible-material rule), 0 silent errors.
- Largest remaining silent-error source on angled photos: the automatic
  side-view turn (wrong on 29/169 ChessReD val side views, 3/162 test).
- Product: drag-the-corners board fix, non-chessboard rejection, works
  inside a Hugging Face Space iframe, public-deployment safeguards (body
  limit, image bombs, global engine/vision limits, session handling).
- **Free deploy: `web/`**, the whole app in the browser (Pyodide runs the
  same `chess_vision` package; onnxruntime-web or OpenCV DNN runs the
  models; Stockfish WASM). `.github/workflows/pages.yml` builds, tests in
  Chromium (both runtimes) and publishes to GitHub Pages from `main`
  (needs Settings -> Pages -> Source: GitHub Actions). ~2.3 s per photo.
- Board-corner detector (`chess_vision/board_corners.py`,
  `models/board-corners.onnx`): angled ChessReD boards found 305/306 (was
  213); optional, a no-op without the file.
- Server deploy: Dockerfile (pinned, never built here) or
  `deploy_hf_space.py` (HF needs PRO).

---

## What this project does

Upload a photo of a physical chessboard -> position (FEN) -> the user fixes
anything flagged -> Stockfish's best move with evaluation and lines.

```
photo -> find board -> read pieces -> chess-rule checks -> user fixes -> Stockfish
```

Two front ends share the same vision package: a FastAPI server (`app/`) and
a static in-browser build (`web/`).

---

## Architecture in one page

### Vision (`chess_vision/`, pure Python; also runs in Pyodide)

| file | purpose |
|---|---|
| `vision.py` | `analyze_image()`, image loading (EXIF, size caps), `ClassicalBackend` (board finding + template-matching fallback) |
| `board_corners.py` | optional learned corner detector -> extra board candidate |
| `model_backend.py` | YOLO piece reader (TTA, rotation recovery, doubt gate); onnxruntime or OpenCV DNN |
| `sanity.py` | chess-rule repairs/flags, impossible material |
| `position.py` | `fen_problem()`, `confidence_fill()` (shared by server + browser) |
| `orientation.py`, `render.py`, `cli.py` | 180-degree rotation, board SVG, CLI |

`CVC_BACKEND=model` (default; falls back to `classical` with a warning if
the model or runtime is missing). `CVC_MODEL_PATH`, `CVC_RUNTIME`
(`auto|onnxruntime|opencv`), `CVC_CORNER_MODEL` (`0` disables).

### Server (`app/`)

`main.py` routes: `GET /`, `POST /analyze` (photo or re-read with user
corners), `GET /photo/{token}`, `POST /correct`, `POST /best-move`,
`POST /play-move`, `GET /download-fen`. `engine.py`: Stockfish UCI.
`session.py`: in-memory sessions (header + cookie), cap 4096. Safeguards:
body limit, image-bomb check, global vision/engine semaphores, per-session
single flight. Templates: Jinja2 + vendored htmx.

### Browser (`web/`)

`src/index.html` + `app.js` (UI), `vision-worker.js` (Pyodide + models),
`py/cvc_web.py` (glue). `tools/build.mjs` -> `dist/` static site;
`tools/e2e.mjs` headless Chromium test. See `web/README.md`.

---

## Vision pipeline detail

### Board finding (`vision.py:ClassicalBackend.locate_board`)

Candidates, each snapped onto the exact 8x8 grid by `_refine_quad` (smooth
search score, explicit +/-1-square shift tests, then a sharp corner-patch
score to fine-tune and judge); the best sharp fit wins:

1. whole frame; 2. largest contour quads (dilate 1/2/0); 3. the learned
corner detector (`board_corners.py`, `models/board-corners.onnx`, optional);
4. a sliding scan, only when nothing cheaper fits well (>= 0.90).

Fit < `NO_BOARD_FIT` (0.68): "no chessboard" (0/508 real boards rejected,
94/94 ordinary photos rejected). Fit < 0.80: the whole read is tinted as
doubtful. User-dragged corners (`locate_board(img, corners)`) skip the search
but get a guarded snap.

### Piece reading (`model_backend.py`)

YOLOv8n @640 on the warped board, mirrored second pass (TTA; disagreements
always tint), sanity layer (`sanity.py`: back-rank pawns, kings, pawn count,
impossible material). Sideways boards: square-colour parity of all 64
squares decides; the clockwise/anticlockwise choice prefers both kings,
then White's army lower on the board, and adds a note with the one-tap fix.
Runtime: onnxruntime, else OpenCV DNN (identical results; `CVC_RUNTIME`).
Class map and input size come from the ONNX metadata (read directly from the
protobuf), so dropping in a retrained model needs no code change.

---

## Vendored models

| file | size | role |
|---|---|---|
| `yolov8n640-chess-finetuned.onnx` | 13 MB | **default piece model**: trained in-repo on ChessReD2K + samryan18 @640 (AGPL-3.0 and CC BY-NC-SA 4.0: non-commercial). samryan18-only weights: git history at 008fc66 |
| `board-corners.onnx` | 13 MB | board-corner detector (YOLOv8n, 1 class), trained on ChessReD2K train |
| `yolov8n-chess-finetuned.onnx` | 12 MB | old @416 fallback |
| `yolov8m-chess.onnx`, `yolo11n-chess.onnx` | 99 / 11 MB | third-party baselines, unused |

All YOLO weights are AGPL-3.0 (Ultralytics). Commercial use needs a
commercial Ultralytics license or another detector, *and* weights trained
without ChessReD.

Accuracy: see `bench/RESULTS.md` (samryan18 + ChessReD) and
`bench/RESULTS_more_sets.md` (sets 3 and 4, corner detector, combined
model).

---

## Confidence tinting

`chess_vision/position.py:confidence_fill()` (shared by server and browser):
conf < 0.55 red, < 0.80 amber, else no tint. A corrected square gets 1.0.
"Silent" errors = wrong and not tinted; that is the metric to watch.

## Engine

Server: `app/engine.py`, Stockfish over UCI, presets fast (d12/0.4 s), normal
(d18/1 s), strong (d22/2 s), MultiPV 3. Browser: Stockfish 19 lite WASM with
the same presets (`web/src/app.js`).

## Known limitations

1. **Unseen piece sets** read badly (set 3 tournament set ~85%, set 4 wooden
   set ~50% per square before the combined retrain). The impossible-material
   rule tints such boards. More sets need labelled photos.
2. **Side views**: the auto-turn is wrong on 29/169 ChessReD val side views
   (3/162 test); the note tells the user how to flip.
3. **Server**: in-memory sessions, one worker; one 4-core box reads about one
   photo at a time (queues, doesn't fail). The browser build has no server.
4. **Browser**: first visit downloads ~60 MB; fast path needs JSPI
   (Chrome/Edge 137+), others take ~3x longer. Phones roughly 2x slower.
5. **License**: weights non-commercial (see above).

---

## How to run

```bash
cd chess-vision-coach
./run.sh                                   # server at http://localhost:8000
python -m chess_vision.cli photo.jpg       # CLI
pytest -q                                  # tests
CVC_BACKEND=model python bench/full_eval.py            # samryan18 accuracy
python bench/full_eval.py --photos DIR --use-corners   # any labelled folder
cd web && npm ci && node tools/fetch_runtime.mjs && node tools/build.mjs \
  && node tools/serve.mjs 8080             # browser build
```

## Project layout (main parts)

```
chess-vision-coach/
  chess_vision/          vision package (+ models/)
  app/, templates/, static/   server
  web/                   in-browser build (src/, tools/)
  bench/                 evaluation scripts + RESULTS*.md
  tests/                 pytest suite (fixtures/ with CC0 non-board photos)
  prepare_mixed_dataset.py, finetune_train.py   retraining
  Dockerfile, deploy_hf_space.py               server deploy
```

---

## Retrain on ChessReD (how the current default model was made)

Data: ChessReD annotations.json + chessred2k.zip (not the 24.6 GB full
zip) from data.4tu.nl, unpacked OUTSIDE the repo; never commit dataset
files; read with `python -I`. Exact commands, hyperparameters and the
old-vs-new gate are in `bench/RESULTS.md` ("Retrain on ChessReD").
Summary: `prepare_mixed_dataset.py` (1441 ChessReD2K train + 388 samryan18
train, crops made exactly like the app's model input, cell-sized boxes
from the annotated squares) -> `finetune_train.py` with
`CVC_FT_DATA/IMGSZ=640/EPOCHS=25/PATIENCE=6`, yolov8n.pt base, lr0 0.002,
degrees 8, fliplr 0.5, mosaic 0; ~2.5 h on 4 CPU cores. For a combined
retrain with more sets, add their crops to the same layout.

Evaluating: `bench/chessred_prepare.py` (orientation label now matches
what the app shows; the old label flipped half the side views) and
`bench/full_eval.py --photos DIR [--use-corners]`; `bench/square_dump.py`
dumps per-square truth/pred/conf for calibration work (use val, not test).

## Fine-tuning (how the previous samryan18-only model was made)

The 18.9% recall of the pre-trained models was a domain gap, not a tuning
problem (lowering `_CONF_THRESHOLD` to 0.15 and CLAHE preprocessing both
failed to help — verified). The fix was to fine-tune on real photos:

```bash
# 1. Build YOLO training data from the labeled dataset (no GPU; ~3 min).
#    Crops/warps each board, derives per-cell boxes from the filename FEN,
#    filters low-quality crops, writes 388 train / 68 val to finetune_data/.
python prepare_finetune_dataset.py

# 2a. Train on CPU (~14 min, yolov8n @ 416, 20 epochs) — what produced the
#     first fine-tuned model:
python finetune_train.py
#     -> exports chess_vision/models/yolov8n-chess-finetuned.onnx

# 2b. OR train on GPU for higher ceiling (yolov8m @ 640, 50 epochs):
#     open finetune_colab.ipynb in Google Colab (T4).

# 3. Benchmark on the HELD-OUT val originals (leak-free):
mkdir -p /tmp/chess-val-originals
for f in finetune_data/images/val/*.jpg; do
  cp "/tmp/chess-dataset/labeled_originals/$(basename "$f" .jpg).JPG" /tmp/chess-val-originals/
done
python bench/benchmark.py --source /tmp/chess-val-originals --n 68
```

Result at the time: 98.3% per square on the 68 held-out photos (later 99.8% with @640, sanity layer and TTA).

## What to work on next

1. **More piece sets.** Biggest lever. Each set needs a few hundred labelled
   photos (photograph a game move by move: the moves label the photos);
   add them to `prepare_mixed_dataset.py`'s layout and retrain (~2-3 h CPU).
2. **Side-view orientation** (29/169 ChessReD val side views upside down):
   try a pawn/king-placement prior, validate on ChessReD val and
   `bench/sideways_tiebreak.py`.
3. **Commercial-safe model**: retrain without ChessReD (and with a
   non-AGPL detector) if the app is ever sold.
4. **Browser download size**: fp16 models would roughly halve the ~25 MB of
   ONNX weights.

History: `git log` (commit messages carry the measured numbers).

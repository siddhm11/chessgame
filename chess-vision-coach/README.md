# Chess Vision Coach

Photograph a physical chessboard; get the position and the best move.

```
photo -> find the board -> read the pieces -> check against chess rules
      -> you fix anything flagged -> Stockfish suggests the best move
```

## What it does

1. **Upload a photo** (phone or camera, up to 25 MB).
2. The app **finds the board** and outlines it on your photo. If the outline
   is wrong, **drag its 4 corner dots** and press *Re-read board*.
3. A fine-tuned **YOLO model reads every square**. Chess-rule checks repair
   impossible reads (two white kings, pawns on the back rank, 11 queens).
4. Squares the app is unsure of are **tinted amber/red**, with a note saying
   why. **Click any square** to fix it.
5. **Stockfish** gives the best move, an evaluation and the follow-up line;
   one click plays the move.

Photos that don't contain a chessboard are rejected instead of being read.

## Two ways to run it

- **In the browser (free, no server):** `web/` is a static site that runs the
  same Python vision code, the model and Stockfish entirely on the visitor's
  device. Photos never leave it. Host it free on GitHub Pages; see
  [`web/README.md`](web/README.md).
- **As a server:** FastAPI + htmx (below), for your own machine or any
  container host.

## How accurate is it?

Measured with `bench/full_eval.py` and `bench/score_more_sets.py` on photos
with known positions (details: [`bench/RESULTS.md`](bench/RESULTS.md),
[`bench/RESULTS_more_sets.md`](bench/RESULTS_more_sets.md)):

| photos | squares right | boards perfect | same best move |
|---|---|---|---|
| 68 held-out overhead photos (green/white board, Staunton set) | **99.66%** | 65/68 | 97% |
| 44 harder held-out photos, same set | **99.86%** | 41/44 | 100% |
| 306 ChessReD phone photos (another set, angled), board found automatically | **99.13%** | 249/306 | 88% |

"Silent" errors (wrong *and* not tinted) are 0.2–0.4 per board; everything
else the app is unsure of is tinted so you can tap to fix it.

**Limits.** It knows these two board styles well. On piece sets it has never
seen (a wooden Staunton set, a plastic tournament set) most squares are still
misread; the app flags such boards (every square tinted, plus a note) rather
than pretending. Photos taken from the side of the board are turned automatically;
on ChessReD 3/162 (test) and 29/169 (validation) came out upside down. The app
says when it turned a photo, and one tap flips it.

## Run it locally

You need Python 3.10+ and [Stockfish](https://stockfishchess.org/download/).

**Linux / macOS**

```bash
./run.sh            # creates .venv, installs, checks Stockfish, starts on :8000
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"   # if it's not on PATH
uvicorn app.main:app --port 8000 --workers 1
```

Then open <http://localhost:8000>. Cairo is not needed to run the web app.

Stockfish is found on `PATH`, at `/usr/games/stockfish`, `/usr/local/bin`,
`/opt/homebrew/bin`, or wherever `STOCKFISH_PATH` points.

## Deploy

**Free: GitHub Pages (recommended).** `.github/workflows/pages.yml` builds
the browser version, tests it in headless Chromium and publishes it from
`main`. One-time: repo **Settings -> Pages -> Source: GitHub Actions**, then
merge to `main`. Details: [`web/README.md`](web/README.md).

**Server:** the Dockerfile runs on any container host (Cloud Run, Fly.io,
Render, a VM). `deploy_hf_space.py` targets a Hugging Face Docker Space, which
needs a PRO account; see [`DEPLOY_HF_SPACE.md`](DEPLOY_HF_SPACE.md).

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `CVC_BACKEND` | `model` | `model` (YOLO) or `classical` (template matching; rendered diagrams only) |
| `CVC_VISION_CONCURRENCY` | 1 | photo analyses at once (each already uses every core) |
| `CVC_ENGINE_CONCURRENCY` | 1 | Stockfish searches at once, across all users |
| `CVC_THREADS` | CPU quota | threads for the model/OpenCV (default: the container's CPU limit) |
| `CVC_MAX_SESSIONS` | 4096 | sessions kept in memory |
| `STOCKFISH_PATH` | auto | Stockfish binary |

Run **one** uvicorn worker: sessions live in process memory.

## Public-deployment safeguards

Uploads are size-checked before being read; image "bombs" are refused from
their header; Stockfish and photo analysis are limited globally with "busy,
try again" answers; photos are served only by an unguessable per-photo link
and deleted after 24 h; sessions work inside a cross-site iframe (how a
Hugging Face Space page embeds the app); an expired session asks you to
re-upload instead of showing a wrong board. Each is pinned by a test.

## How it works

- `chess_vision/vision.py`: loads the photo (EXIF-upright, capped at 2000 px);
  finds the board from several candidates (whole frame, contour quads, a
  sliding scan) and snaps each onto the exact 8×8 grid (`_refine_quad`), so
  the best grid fit wins. Below a fit of 0.68 there is no board; below 0.80
  the whole read is tinted as doubtful.
- `chess_vision/board_corners.py`: a small YOLOv8n corner detector adds one
  more board candidate (finds angled boards: 305/306 ChessReD photos).
- `chess_vision/model_backend.py`: YOLOv8n (ONNX, 640 px) trained on
  samryan18 + ChessReD, on the warped board, with a mirrored second pass;
  90° rotation recovery. Runs on onnxruntime, or OpenCV DNN without it.
- `chess_vision/sanity.py`: chess-rule repairs and flags.
- `app/`: FastAPI + htmx pages, Stockfish over UCI, in-memory sessions.

## Tests and evaluation

```bash
pytest -q                                    # 211 tests (CI runs them on every push)
python bench/full_eval.py                    # accuracy on the labelled photo set
python bench/e2e_load.py --users 4           # real HTTP, concurrent users
python bench/smoke_bundle.py                 # the deploy bundle, from an empty folder
python bench/board_presence_eval.py ...      # "is there a board?" thresholds
cd web && node tools/e2e.mjs photo.JPG       # the browser build, end to end
```

## Next steps

- **More piece sets.** The model reads what it was trained on. Each new set
  needs a few hundred labelled photos; the cheapest source is photographing
  a game move by move (the moves label every photo). `prepare_mixed_dataset.py`
  and `finetune_train.py` retrain on CPU in about 2-3 hours.
- **Side-view orientation**: still the main source of silent errors on
  angled photos.

## License

Model weights: Ultralytics YOLO derivatives (AGPL-3.0) **and** trained on
ChessReD (CC BY-NC-SA 4.0), so **non-commercial** use only. A commercial
version needs a model trained without ChessReD and a commercial Ultralytics
license (or another detector). Stockfish and python-chess are GPL-3.0.

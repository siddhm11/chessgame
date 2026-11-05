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

## How accurate is it?

Measured with `bench/full_eval.py` on 636 photo analyses with known positions
(details and how to reproduce: [`bench/RESULTS.md`](bench/RESULTS.md)):

| photos | per-square accuracy |
|---|---|
| 68 held-out photos, same board/pieces as training | **99.8%** |
| 44 harder held-out photos, same set | **99.3%** |
| same, rotated via EXIF / darkened to 55% | 99.7% / 99.8% |

**Important limit: other boards and piece sets.** The model was trained on
one board style (green-and-white roll-up board, Staunton pieces, shot from
above). On two photos of *other* sets it read 37/64 and 30/64 squares wrong.
The app flags those reads (every wrong square tinted, plus a note), but it
can't read them. Making it work on any set needs more varied training data;
see *Next steps*.

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

`python deploy_hf_space.py --space <you>/chess-vision-coach` uploads exactly
the 25 files the app needs to a Hugging Face Docker Space. **Hugging Face now
requires a PRO account to create Docker Spaces**; the script checks this first.
The Dockerfile works on any container host. See
[`DEPLOY_HF_SPACE.md`](DEPLOY_HF_SPACE.md).

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
- `chess_vision/model_backend.py`: fine-tuned YOLOv8n (ONNX, 640 px) on the
  warped board, with a mirrored second pass; 90° rotation recovery.
- `chess_vision/sanity.py`: chess-rule repairs and flags.
- `app/`: FastAPI + htmx pages, Stockfish over UCI, in-memory sessions.

## Tests and evaluation

```bash
pytest -q                                    # 168 tests (CI runs them on every push)
python bench/full_eval.py                    # accuracy on the labelled photo set
python bench/e2e_load.py --users 4           # real HTTP, concurrent users
python bench/smoke_bundle.py                 # the deploy bundle, from an empty folder
python bench/board_presence_eval.py ...      # "is there a board?" thresholds
```

## Next steps

- **Train on other piece sets.** The single biggest improvement. ChessReD
  (10,800 phone photos of another set, CC BY-NC-SA) can't be downloaded from
  the environment this was built in. `bench/chessred_prepare.py` turns it into
  a scored test set; the same corner labels can produce training crops.
- **Deploy**: needs a Hugging Face PRO account or another container host.

## License

Model weights are Ultralytics YOLO derivatives (AGPL-3.0): fine for personal
and learning use; commercial use needs an Ultralytics license or another model.

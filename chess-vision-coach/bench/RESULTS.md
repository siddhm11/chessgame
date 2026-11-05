# Evaluation results

Every number here comes from a script in this folder, run against the
production entry point (`vision.analyze_image` with `CVC_BACKEND=model`, or
the real HTTP server). Model: `yolov8n640-chess-finetuned.onnx`.

## Accuracy — 636 photo analyses

Photos: [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset),
500 real overhead photos labeled with the true position. Each photo falls in
exactly one bucket, based on what the model saw in training:

| set | photos | per-square exact | perfect boards | wrong squares | of which **silent** |
|---|---|---|---|---|---|
| **val** — held out from training | 68 | **99.77%** | 60 (88%) | 10 | **2** |
| **rejected** — never trained on, hard (data prep dropped them) | 44 | **99.33%** | 32 (73%) | 19 | **3** |
| val, stored sideways + EXIF rotate tag | 68 | 99.72% | 59 (87%) | 12 | 3 |
| val, darkened to 55% brightness | 68 | 99.77% | 61 (90%) | 10 | 3 |
| train (optimistic by construction) | 388 | 99.67% | 325 (84%) | 83 | 23 |
| other photos, no ground truth | 5 | — | 5/5 parsed, 0 crashes | — | — |

0 crashes and 0 fallbacks across all 636 analyses.

**Silent** = wrong *and* shown with confidence ≥ 0.80, so the square is not
tinted. That is the number that matters to a user: a tinted wrong square
costs one click; an untinted one costs a wrong coaching answer. Across the
held-out sets it is about **1 silent wrong square per 25 boards**.

Remaining errors are mostly piece *type* confusions on a correctly located
piece (black pawn read as bishop, black king as queen) — the model, not the
geometry.

### What changed to get here (held-out, before → after)

| set | exact | silent wrong squares |
|---|---|---|
| rejected (44) | 86.01% → **99.33%** | 293 → **3** |
| val + dark | 97.45% → **99.77%** | 88 → **3** |
| val (68) | 99.79% → 99.77% (10 vs 9 wrong squares) | 2 → 2 |

1. **Grid refinement** (`ClassicalBackend._refine_quad`). Real boards sit on
   a mat with a coordinate-label strip; the old finder cropped the mat or a
   coarse scan window up to half a square off, so pieces were read on the
   neighbouring square. Each candidate crop is now snapped onto the exact
   8×8 grid: a smooth score to search, explicit ±1-square shift tests, then
   a sharp corner-patch score to fine-tune and to judge.
2. **Board-doubt gate** (`ModelBackend._apply_board_doubt`). On 112 held-out
   photos the 5% with grid fit < 0.80 held 78% of all wrong squares. Those
   boards are now tinted throughout with an explanation, instead of looking
   trustworthy.
3. **Faster**: JPEGs decoded at reduced scale and capped at 2000 px; the
   slow sliding scan only runs when cheaper candidates don't fit well.
   Median analysis **1.46 s → 0.63 s**.

## End-to-end over HTTP — `e2e_load.py`

Real uvicorn server, 4 concurrent users, each with its own session running
upload → analyze → best move → play move → download FEN on held-out photos.

Soak run: 4 users × 16 photos × 8 rounds = **128 full flows**, with the
production settings (`MALLOC_ARENA_MAX=2`, one vision slot):

| | result |
|---|---|
| errors | **0** |
| cross-session leaks | **0** (no user ever saw another user's board) |
| server memory per round | 325, 305, 317, 317, 317, 323, 317, 305 MB — **flat** |
| `/analyze` | p50 2.4 s, p95 3.5 s with 4 users queued (≈0.6 s alone) |
| `/best-move` | p95 ≈ 0.7 s |
| `/play-move`, `/download-fen` | ≤ 0.02 s |

A 3-round × 32-photo run gave FEN exact over HTTP 81/96 (matching the
offline per-board rate on the same photos); the 12 parses that produced an
illegal position were stopped before Stockfish.

Production bugs this test found, all now fixed:
- `/analyze` ran the vision pipeline on the event loop, freezing every
  other user's requests (`/play-move` took up to 1.76 s). Now in a threadpool.
- Analyses then oversubscribed the CPU (`/analyze` p95 21 s). Now limited
  to `CVC_VISION_CONCURRENCY` (default 1; measured: p95 5.3 s with 1 slot
  vs 9.4 s with 2, on 4 cores).
- A race in the doubt gate: concurrent requests could read each other's
  board fit (measured ~12% of calls). Pinned by
  `tests/test_concurrency.py`, which fails 5/5 against the old pattern.
- Memory crept 350 → 479 MB over 8 rounds from glibc per-thread arena
  fragmentation (not a leak: non-monotonic, and gone with
  `MALLOC_ARENA_MAX=2`, now set in the Dockerfile and `run.sh`).

## Other boards and piece sets — the real limit

All labelled photos above show one board style. On the two photos of other
sets available (`test_photos/`, iStock): a wooden Staunton set in the
starting position read **37/64 squares wrong**; another midgame read
**30/64 wrong** (occupancy and colour; I couldn't label its piece types
reliably by eye). The model doesn't recognise those pieces.

What the app does about it: a chess-rule check (extra queens/rooks/bishops/
knights can't exceed 8 minus the pawns on the board — it fires on 0 of the
500 real positions) tints the whole board and explains the piece set may be
unfamiliar. Silent errors on both photos: **0**. The read is still wrong; it
just no longer looks right.

Fixing it needs training data from many sets. ChessReD (10,800 phone photos,
another set, angled) is the obvious source, but its host is blocked from the
build environment. `bench/chessred_prepare.py` + `full_eval.py --photos` score
it in one command once downloaded elsewhere.

## Is there a board at all? — `board_presence_eval.py`

508 real board photos scored a grid fit of 0.732–1.01; 96 ordinary non-chess
photos (OpenCV and scikit-image samples) 0.48–0.63. The cut-off 0.68 rejects
all 96 and wrongly rejects 0 of 508 boards. Calibration checkerboards and
empty checkerboards (hard negatives) can pass; they read as empty boards.

## Honest limits

- **One board style.** See above: on other sets the reads are flagged but wrong.
- **Overhead only.** Steep angles are not handled by the board finder.
- **CPU-bound.** One 4-core box serves one analysis at a time; more users
  means more queueing, not failures.

## Reproduce

```bash
git clone --depth 1 https://github.com/samryan18/chess-dataset /tmp/chess-dataset
python prepare_finetune_dataset.py          # recreates the exact train/val split
CVC_BACKEND=model python bench/full_eval.py --json results.json
python bench/e2e_load.py --users 4 --photos 32 --rounds 3
```

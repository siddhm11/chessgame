# Evaluation results

Every number here comes from a script in this folder, run against the
production entry point (`vision.analyze_image` with `CVC_BACKEND=model`).
Model: `yolov8n640-chess-finetuned.onnx`, retrained on ChessReD2K +
samryan18 (see "Retrain on ChessReD" below). Board finding includes the
learned corner model (`board-corners.onnx`).

## Accuracy — current model

**Silent** = wrong *and* shown with confidence ≥ 0.80, so the square is not
tinted. That is the number that matters: a tinted wrong square costs one
click; an untinted one costs a wrong coaching answer. **False tints** =
correct squares tinted anyway. **Best move** = Stockfish (depth 12, White
to move) picks the same move on the predicted and the true position; an
illegal predicted position counts as a disagreement.

| set | photos | exact | perfect boards | silent / board | false tints / board | best move same |
|---|---|---|---|---|---|---|
| samryan18 **val** (held out) | 68 | **99.66%** | 65 (96%) | 0.19 | 0.06 | 61/63 (97%) |
| samryan18 **rejected** (never trained on) | 44 | **99.86%** | 41 (93%) | 0.07 | 1.64 | 42/42 (100%) |
| val, sideways + EXIF rotate tag | 68 | 99.95% | 67 (99%) | 0.03 | 0.10 | 63/63 |
| val, darkened to 55% | 68 | 99.89% | 64 (94%) | 0.04 | 0.03 | 60/63 |
| **ChessReD2K test**, annotated corners given | 306 | **99.15%** | 251 (82%) | 0.22 | 3.68 | 261/300 (87%) |
| **ChessReD2K test**, board found automatically | 306 | **99.13%** | 249 (81%) | 0.36 | 3.81 | 263/300 (88%) |
| other photos, no ground truth | 5 | — | 5/5 parsed, 0 crashes | — | — | — |

ChessReD2K test is a different board, piece set, camera and room from
samryan18, mostly angled phone photos; none of its photos were trained on.
Board finder within ¼ square of the true corners on 305/306 (0 no-board
fallbacks). 0 crashes across all sets.

The samryan18 split is pinned in `bench/samryan18_split.json` (the
original 388 train / 68 val; re-running `prepare_finetune_dataset.py` today
picks a slightly different split). The old README figure of 99.77% on val
predates the sideways tie-break (008fc66), which costs one photo; the old
model scores 99.49% on today's code.

### Retrain on ChessReD — old vs new model, same code, same split

| set | old model | new model |
|---|---|---|
| samryan18 val | 99.49%, 59/68 perfect, silent 0.18/board, best move 56/63 | **99.66%**, 65/68, 0.19/board, 61/63 |
| samryan18 rejected | 99.33%, 32/44, silent 0.07/board, 37/42 good move | **99.86%**, 41/44, 0.07/board, 42/42 |
| ChessReD2K test + corners | 62.77%, 0/306 perfect, silent 3.63/board, false tints 15.4/board | **99.15%**, 251/306, 0.22/board, 3.68/board |
| ChessReD2K test, auto | 63.09%, 0/306, silent 3.47/board | **99.13%**, 249/306, 0.36/board |
| iStock starting position (other set) | 37/64 wrong, all 64 tinted, 0 silent | 28/64 wrong, all 64 tinted, 0 silent |

Gate (all met): no regression on samryan18 val or rejected; clear gain on
ChessReD2K test; iStock photos still fully flagged (0 silent).
`board_presence_eval.py`: NO_BOARD_FIT 0.68 still rejects 0/508 boards and
all 94 ordinary non-board photos (it uses the grid fit, which the piece
model doesn't change). pytest green.

**Silent errors, checked on ChessReD2K val (never test).** Before the
eval fix below, 99% of the new model's silent errors on val (1858/1877)
sat in 68 side-view photos scored as rotated 180°. The cause was the eval's
orientation label, not the model or the pipeline:
`chessred_prepare.py` labelled each photo 'w'/'b' by which board edge sat
lower in the photo, but in the app 'b' means "rotate the auto-turned board
180°", and a side view has no nearer side. Now labelled by where a8 lands in
the warp (`camera_side`, tested). On correctly oriented boards silent
errors were 0.07/board (val), so no per-square calibration (class cap or
tint threshold) was added: it could only add false tints. The residue: on
val the automatic side-view turn is wrong on 29/169 side views (test:
3/162) — a real but modest tie-break weakness, now the largest single
source of silent errors on angled photos.

**Numbers before the label fix** (same model, side views mislabelled):
ChessReD2K test + corners 85.84% (old model 57.80%), auto 60.78% (41.23%).
Don't compare those with the table above.

### Training

`prepare_mixed_dataset.py` builds 1441 ChessReD2K train + 388 samryan18
train images (val: 330 + 68), each produced exactly as the app produces
the model's input (EXIF-upright, `locate_board` with the annotated corners,
sideways warps turned upright), labelled with cell-sized boxes from the
annotated squares. Then, on CPU (~6 min/epoch, 4 cores):

```bash
python -I prepare_mixed_dataset.py --annotations /data/chessred/annotations.json \
    --images-root /data/chessred/extracted --out /data/mix
CVC_FT_DATA=/data/mix/dataset.yaml CVC_FT_MODEL=yolov8n.pt CVC_FT_IMGSZ=640 \
CVC_FT_EPOCHS=25 CVC_FT_PATIENCE=6 CVC_FT_RUN=mix640 \
CVC_FT_EXPORT=candidate-mix640.onnx python finetune_train.py
```

Hyperparameters: yolov8n.pt base, imgsz 640, 25 epochs (best epoch 24,
val mAP50 0.991), batch 16, lr0 0.002, optimizer auto, degrees 8,
translate 0.1, scale 0.5, fliplr 0.5, mosaic 0, seed 0. Export opset 12,
same IO as before (`images` [1,3,640,640] → `output0` [1,17,8400]).

**License:** ChessReD is CC BY-NC-SA 4.0, so these weights are
**non-commercial** (share-alike), on top of the AGPL-3.0 the Ultralytics
fine-tune already carries.

### Earlier changes (old model, samryan18 only)


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

*Measured with the previous model; the piece model is the same size and
input, so latency and concurrency results carry over.*

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

## Other boards and piece sets

ChessReD2K test (above) is the first labelled check on a different set:
99.1% per square, 4 in 5 boards perfect. On the two iStock photos of yet
other sets (`test_photos/`) the reads are still wrong (starting position
28/64 squares wrong) but every square is tinted, and the impossible-material
rule explains why: 0 silent errors. More sets in training is still the fix;
see `RESULTS_more_sets.md` for further sets.

## Is there a board at all? — `board_presence_eval.py`

508 real board photos scored a grid fit of 0.732–1.01; 94 ordinary non-chess
photos (OpenCV and scikit-image samples) 0.48–0.63. The cut-off 0.68 rejects
all 94 and wrongly rejects 0 of 508 boards. Calibration checkerboards and
empty checkerboards (hard negatives, 14/31 pass) read as empty boards.

## Honest limits

- **Two board styles measured.** samryan18 and ChessReD; other sets are
  flagged, not read.
- **Side views:** the automatic turn is wrong on ~1 in 6 ChessReD val side
  views (test: 3/162); those boards are read rotated and look confident.
- **Non-commercial weights** (ChessReD CC BY-NC-SA) plus AGPL.
- **CPU-bound.** One 4-core box serves one analysis at a time; more users
  means more queueing, not failures.

## Reproduce

```bash
git clone --depth 1 https://github.com/samryan18/chess-dataset /tmp/chess-dataset
CVC_BACKEND=model python bench/full_eval.py --json results.json      # samryan18 sets
# ChessReD: annotations.json + chessred2k.zip from data.4tu.nl, unpacked outside the repo
python -I bench/chessred_prepare.py --annotations A.json --images-root DIR --split test --out /data/chessred_test
CVC_BACKEND=model python bench/full_eval.py --photos /data/chessred_test [--use-corners]
python bench/square_dump.py --photos /data/chessred_val --use-corners --out val.json  # per-square dump
python bench/e2e_load.py --users 4 --photos 32 --rounds 3
```

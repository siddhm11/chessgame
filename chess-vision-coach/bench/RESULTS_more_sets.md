# More piece sets: data preparation and baseline (sets 3 and 4)

The current model (`yolov8n640-chess-finetuned.onnx`) has only been trained on
one piece set (samryan18). This file covers two more sets: where they come
from, how they were prepared, and how the **current, unchanged** model scores
on them. Nothing here trains or replaces the model. Set 2 (ChessReD) is
covered in `RESULTS.md`.

| | set 3 | set 4 |
|---|---|---|
| source | Hugging Face `surawut/chess-pieces-dataset-yolo` (Roboflow export, workspace chess-detection-8s15a, project chess-pieces-detection-1yb2h v4) | chesscog "transfer learning" sample (github.com/georg-wolflein/chesscog, Google Drive zip `1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W`) |
| license | **CC BY 4.0** (HF card and Roboflow README) | none stated for the sample; chesscog's code is MIT. Use for evaluation only, don't redistribute |
| pieces / board | plastic tournament Staunton pieces, green/white vinyl board | wooden Staunton pieces, black/maple board |
| view | top-down video frames, already warped edge-to-edge to 640×640 | 1200×900 phone photos from a player's seat (angled) |
| role | training data + held-out val/test | **held-out test only, never train** |
| prepared at | `/data/set3_prepared` | `/data/set4_prepared` |

**ChessReD license (set 2), checked against the 4TU API**
(`data.4tu.nl/v2/articles/99b5c721-280b-450b-b058-b2900b69a90f`): **CC BY-NC-SA
4.0** for both versions 1 and 2 (DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2).
It is non-commercial, so it can't go into a commercially deployed model.

**Network notes.** Hugging Face serves dataset files from `us.aws.cdn.hf.co`.
The proxy refused it (403 CONNECT) until that host was added to the
environment's allow-list. Google Drive's `doc-0s-64-docs.googleusercontent.com`
is blocked, but `drive.usercontent.google.com/download?id=…&confirm=t` works.

## Set 3: `bench/set3_prepare.py`

3,137 images:

- 2,499 frames from 5 videos. Each training frame comes as 3 Roboflow copies:
  the original plus 2 augmented ones (random horizontal/vertical flip,
  ±20% brightness, ±15% exposure). Roboflow's "valid" frames are single,
  un-augmented originals.
- 638 `ext_*` images. These are mostly close-ups of 2–5 squares, not whole
  boards, so they can't give a 64-square position. They are left out.

**Split by video, never by frame.** Neighbouring frames are near-duplicates.
Frames are grouped by the name before `_frame_` (video) and before
`_jpg.rf.<hash>` (one source frame's copies).

| split | videos | source frames | clean source frames | clean images written |
|---|---|---|---|---|
| train | Bonus-Long-Video-Label, 2_move_student | 678 | 136 | 352 |
| val | 6_Move_student | 84 | 14 | 28 |
| test | 8_Move_student, 2_Move_rotate_student | 199 | 36 | 96 |

All five videos show the same board, pieces and camera. "Held out by video"
means unseen frames and positions, not an unseen scene.

**Hands.** 2,023 of the 2,526 video images (80%) contain a `Hand` box: the
videos film moves being played. These are excluded from evaluation **and**
training. A hand hides pieces, and the pieces it holds have no square.
They are written to `images/rejected/` for inspection.

**Flips.** Each augmented copy is matched against the median of the video's
un-augmented frames under the 4 possible flips (none, H, V, H+V). The camera
is static, so lighting gradients make this unambiguous:

- 100% correct on synthetic flips of the originals; the best-vs-second
  score margin is always ≥ 0.24.
- Every source frame's triple contains exactly one unflipped copy.

Flipped copies are flipped back, so no mirrored board is written. Evaluation
uses only the unflipped copy, one per source frame.

**Orientation.** Each video is rotated as a whole so that a8 is light and
White is at the bottom. This is decided by voting over its clean frames;
every frame voted the same way, and all five videos turned out to be
already canonical.

**Per-square truth from boxes.** Pieces lean away from the camera, so a box
centre can sit on the next square. The base is the disc at the box end
opposite the lean. The lean direction is constant per video (the camera sat
beyond one edge of the board) and is measured from pawn boxes:

- Student videos: pawn box height grows from 0.61 to 0.88 squares, top to
  bottom, so pieces lean down.
- Bonus video: pawn box width grows toward the left, so pieces lean left.

A frame is **clean** only if it has:

- no Hand box and no two pieces on one square;
- no base point within 0.10 square of a square edge;
- exactly one king per side, at most 8 pawns per side, at most 16 pieces per
  side, and no pawn on rank 1/8;
- a light a8.

**Result: 186 of 186 non-hand source frames derive cleanly.** Every
collision and every edge case was in a hand frame, where the hand is holding
a piece in mid-air.

**Checked by eye.** Derived positions were drawn onto 8 frames (val, test
including the rotate video, and train): boxes, base points, grid and the
derived letter on each square. Every piece's letter sat on the square its
base stands on, a8 was light and the white king was on e1. Frames from the
Bonus video read as a real opening (1.f4 d5 2.Nf3 Bf5 3.Nc3).

**Output format** is our inference format: a 640×640 crop with no margin, in
canonical orientation.

- `labels/`: our class order (`1 white_king … 12 black_pawn`), full-cell
  boxes, the same style as `prepare_finetune_dataset.py`.
- `labels_boxes/`: the original tight boxes, re-mapped.
- `dataset.yaml`: class names match the ONNX metadata.
- `positions.jsonl`: the FEN board part, split, flip and issues for each image.

## Set 4: `bench/set4_prepare.py`

29 photos with JSON labels `{"white_turn": bool, "fen": placement}`: 27
`test/` photos (one game) and 2 `train/` photos (the starting position from
each side; chesscog fine-tunes on these).

- `white_turn` is both the side to move and the camera side. Checked by eye
  on all 29 photos: the mover's pieces are always nearest the camera.
- The 27 test photos are converted to `<placement>__<w|b>__test-NN.png`, plus
  `corners.json` and `side_to_move.json`, i.e. the `full_eval.py --photos`
  format.

**Corners: the labels have none, so they were derived.**

1. OpenCV's `findChessboardCornersSBWithMeta` (with `CALIB_CB_LARGER`) finds
   a sub-grid of the board's lattice.
2. A homography from that sub-grid lets the script test every possible 8×8
   window. It picks the one with strong checker contrast inside and the dark
   frame just outside.

This gave a clean fit on 28 of 29 photos (score 1.72–1.88; the failure
scored 1.09). The corners of `test-07` were marked by hand from zoomed crops
(`/data/set4_meta/corners_override.json`). All 29 grid overlays were checked
by eye (`/data/set4_prepared/corner_check/`).

These corners are not independent ground truth. Treat full_eval's "board
finder within ¼ square" number as agreement with them.

## Baseline: current model, unchanged

Scored with `bench/score_more_sets.py`, which adds best-move agreement:
Stockfish depth 12, 1 thread, no castling rights. An invalid predicted
position counts as disagree; a board whose true position is invalid is left
out. Set 4 was also scored with `bench/full_eval.py --photos [--use-corners]`,
which gives the same numbers.

**Silent** means wrong and shown with confidence ≥ 0.80 (not tinted in the app).

| set / mode | boards | per-square exact | perfect | wrong squares | silent | silent/board | best move agrees | predicted FEN invalid |
|---|---|---|---|---|---|---|---|---|
| set 3 test, crop (finder bypassed) | 36 | 84.07% | 0 | 367 | 154 | 4.28 | 0/35 | 32 |
| set 3 test, full pipeline (1-square grey border) | 36 | 87.37% | 0 | 291 | 118 | 3.28 | 2/35 * | 34 |
| set 3 val, crop | 14 | 84.49% | 0 | 139 | 67 | 4.79 | 0/14 | 14 |
| set 3 val, full pipeline | 14 | 88.84% | 0 | 100 | 31 | 2.21 | 0/14 | 13 |
| set 4 test, full pipeline | 27 | 45.25% | 0 | 690 † | 0 | 0 | 0/27 | 23 |
| set 4 test, pipeline + true corners | 27 | 52.72% | 0 | 817 | 12 | 0.44 | 0/27 | 27 |

\* Both agreements are luck: those boards had 7 and 8 wrong squares.
† Plus 4 fallbacks (no board found; returns the starting position), which are
not counted as wrong squares here. Counted as 64 wrong each, the total is 946.
On set 3, one test position is invalid with White to move, so 35 boards are scored.

What the numbers say:

- **Set 3 (plastic tournament set, top-down).** Locations are fine; piece
  *types* are not.
  - Top confusions: white pawn → bishop (91 on test, crop), black pawn →
    bishop (88), queen/king/knight → bishop, queen → rook.
  - Every board has 4–14 wrong squares, and about 3–5 per board are
    confident (silent).
  - The sanity layer cannot help: the reads are plausible-looking but wrong.
  - Full pipeline scores higher than crop-only. On the padded frames the
    finder lands within 10–14 px of the true edge (a square is 80 px) with
    grid fit 0.79–0.83, so the doubt gate tints some boards. That lowers
    silent errors, and the slightly different crop helps a little.
- **Set 4 (wooden set, angled).** Nothing usable.
  - About half the squares are wrong.
  - Pawns are read as knights or queens or missed, and phantom knights appear.
  - The board finder misses 4 boards outright and is within ¼ square on 21/27.
  - The app's unfamiliar-set warning does fire: 0 silent errors without
    corners. With corners given, 12 squares are silent.

Both sets confirm the limit stated in `RESULTS.md`: the model reads one piece
set. Set 3 has clean, correctly oriented training crops ready (352 images
from 136 frames, plus val/test held out by video). Set 4 stays a
never-trained test of generalization to an unseen set and an angled camera.

## Reproduce

```bash
# from chess-vision-coach/, data outside the repo
.venv/bin/python -I bench/set3_prepare.py --raw /data/set3/raw --out /data/set3_prepared
.venv/bin/python -I bench/set4_prepare.py --raw /data/set4/raw --out /data/set4_prepared \
    --corners-override /data/set4_meta/corners_override.json
.venv/bin/python -I bench/score_more_sets.py --set3 /data/set3_prepared --mode crop
.venv/bin/python -I bench/score_more_sets.py --set3 /data/set3_prepared --mode pipeline
.venv/bin/python -I bench/score_more_sets.py --photos /data/set4_prepared/photos [--use-corners]
CVC_BACKEND=model .venv/bin/python -I bench/full_eval.py --photos /data/set4_prepared/photos [--use-corners]
```

Unit tests for the converters and scorer, all synthetic:
`tests/test_more_sets_prepare.py`.

## Board-corner detector (learned candidate for the board finder)

**Why.** On ChessReD2K test (angled phone photos from a player's seat) the
classical board finder said "no board" on 91/306 photos and landed within ¼
square of the true corners on only 213/306. On those photos, board *finding*
was the bottleneck, not piece reading.

**What.** `chess_vision/models/board-corners.onnx` is YOLOv8n, one class,
"corner": a box centred on each corner of the playing area, imgsz 640, 12.2 MB
fp32. `chess_vision/board_corners.py` runs it with onnxruntime, or with OpenCV
DNN when onnxruntime is missing (outputs agree to 3e-4), and returns the 4
best corners if they form a plausible convex quad.

`ClassicalBackend.locate_board` adds that quad as **one more candidate**,
after the contour quads and before the sliding scan. It is snapped by
`_refine_quad` and judged by the same sharp grid fit, so it can only add a
better-fitting crop, never bypass the judge. Without the file (or with
`CVC_CORNER_MODEL=0`) behaviour is unchanged.

Corners are detected **unordered**. A 4-keypoint pose model needs a fixed
order (top-left first, or a1 first), and that order flips abruptly as the
board rotates in the photo: the same physical corner swaps labels. Detecting
"a corner" avoids that; the 4 points are ordered geometrically afterwards
(`_order_points`).

**Data.** `bench/board_corners_prepare.py` uses the ChessReD2K train split
(1,442 photos) and its val split (330), with the annotated corners, photos
resized to 640. The box side is one board square. The ChessReD2K test split,
set 4 and samryan18 were never used for training. CPU training: 18 epochs at
4 min each; val mAP50 0.995, mAP50-95 0.92.

**Gate** (current piece model throughout; `full_eval.py`, `board_presence_eval.py`):

| check | before (no detector) | after |
|---|---|---|
| ChessReD2K test (306), no corners: board within ¼ square | 213/306 | **305/306** |
| ChessReD2K test: fallbacks ("no board") | 91 | **0** |
| ChessReD2K test: per-square exact (fallback = 64 wrong) | 41.07% | **57.97%** |
| ChessReD2K test: silent wrong squares | 979 (on 215 read boards, 4.6/board) | 1,594 (on 306 boards, 5.2/board) |
| ChessReD2K test: best move same | 0/300 | 0/300 |
| samryan18 val (68) | 99.52%, 60 perfect, 12 silent, best move 57/63 | identical |
| samryan18 rejected (44) | 99.33%, 32 perfect, 3 silent, best move 35/42 | identical |
| val + EXIF / val + dark | 99.72% / 99.77% | identical |
| set 4 (27), no corners | 45.25%, 4 fallbacks, ¼-square 21/27 | identical |
| board presence at the app's cut-off (fit < 0.68) | 94/94 ordinary photos rejected, 0/508 boards wrongly rejected | identical |
| pytest | 202 + 4 new | 206 passed |

The detector fixes board finding on angled photos and changes nothing on the
others. The remaining ChessReD errors are the piece model's: it doesn't know
that piece set. That is why silent errors rise: 91 more boards are now read
instead of falling back. The combined piece retrain is meant to fix that.
Negatives: OpenCV samples/data + scikit-image data (125 images, 31 of them
hard negatives such as calibration checkerboards; those behave exactly as before).

## Combined piece retrain (sets 1-3): trained, gated, NOT swapped

**Data:** `/data/mix`, never committed.
- Train: ChessReD2K train (1,441) + samryan18 train (388) + set 3 train (352) repeated 3×, so 2,885 images.
- Val: ChessReD2K val (330) + samryan18 val (68) + set 3 val (28), so 426 images.
- Never trained on: ChessReD2K test, set 3 test, set 4, samryan18 val/rejected.
- ChessReD labels come from the corrected `chessred_prepare.py` (56efed5).

**Training:**
- yolov8n, 640 px, 25 epochs, patience 6, CPU.
- Same hyperparameters as `finetune_train.py`: lr0 0.002, lrf 0.01, warmup 2, degrees 8, fliplr 0.5, no mosaic, hsv 0.015/0.4/0.4.
- All 25 epochs ran. Val mAP50 was 0.988 and mAP50-95 0.976.
- ONNX export (opset 12, simplified) has the same IO as the shipped model: `images [1,3,640,640]` → `output0 [1,17,8400]`, same `names` map, loads in onnxruntime and in cv2.dnn.

**Gate:**
- Every number below comes from the same code at c75eea8, with the board-corner detector on.
- ChessReD rows use the corrected `/data/chessred_test` labels.
- "Silent" means wrong with conf ≥ 0.80. "Tints" means false tints: right squares with conf < 0.80. Best move uses Stockfish depth 12.

| set | original (008fc66) | branch ca6c364 | combined |
|---|---|---|---|
| samryan18 val (68) | 99.52%, 60 perfect, 12 silent, bm 57/63 | 99.66%, 65 perfect, 13 silent, bm 61/63 | 99.66%, 65 perfect, 12 silent, bm 62/63 |
| samryan18 rejected (44) | 99.33%, 32 perfect, 3 silent, bm 35/42 | 99.86%, 41 perfect, 3 silent, bm 42/42 | 99.82%, 40 perfect, 2 silent, bm 41/42 |
| ChessReD2K test (306), auto | 62.92%, 0 perfect, 3.51 silent/b, 15.6 tints/b, bm 1/300 | **99.14%, 251 perfect, 0.36 silent/b**, 3.4 tints/b, bm 254/300 | 98.71%, 237 perfect, 0.58 silent/b, 4.3 tints/b, bm 249/300 |
| ChessReD2K test, true corners | 62.84%, 0 perfect, 3.73 silent/b | **99.31%, 252 perfect, 0.12 silent/b**, bm 257/300 | 98.40%, 235 perfect, 0.60 silent/b, bm 251/300 |
| set 3 test (36), crop | 84.07%, 0 perfect, 4.28 silent/b, 7.6 tints/b, bm 0/35 | 93.49%, 0 perfect, 1.39 silent/b, 7.7 tints/b, bm 7/35 | **94.75%, 8 perfect**, 1.94 silent/b, **1.3 tints/b**, bm 9/35 |
| set 3 test, pipeline | 87.37%, 0 perfect, 3.28 silent/b, 7.4 tints/b, bm 2/35 | 93.32%, 0 perfect, 1.72 silent/b, 6.4 tints/b, bm 9/35 | **94.84%, 8 perfect, 1.58 silent/b**, 6.4 tints/b, bm 10/35 |
| set 4 (27), auto | 45.25%, 0 silent, 4 fallbacks | **59.43%, 0 silent**, 38.0 tints/b, 4 fallbacks | 57.70%, 0.56 silent/b, 34.0 tints/b, 4 fallbacks |
| set 4, true corners | 52.72%, 0.44 silent/b | **71.64%, 0 silent**, 45.9 tints/b | 69.10%, 0 silent, 44.2 tints/b |

Best move on set 4 is 0/27 for every model.

**Decision: keep ca6c364's model.** The rule was "hold the other sets within noise AND clearly help set 3 and/or set 4". The combined model fails the first half:
- **ChessReD2K test drops beyond noise:** 14 fewer perfect boards, silent errors per board up 0.36 → 0.58, and with true corners 0.12 → 0.60, five times as many.
- **samryan18 is flat.**
- **Set 3 gains** 8 perfect boards against 0, +1.3–1.5 pts exact, and in crop mode far fewer false tints. But in crop mode silent errors per board rise 1.39 → 1.94.
- **Set 4 gets slightly worse**, and it now has silent errors where ca6c364 had none.

The set-3 gain is narrow: the test set is 36 frames from 2 videos of one board. It does not pay for the ChessReD regression or for silent errors on set 4.

If set 3 matters later, two options could keep ca6c364's ChessReD numbers:
- fine-tune ca6c364 briefly on set 3 + a ChessReD replay, at a low learning rate;
- oversample set 3 less than 3×.

Nothing shipped changed, so pytest, `board_presence_eval.py` and the browser e2e were not rerun.

**Set 4 orientation check:** labels as prepared score 52.72% with the original model. With every white/black piece swapped they score 37.91%. So the white-to-move = camera-side reading is right, and set 4 needs no relabelling.

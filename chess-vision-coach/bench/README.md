# Vision-backend benchmark

Objective comparison of every vision backend on a real labeled dataset.
No more "this model feels better."

## What it does

1. Clones [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset)
   to `/tmp/chess-dataset` on first run (~1.3 GB; 500 real top-down photos
   of physical Staunton pieces on a green-and-white tournament board,
   labeled with FEN in the filename).
2. For each backend (classical + every `.onnx` file dropped into
   `chess_vision/models/`) and each sampled photo, predicts the position
   and compares per-square to the ground-truth FEN.
3. Emits a table with four metrics + per-image latency:

   | metric | what it measures |
   |---|---|
   | **exact**  | per-square exact match. The headline "is the parse right" number. |
   | **empty**  | per-square: did we agree on whether the square is empty? |
   | **recall** | of squares the truth says have a piece, how often did we predict *a* piece (any type)? |
   | **type\|p** | given both predicted-and-truth have a piece, how often does the type match? |

## Run it

```bash
# Default: 20 photos
python bench/benchmark.py

# Bump for a more stable signal (slower)
python bench/benchmark.py --n 100

# Point at your own labeled dir (files named like
# `r1bqk2r-pppp1ppp-2n5-2b1p3-...8-RNBQK1NR.JPG`)
python bench/benchmark.py --source path/to/photos
```

## Adding a new model to compare

Drop the `.onnx` file into `chess_vision/models/` and rerun. The
benchmark auto-discovers it.

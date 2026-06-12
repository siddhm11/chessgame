---
title: Chess Vision Coach
emoji: ♟️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: agpl-3.0
---

# Chess Vision Coach

Upload a photo of a physical chessboard. The app extracts the position to a
FEN, lets you fix any misclassified pieces by clicking squares, and uses
Stockfish to suggest the best move with an evaluation.

```
photo  →  board detection  →  FEN  →  your corrections  →  Stockfish
```

## What this Space does

- **Vision** — a YOLOv8n model fine-tuned on top-down tournament photos
  (99.79% per-square accuracy on a held-out 68-photo validation set).
- **Sanity layer** — chess rules (no pawns on the back rank, exactly one
  king per colour, etc.) repair impossible reads from the model.
- **Click-to-correct** — any square the vision pass was unsure about is
  tinted amber/red; click to swap the piece.
- **Stockfish** — Fast / Normal / Strong strength selector, top-3 candidate
  moves with full principal-variation lines.

## Honest limitations

- The vision model was trained on **top-down photos**. Steeply angled
  shots, heavy shadows, or unusual piece sets will read worse — but the
  tint + click-fix flow recovers gracefully.
- The Space runs on **CPU only** (free tier). Stockfish at "Strong"
  (depth 22) takes 2–3 seconds.
- Sessions live in process memory — your previous uploads are kept while
  the Space is warm, but reset on each cold start (Spaces sleep after
  ~48 h of inactivity).

## License

AGPL-3.0 — the model weights are Ultralytics YOLO derivatives. Fine for
personal and learning use; a commercial deployment needs an Ultralytics
commercial license or a non-YOLO model.

The source code lives at <https://github.com/siddhm11/chessgame> on the
`claude/chess-vision-coach-DwzKV` branch.

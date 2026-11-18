# Chess Vision Coach in the browser

The whole app as a static website: no server, no hosting bill, and photos
never leave the user's device. Free to host on GitHub Pages (or any static
host).

## How it works

| piece | runs as |
|---|---|
| board finder, sanity checks, position logic | the server's own `chess_vision` Python package, unchanged, in [Pyodide](https://pyodide.org) (Python on WebAssembly) with numpy, OpenCV and Pillow |
| piece model (`yolov8n640-chess-finetuned.onnx`) | **onnxruntime-web** where the browser supports JSPI (Chrome/Edge 137+): same engine as the server. Elsewhere OpenCV's DNN module inside Python |
| engine | Stockfish 19 lite, single-threaded WASM |
| UI | `src/index.html` + `src/app.js`, same layout and features as the server app |

Both model runtimes give the same reads as the server: OpenCV DNN matched
onnxruntime on all 112 held-out photos (identical per-square results), and
onnxruntime-web is onnxruntime.

Measured in headless Chromium on a 4-core server CPU (`tools/e2e.mjs`):

| | fast path (onnxruntime-web) | compatibility (OpenCV) |
|---|---|---|
| first load | ~5 s + download (~45 MB, then cached) | same |
| read a photo | **~1.8 s** (5 s worst case) | ~6 s (9 s worst case) |
| best move (normal) | ~1.3 s | same |

Phones are slower; expect roughly 2x.

## Build and test

```bash
cd chess-vision-coach/web
npm ci
node tools/fetch_runtime.mjs      # Pyodide + numpy/OpenCV/Pillow wheels, sha256-checked
node tools/build.mjs              # -> dist/  (static site, ~56 MB)
node tools/serve.mjs 8080         # http://localhost:8080
node tools/e2e.mjs photo.JPG ...  # headless Chromium end-to-end test
node tools/e2e.mjs --no-jspi photo.JPG   # same, on the compatibility path
```

`?runtime=opencv` in the page URL forces the compatibility path.

## Deploy (free): GitHub Pages

`.github/workflows/pages.yml` builds and tests the site on every push and
publishes it from `main`. One-time setup in the GitHub repo:

1. **Settings -> Pages -> Build and deployment -> Source: GitHub Actions.**
2. Merge this branch into `main` (or run the workflow on `main`).

The site appears at `https://<user>.github.io/<repo>/`.

Any other static host works too (Netlify, Cloudflare Pages, an S3 bucket):
upload `dist/`.

## Licenses

Pyodide (MPL-2.0), OpenCV (Apache-2.0), onnxruntime-web (MIT), Stockfish
(GPL-3.0, source: github.com/official-stockfish/Stockfish), python-chess
(GPL-3.0). The model weights are Ultralytics YOLO derivatives (AGPL-3.0).

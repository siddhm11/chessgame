// Assemble the static site in web/dist/ (run fetch_runtime.mjs first).
//
//   dist/index.html, app.js, vision-worker.js, app.css, web.css
//   dist/py/        chess_vision/*.py + cvc_web.py  (listed in manifest.json)
//   dist/models/    the ONNX model the server uses
//   dist/pyodide/   Python runtime + verified wheels (fetch_runtime.mjs)
//   dist/ort/       onnxruntime-web (CPU/WASM build)
//   dist/engine/    Stockfish (lite, single-threaded WASM)
//
// The python-chess wheel is built from the version pinned in
// requirements.txt (pure Python): `pip wheel` must be on PATH as $PYTHON.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.join(WEB, "..");
const DIST = path.join(WEB, "dist");
const NM = path.join(WEB, "node_modules");
const MODEL = process.env.CVC_MODEL || "yolov8n640-chess-finetuned.onnx";

const copy = (src, dest) => {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
};

// Page + styles (app.css is the server's stylesheet, shared).
for (const f of ["index.html", "app.js", "vision-worker.js", "web.css"]) {
  copy(path.join(WEB, "src", f), path.join(DIST, f));
}
copy(path.join(ROOT, "static", "app.css"), path.join(DIST, "app.css"));
fs.writeFileSync(path.join(DIST, ".nojekyll"), "");   // GitHub Pages: serve files as-is

// Python sources: the server's chess_vision package, unchanged, plus glue.
fs.rmSync(path.join(DIST, "py"), { recursive: true, force: true });
const python = [];
for (const f of fs.readdirSync(path.join(ROOT, "chess_vision")).sort()) {
  if (!f.endsWith(".py") || f === "cli.py") continue;
  copy(path.join(ROOT, "chess_vision", f), path.join(DIST, "py", "chess_vision", f));
  python.push(`chess_vision/${f}`);
}
copy(path.join(WEB, "src", "py", "cvc_web.py"), path.join(DIST, "py", "cvc_web.py"));
python.push("cvc_web.py");

// Model.
fs.rmSync(path.join(DIST, "models"), { recursive: true, force: true });
copy(path.join(ROOT, "chess_vision", "models", MODEL), path.join(DIST, "models", MODEL));

// python-chess wheel, pinned by the server's requirements.txt.
const pin = fs.readFileSync(path.join(ROOT, "requirements.txt"), "utf8").match(/^chess==([\d.]+)/m)[1];
const pyDir = path.join(DIST, "pyodide");
let chessWheel = fs.readdirSync(pyDir).find((f) => f === `chess-${pin}-py3-none-any.whl`);
if (!chessWheel) {
  const py = process.env.PYTHON || "python3";
  execFileSync(py, ["-m", "pip", "wheel", "--no-deps", "-q", `chess==${pin}`, "-w", pyDir], { stdio: "inherit" });
  chessWheel = `chess-${pin}-py3-none-any.whl`;
}

// onnxruntime-web: the plain WASM build (CPU), loader + binary.
for (const f of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copy(path.join(NM, "onnxruntime-web", "dist", f), path.join(DIST, "ort", f));
}

// Stockfish: lite single-threaded build (no cross-origin isolation needed).
for (const f of ["stockfish-19-lite-single.js", "stockfish-19-lite-single.wasm"]) {
  copy(path.join(NM, "stockfish", "bin", f), path.join(DIST, "engine", f));
}

fs.writeFileSync(path.join(DIST, "manifest.json"), JSON.stringify({ model: MODEL, chessWheel, python }, null, 2));

let total = 0;
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).forEach((e) => {
  const p = path.join(d, e.name);
  if (e.isDirectory()) walk(p);
  else if (!e.name.endsWith(".map") && e.name !== "package.json") total += fs.statSync(p).size;
});
walk(DIST);
console.log(`site built in ${path.relative(process.cwd(), DIST)} (${(total / 1e6).toFixed(1)} MB)`);

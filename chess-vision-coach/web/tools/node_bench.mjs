// Run the in-browser pipeline under Node -- the same Pyodide runtime and
// cvc_web.py glue the page uses -- on a few photos; print FEN + time each.
//
//   node tools/node_bench.mjs [--ort] photo1.jpg [photo2.jpg ...]
//
// --ort runs the model in onnxruntime-web through JSPI, as Chrome does
// (needs: node --experimental-wasm-jspi). Default: OpenCV DNN fallback.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUNTIME = path.join(WEB, "dist", "pyodide") + path.sep;
const PKG = path.join(WEB, "..", "chess_vision");
const MODEL = "yolov8n640-chess-finetuned.onnx";

const args = process.argv.slice(2);
const useOrt = args[0] === "--ort";
const photos = useOrt ? args.slice(1) : args;

let t = Date.now();
const py = await loadPyodide({ indexURL: RUNTIME });
const chessWheel = fs.readdirSync(RUNTIME).find((f) => f.startsWith("chess-"));
await py.loadPackage(["numpy", "opencv-python", "pillow", RUNTIME + chessWheel], { messageCallback: () => {} });
console.log(`runtime + packages: ${Date.now() - t} ms`);

py.FS.mkdirTree("/app/chess_vision/models");
for (const f of fs.readdirSync(PKG)) {
  if (f.endsWith(".py")) py.FS.writeFile(`/app/chess_vision/${f}`, fs.readFileSync(path.join(PKG, f)));
}
py.FS.writeFile("/app/cvc_web.py", fs.readFileSync(path.join(WEB, "src", "py", "cvc_web.py")));
const modelBytes = fs.readFileSync(path.join(PKG, "models", MODEL));
py.FS.writeFile(`/app/chess_vision/models/${MODEL}`, modelBytes);
py.runPython('import sys; sys.path.insert(0, "/app")');
const web = py.pyimport("cvc_web");

if (useOrt) {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const jsRun = async (data) => {
    const input = new ort.Tensor("float32", new Float32Array(data), [1, 3, 640, 640]);
    const out = await session.run({ [session.inputNames[0]]: input });
    return out[session.outputNames[0]].data;
  };
  const probe = await jsRun(new Float32Array(3 * 640 * 640));
  console.log("runtime:", web.init(`/app/chess_vision/models/${MODEL}`, jsRun, py.toPy([1, 17, probe.length / 17])));
} else {
  console.log("runtime:", web.init(`/app/chess_vision/models/${MODEL}`));
}

for (const [i, p] of photos.entries()) {
  py.FS.writeFile("/tmp/upload", fs.readFileSync(p));
  t = Date.now();
  web.load_image("/tmp/upload");
  // callPromising (stack switching) only when JSPI is in play; a plain call otherwise.
  const call = useOrt ? (...a) => web.analyze.callPromising(...a) : async (...a) => web.analyze(...a);
  const res = JSON.parse(await call("white", "white", null));
  console.log(`${((Date.now() - t) / 1000).toFixed(2).padStart(6)}s  ${res.fen.split(" ")[0]}  ${path.basename(p)}`);
}

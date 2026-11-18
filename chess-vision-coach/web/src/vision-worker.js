// Vision worker: Python (Pyodide) + the chess_vision package + the model.
// Runs off the main thread so the page stays responsive while a photo is
// read. Protocol: {id, cmd, args} in; {id, ok, result | error} out, plus
// {type: "progress", text, pct} while booting.

import { loadPyodide } from "./pyodide/pyodide.mjs";

const BASE = new URL("./", import.meta.url).href;
const FORCED = new URL(import.meta.url).searchParams.get("runtime");
const JSPI = typeof WebAssembly.Suspending === "function" && FORCED !== "opencv";

let web = null;          // the cvc_web Python module
let runtime = "loading";
let booting = null;

const progress = (text, pct) => self.postMessage({ type: "progress", text, pct });

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function startOrt(modelBytes) {
  const ort = await import("./ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = BASE + "ort/";
  ort.env.logLevel = "error";   // CPU-detection warnings are noise here
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const [inName] = session.inputNames;
  const [outName] = session.outputNames;
  const jsRun = async (data) => {
    const input = new ort.Tensor("float32", new Float32Array(data), [1, 3, 640, 640]);
    const out = await session.run({ [inName]: input });
    return out[outName].data;
  };
  const probe = await jsRun(new Float32Array(3 * 640 * 640));
  return { jsRun, rows: probe.length / 8400 };
}

async function boot() {
  progress("Loading Python…", 5);
  const manifest = await (await fetch(BASE + "manifest.json")).json();
  const py = await loadPyodide({ indexURL: BASE + "pyodide/" });
  progress("Loading image libraries…", 30);
  await py.loadPackage(
    ["numpy", "opencv-python", "pillow", BASE + "pyodide/" + manifest.chessWheel],
    { messageCallback: () => {} },
  );
  progress("Loading the chess reader…", 60);
  py.FS.mkdirTree("/app/chess_vision/models");
  await Promise.all(manifest.python.map(async (rel) => {
    py.FS.writeFile("/app/" + rel, await fetchBytes(BASE + "py/" + rel));
  }));
  const modelPath = "/app/chess_vision/models/" + manifest.model;
  const modelBytes = await fetchBytes(BASE + "models/" + manifest.model);
  py.FS.writeFile(modelPath, modelBytes);
  py.runPython('import sys; sys.path.insert(0, "/app")');
  web = py.pyimport("cvc_web");

  progress("Starting the piece model…", 85);
  if (JSPI) {
    try {
      const { jsRun, rows } = await startOrt(modelBytes);
      runtime = web.init(modelPath, jsRun, py.toPy([1, rows, 8400]));
    } catch (err) {
      console.warn("onnxruntime-web unavailable, using OpenCV:", err);
    }
  }
  if (runtime === "loading") runtime = web.init(modelPath);
  self.pyodide = py;
  progress("Ready", 100);
  return { runtime, jspi: JSPI };
}

const handlers = {
  async boot() {
    booting ??= boot();
    return booting;
  },
  async analyze({ bytes, orientation, sideToMove, corners }) {
    if (bytes) {   // a new photo; without bytes, re-read the current one
      self.pyodide.FS.writeFile("/tmp/upload", new Uint8Array(bytes));
      const loaded = JSON.parse(web.load_image("/tmp/upload"));
      self.pyodide.FS.unlink("/tmp/upload");
      if (loaded.error) return loaded;
    }
    // Stack switching (JSPI) only when the model runs in onnxruntime-web.
    const args = [orientation, sideToMove, corners ? JSON.stringify(corners) : null];
    const json = runtime === "onnxruntime-web"
      ? await web.analyze.callPromising(...args)
      : web.analyze(...args);
    return JSON.parse(json);
  },
  async photo() {
    // The upright, size-capped pixels the pipeline analysed: corner
    // coordinates are relative to exactly this image.
    const jpeg = web.display_jpeg();
    const bytes = jpeg.toJs();
    jpeg.destroy();
    return bytes;
  },
  view({ fen, psc, arrow }) {
    return JSON.parse(web.view(fen, psc ? JSON.stringify(psc) : null, false, arrow || null));
  },
  correct({ fen, square, piece }) { return web.correct(fen, square, piece); },
  sideToMove({ fen, side }) { return web.set_side_to_move(fen, side); },
  play({ fen, uci }) { return web.play(fen, uci) ?? null; },
  pvToSan({ fen, moves }) { return web.pv_to_san(fen, self.pyodide.toPy(moves)); },
  gameOver({ fen }) { return web.game_over(fen) ?? null; },
};

self.onmessage = async ({ data }) => {
  const { id, cmd, args } = data;
  try {
    const result = await handlers[cmd](args || {});
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};

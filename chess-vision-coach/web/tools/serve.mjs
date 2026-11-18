// Minimal static server for web/dist (local testing only).
//
//   node tools/serve.mjs [port] [--isolated]
//
// --isolated sends COOP/COEP headers so the page is cross-origin isolated
// (lets onnxruntime-web use several threads). Without it the page behaves
// like plain static hosting such as GitHub Pages.

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIST = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist");
const args = process.argv.slice(2);
const port = Number(args.find((a) => /^\d+$/.test(a)) || 8080);
const isolated = args.includes("--isolated");
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
  ".whl": "application/zip", ".zip": "application/zip", ".onnx": "application/octet-stream",
  ".svg": "image/svg+xml",
};

http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  let file = path.normalize(path.join(DIST, decodeURIComponent(url.pathname)));
  if (!file.startsWith(DIST)) { res.writeHead(403); return res.end(); }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!fs.existsSync(file)) { res.writeHead(404); return res.end("not found"); }
  const headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" };
  if (isolated) {
    headers["Cross-Origin-Opener-Policy"] = "same-origin";
    headers["Cross-Origin-Embedder-Policy"] = "require-corp";
  }
  res.writeHead(200, headers);
  fs.createReadStream(file).pipe(res);
}).listen(port, () => console.log(`serving ${DIST} on http://localhost:${port}${isolated ? " (isolated)" : ""}`));

// Assemble the Python-in-the-browser runtime into web/dist/pyodide/.
//
// Pyodide's core (interpreter + stdlib) comes from the npm package; the three
// compiled wheels the pipeline needs (numpy, opencv-python, pillow) come from
// Pyodide's CDN, or from a local directory given in PYODIDE_CACHE (e.g. an
// extracted official release). Every wheel is checked against the sha256 in
// pyodide-lock.json before it is used.
//
//   node tools/fetch_runtime.mjs

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CORE = path.join(WEB, "node_modules", "pyodide");
const OUT = path.join(WEB, "dist", "pyodide");
const VERSION = JSON.parse(fs.readFileSync(path.join(CORE, "package.json"))).version;
const CDN = `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/`;
export const PACKAGES = ["numpy", "opencv-python", "pillow"];
const CORE_FILES = [
  "pyodide.mjs", "pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
  "python_stdlib.zip", "pyodide-lock.json",
];

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

async function wheel(fileName) {
  const cache = process.env.PYODIDE_CACHE;
  if (cache && fs.existsSync(path.join(cache, fileName))) {
    return fs.readFileSync(path.join(cache, fileName));
  }
  const res = await fetch(CDN + fileName);
  if (!res.ok) throw new Error(`${CDN + fileName}: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

fs.mkdirSync(OUT, { recursive: true });
for (const f of CORE_FILES) fs.copyFileSync(path.join(CORE, f), path.join(OUT, f));
// pyodide.asm.js is CommonJS-style; web/package.json says "module". Only
// matters to Node (tools/node_bench.mjs); browsers ignore package.json.
fs.writeFileSync(path.join(OUT, "package.json"), '{"type": "commonjs"}\n');

const lock = JSON.parse(fs.readFileSync(path.join(OUT, "pyodide-lock.json")));
for (const name of PACKAGES) {
  const pkg = lock.packages[name];
  const dest = path.join(OUT, pkg.file_name);
  if (fs.existsSync(dest) && sha256(fs.readFileSync(dest)) === pkg.sha256) continue;
  const buf = await wheel(pkg.file_name);
  if (sha256(buf) !== pkg.sha256) throw new Error(`checksum mismatch: ${pkg.file_name}`);
  fs.writeFileSync(dest, buf);
  console.log(`  ${pkg.file_name} (${(buf.length / 1e6).toFixed(1)} MB) verified`);
}
console.log(`pyodide ${VERSION} runtime ready in ${path.relative(process.cwd(), OUT)}`);

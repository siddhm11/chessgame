// End-to-end test of the built site in headless Chromium.
//
//   node tools/e2e.mjs [--isolated] [--no-jspi] photo1.JPG [...]
//
// Serves web/dist, opens the page, uploads each photo, and checks the FEN
// against the file name (samryan18 naming: placement with '-' for '/').
// Then asks for the best move on the first photo, plays it, corrects a
// square, and re-reads with dragged corners. Exits non-zero on failure.
// --no-jspi loads ?runtime=opencv: the OpenCV fallback that browsers
// without JSPI get.

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const isolated = args.includes("--isolated");
const noJspi = args.includes("--no-jspi");
const photos = args.filter((a) => !a.startsWith("--"));
const port = 8000 + Math.floor(Math.random() * 900);

const server = spawn(process.execPath, [path.join(WEB, "tools", "serve.mjs"), String(port), ...(isolated ? ["--isolated"] : [])]);
await new Promise((r) => server.stdout.once("data", r));

let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failures++; };

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  let t = Date.now();
  await page.goto(`http://localhost:${port}/${noJspi ? "?runtime=opencv" : ""}`);
  await page.waitForSelector("#analyze-btn:not([disabled])", { timeout: 180000 });
  const runtime = await page.textContent("#runtime-note");
  console.log(`boot ${((Date.now() - t) / 1000).toFixed(1)} s -- ${runtime}`);
  check(noJspi ? /Compatibility/.test(runtime) : /Fast/.test(runtime), `runtime as expected (${noJspi ? "OpenCV fallback" : "onnxruntime-web"})`);

  let exact = 0;
  for (const p of photos) {
    if (!(await page.isVisible("#upload-view"))) await page.click("#new-photo");
    await page.setInputFiles("#photo-input", p);
    t = Date.now();
    await page.click("#analyze-btn");
    await page.waitForSelector("#analyze-view:not([hidden])", { timeout: 120000 });
    await page.waitForFunction(() => document.querySelector("#fen-text").textContent.length > 0);
    const fen = (await page.textContent("#fen-text")).trim();
    const truth = path.basename(p).replace(/\.[^.]+$/, "").replace(/-/g, "/");
    const ok = fen.split(" ")[0] === truth;
    exact += ok;
    console.log(`  ${((Date.now() - t) / 1000).toFixed(1)} s  ${ok ? "exact" : "DIFF "}  ${fen.split(" ")[0]}  ${path.basename(p)}`);
  }
  if (photos.length) check(exact >= photos.length - 1, `${exact}/${photos.length} photos read exactly`);

  // Engine: best move, then play it.
  const before = (await page.textContent("#fen-text")).trim();
  if (!(await page.isVisible("#fen-status .banner-warn"))) {
    t = Date.now();
    await page.click("#best-move-btn");
    await page.waitForSelector("#play-btn", { timeout: 60000 });
    const best = (await page.textContent(".eval-headline strong")).trim();
    console.log(`  best move ${best} (${((Date.now() - t) / 1000).toFixed(1)} s); candidates: ${await page.locator(".candidates li").count()}`);
    check(/^[KQRBNa-h]/.test(best) || best.startsWith("O-O"), "engine returned a SAN move");
    check(await page.locator("#board-svg svg .arrow, #board-svg svg line, #board-svg svg polygon").count() > 0, "best-move arrow drawn");
    await page.click("#play-btn");
    await page.waitForFunction((b) => document.querySelector("#fen-text").textContent.trim() !== b, before);
    const after = (await page.textContent("#fen-text")).trim();
    check(after.split(" ")[1] !== before.split(" ")[1], `move played, side to move flipped (${after})`);
  }

  // Correct a square: put a white queen on d4... then empty it again.
  await page.click('.square-btn[data-square="d4"]');
  await page.click('.picker-btn[data-piece="wQ"]');
  await page.waitForFunction(() => /Q/.test(document.querySelector("#fen-text").textContent.split(" ")[0].split("/")[4]));
  check(true, "square correction updates the FEN");
  await page.click('.square-btn[data-square="d4"]');
  await page.click('.picker-btn[data-piece="empty"]');

  // Drag a corner and re-read.
  const h = page.locator('.corner-handle[data-i="0"]');
  const box = await h.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 12, box.y + 9, { steps: 4 });
  await page.mouse.up();
  await page.click("#reread-btn");
  await page.waitForSelector("#reanalyze-spinner", { state: "visible", timeout: 5000 }).catch(() => {});
  await page.waitForSelector("#reanalyze-spinner", { state: "hidden", timeout: 120000 });
  check((await page.textContent("#corner-hint")).includes("Using the corners you marked"), "re-read with user corners");

  // Not a chessboard -> clear message, no crash.
  await page.click("#new-photo");
  await page.setInputFiles("#photo-input", path.join(WEB, "..", "tests", "fixtures", "not_a_board", "coffee.jpg"));
  await page.click("#analyze-btn");
  await page.waitForSelector("#analyze-view:not([hidden])", { timeout: 120000 });
  check(await page.isVisible("#status-banner .banner-warn"), "non-board photo: 'couldn't find a chessboard' banner");

  await page.screenshot({ path: path.join(WEB, "dist", "e2e-last.png"), fullPage: true });
  check(errors.length === 0, `no page errors${errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""}`);
} finally {
  await browser.close();
  server.kill();
}
console.log(failures ? `${failures} check(s) failed` : "all checks passed");
process.exit(failures ? 1 : 0);

// Chess Vision Coach -- browser UI. Vision runs in vision-worker.js (Python
// via Pyodide), the engine in a Stockfish WASM worker. Nothing is uploaded.

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- workers --

// ?runtime=opencv forces the compatibility path (what browsers without JSPI get).
const workerUrl = new URL("./vision-worker.js", import.meta.url);
const forced = new URLSearchParams(location.search).get("runtime");
if (forced) workerUrl.searchParams.set("runtime", forced);
const vision = new Worker(workerUrl, { type: "module" });
let nextId = 1;
const pending = new Map();
vision.onmessage = ({ data }) => {
  if (data.type === "progress") return onBootProgress(data);
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  data.ok ? p.resolve(data.result) : p.reject(new Error(data.error));
};
vision.onerror = (e) => showBootError(e.message || "The chess reader failed to start.");
function call(cmd, args, transfer) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    vision.postMessage({ id, cmd, args }, transfer || []);
  });
}

// Engine strength presets -> (depth, time limit ms); same as the server.
const LEVELS = { fast: [12, 400], normal: [18, 1000], strong: [22, 2000] };

class Engine {
  constructor() {
    this.worker = null;
    this.listeners = [];
  }
  start() {
    if (this.worker) return this.ready;
    this.worker = new Worker(new URL("./engine/stockfish-19-lite-single.js", import.meta.url));
    this.worker.onmessage = (e) => {
      const line = typeof e.data === "string" ? e.data : "";
      this.listeners.forEach((fn) => fn(line));
    };
    this.ready = this.until("uciok", "uci").then(() => this.until("readyok", "isready"));
    return this.ready;
  }
  send(cmd) { this.worker.postMessage(cmd); }
  until(prefix, cmd) {
    return new Promise((resolve) => {
      const fn = (line) => {
        if (line.startsWith(prefix)) {
          this.listeners = this.listeners.filter((f) => f !== fn);
          resolve(line);
        }
      };
      this.listeners.push(fn);
      if (cmd) this.send(cmd);
    });
  }
  async analyse(fen, level, multipv = 3) {
    await this.start();
    const [depth, ms] = LEVELS[level] || LEVELS.normal;
    const lines = new Map();
    const collect = (line) => {
      if (!line.startsWith("info ") || !line.includes(" pv ")) return;
      const tok = line.split(" ");
      const at = (k) => tok.indexOf(k);
      const k = at("multipv") >= 0 ? +tok[at("multipv") + 1] : 1;
      const s = at("score");
      if (s < 0) return;
      const score = { kind: tok[s + 1], value: +tok[s + 2] };
      lines.set(k, { score, pv: tok.slice(at("pv") + 1) });
    };
    this.listeners.push(collect);
    this.send("ucinewgame");
    this.send(`setoption name MultiPV value ${multipv}`);
    this.send(`position fen ${fen}`);
    await this.until("readyok", "isready");
    const best = await this.until("bestmove", `go depth ${depth} movetime ${ms}`);
    this.listeners = this.listeners.filter((f) => f !== collect);
    return { bestmove: best.split(" ")[1], lines: [...lines.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]) };
  }
}
const engine = new Engine();

// ------------------------------------------------------------------ state --

const state = {
  ready: false,
  fen: "",
  psc: null,              // per-square confidence {e4: 0.7, ...}
  corners: null,          // [[x,y] x4] in [0,1] of the displayed photo
  cornersByUser: false,
  detectionFailed: false,
  confidence: 0,
  notes: [],
  arrow: null,
  photoUrl: null,
  busy: false,
};

const radio = (form, name) => form.querySelector(`input[name="${name}"]:checked`).value;
const setRadio = (form, name, value) => {
  form.querySelectorAll(`input[name="${name}"]`).forEach((el) => { el.checked = el.value === value; });
};

// ------------------------------------------------------------------- boot --

function onBootProgress({ text, pct }) {
  $("boot-text").textContent = text;
  $("boot-fill").style.width = `${pct}%`;
}
function showBootError(message) {
  $("boot").innerHTML = `<div class="banner banner-error">Couldn't start the chess reader: ${escapeHtml(message)}.
    Try reloading, or a recent Chrome, Edge, Firefox or Safari.</div>`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function boot() {
  showView("upload");
  $("boot").hidden = false;
  try {
    const info = await call("boot");
    state.ready = true;
    $("boot").hidden = true;
    $("analyze-btn").disabled = false;
    $("runtime-note").textContent = info.runtime === "onnxruntime-web"
      ? "Fast mode (onnxruntime-web)." : "Compatibility mode: reading a photo takes a few seconds.";
  } catch (err) {
    showBootError(err.message);
  }
}

function showView(which) {
  $("upload-view").hidden = which !== "upload";
  $("analyze-view").hidden = which !== "analyze";
  hidePicker();
}

// ----------------------------------------------------------------- upload --

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("photo-input").files[0];
  if (!file || !state.ready || state.busy) return;
  $("upload-error").hidden = true;
  if (file.size > 25 * 1024 * 1024) return uploadError("That photo is larger than 25 MB. Please choose a smaller one.");
  const form = $("upload-form");
  const orientation = radio(form, "orientation");
  const side = radio(form, "side_to_move");
  setRadio($("reanalyze-form"), "orientation", orientation);
  setRadio($("reanalyze-form"), "side_to_move", side);
  const bytes = await file.arrayBuffer();
  await runAnalysis({ bytes, orientation, side, corners: null, spinner: "upload-spinner" });
});

function uploadError(msg) {
  $("upload-error").textContent = msg;
  $("upload-error").hidden = false;
  showView("upload");
}

async function runAnalysis({ bytes, orientation, side, corners, spinner }) {
  state.busy = true;
  $(spinner).hidden = false;
  $("analyze-btn").disabled = true;
  $("reread-btn").disabled = true;
  try {
    const res = await call("analyze", { bytes, orientation, sideToMove: side, corners }, bytes ? [bytes] : []);
    if (res.error) return uploadError(res.error);
    if (bytes) {
      const jpeg = await call("photo");
      if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
      state.photoUrl = URL.createObjectURL(new Blob([jpeg], { type: "image/jpeg" }));
      $("photo").src = state.photoUrl;
    }
    Object.assign(state, {
      fen: res.fen,
      psc: res.per_square_confidence,
      corners: res.corners,
      cornersByUser: corners != null,
      detectionFailed: res.detection_failed,
      confidence: res.confidence,
      notes: res.notes || [],
      arrow: null,
    });
    showView("analyze");
    renderAnalysis();
  } catch (err) {
    uploadError(`Something went wrong reading that photo: ${err.message}`);
  } finally {
    state.busy = false;
    $(spinner).hidden = true;
    $("analyze-btn").disabled = false;
    $("reread-btn").disabled = false;
  }
}

// --------------------------------------------------------------- analysis --

function renderAnalysis() {
  const status = $("status-banner");
  if (state.detectionFailed) {
    status.innerHTML = `<div class="banner banner-warn">Couldn't find a chessboard in this photo. Drag the 4 dots on
      your photo onto the board's corners and press <b>Re-read board</b> &mdash; or set up the position by tapping squares.</div>`;
  } else {
    status.innerHTML = `<div class="banner banner-info">Parsed at ${Math.round(state.confidence * 100)}% confidence.
      Tap any square to correct a misread piece.</div>`;
  }
  $("notes-banner").innerHTML = state.notes.length
    ? `<div class="banner banner-notes"><strong>Auto-checked while parsing:</strong><ul class="vision-notes">${
      state.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul></div>` : "";
  $("corner-hint").innerHTML = state.detectionFailed
    ? "Drag the 4 dots onto the board's corners, then press <b>Re-read board</b>."
    : state.cornersByUser ? "Using the corners you marked."
      : "Green outline wrong? Drag its 4 dots onto the board's corners.";
  $("reread-btn").classList.toggle("btn-primary", state.detectionFailed);
  $("best-move-panel").innerHTML = "";
  initCorners();
  renderBoard();
}

async function renderBoard() {
  const v = await call("view", { fen: state.fen, psc: state.psc, arrow: state.arrow });
  $("board-svg").innerHTML = v.svg;
  $("fen-text").textContent = state.fen;
  $("lichess-link").href = `https://lichess.org/analysis/standard/${state.fen.replace(/ /g, "_")}`;
  $("conf-legend").hidden = !v.low_conf_count;
  $("conf-count").textContent = `${v.low_conf_count} square${v.low_conf_count === 1 ? "" : "s"} worth a check`;
  state.problem = v.problem;
  $("fen-status").innerHTML = v.problem
    ? `<div class="banner banner-warn">&#9888; ${escapeHtml(v.problem)} Fix it before getting the best move.</div>`
    : `<div class="banner banner-ok">&#10003; Position is legal &mdash; ready for analysis.</div>`;
}

// Click-to-correct grid over the board SVG (a8 top-left, like the server).
(function buildSquareGrid() {
  const grid = $("square-grid");
  for (let rank = 8; rank >= 1; rank--) {
    for (const file of "abcdefgh") {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "square-btn";
      b.title = b.dataset.square = `${file}${rank}`;
      b.addEventListener("click", (e) => openPicker(e, b));
      grid.appendChild(b);
    }
  }
})();

let pickerSquare = null;
function openPicker(evt, btn) {
  evt.stopPropagation();
  pickerSquare = btn.dataset.square;
  document.querySelectorAll(".square-btn.selected").forEach((b) => b.classList.remove("selected"));
  btn.classList.add("selected");
  const picker = $("piece-picker");
  $("picker-square").textContent = pickerSquare;
  picker.style.display = "block";
  const rect = btn.getBoundingClientRect();
  let left = window.scrollX + rect.left;
  const maxLeft = window.scrollX + document.documentElement.clientWidth - picker.offsetWidth - 8;
  left = Math.max(8, Math.min(left, maxLeft));
  picker.style.top = `${window.scrollY + rect.bottom + 6}px`;
  picker.style.left = `${left}px`;
}
function hidePicker() {
  $("piece-picker").style.display = "none";
  document.querySelectorAll(".square-btn.selected").forEach((b) => b.classList.remove("selected"));
}
document.addEventListener("click", (e) => {
  if ($("piece-picker").style.display !== "block") return;
  if (e.target.closest("#piece-picker") || e.target.closest(".square-btn")) return;
  hidePicker();
});
document.querySelectorAll(".picker-btn").forEach((b) => b.addEventListener("click", async () => {
  if (!pickerSquare) return;
  state.fen = await call("correct", { fen: state.fen, square: pickerSquare, piece: b.dataset.piece });
  if (state.psc) state.psc[pickerSquare] = 1.0;   // the user vouched for this square
  state.arrow = null;
  $("best-move-panel").innerHTML = "";
  hidePicker();
  renderBoard();
}));

// Orientation / side to move on the analysis screen.
$("reanalyze-form").addEventListener("change", async (e) => {
  const form = $("reanalyze-form");
  if (e.target.name === "side_to_move") {
    // Same pieces, other side to move: no need to re-read the photo.
    state.fen = await call("sideToMove", { fen: state.fen, side: radio(form, "side_to_move") });
    state.arrow = null;
    $("best-move-panel").innerHTML = "";
    return renderBoard();
  }
  await runAnalysis({
    bytes: null, orientation: radio(form, "orientation"), side: radio(form, "side_to_move"),
    corners: state.cornersByUser ? state.corners : null, spinner: "reanalyze-spinner",
  });
});

$("reread-btn").addEventListener("click", () => {
  const form = $("reanalyze-form");
  runAnalysis({
    bytes: null, orientation: radio(form, "orientation"), side: radio(form, "side_to_move"),
    corners: state.corners, spinner: "reanalyze-spinner",
  });
});

$("new-photo").addEventListener("click", () => {
  $("photo-input").value = "";
  showView("upload");
  window.scrollTo(0, 0);
});

$("copy-fen").addEventListener("click", (e) => {
  if (navigator.clipboard) navigator.clipboard.writeText(state.fen);
  const btn = e.target;
  const old = btn.textContent;
  btn.textContent = "Copied!";
  setTimeout(() => { btn.textContent = old; }, 1200);
});

$("download-fen").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([`${state.fen}\n`], { type: "text/plain" }));
  a.download = "position.fen";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
});

// ------------------------------------------------------------------ engine --

function formatScore({ kind, value }) {
  if (kind === "mate") return `Mate in ${value}`;
  return `${value >= 0 ? "+" : ""}${(value / 100).toFixed(2)}`;
}

$("best-move-btn").addEventListener("click", async () => {
  const panel = $("best-move-panel");
  if (state.problem) {
    panel.innerHTML = `<div class="banner banner-error">${escapeHtml(state.problem)}</div>`;
    return;
  }
  const over = await call("gameOver", { fen: state.fen });
  if (over) {
    panel.innerHTML = `<div class="eval-headline">${escapeHtml(over)} &mdash; the game is already over.</div>`;
    return;
  }
  $("engine-spinner").hidden = false;
  $("best-move-btn").disabled = true;
  try {
    const fen = state.fen;
    const res = await engine.analyse(fen, $("level").value);
    if (fen !== state.fen) return;          // position changed meanwhile
    if (!res.lines.length || res.bestmove === "(none)") {
      panel.innerHTML = `<div class="banner banner-error">Stockfish returned no moves for this position.</div>`;
      return;
    }
    const cands = [];
    for (const line of res.lines) {
      const san = await call("pvToSan", { fen, moves: line.pv.slice(0, 1) });
      const pv = await call("pvToSan", { fen, moves: line.pv.slice(0, 10) });
      cands.push({ uci: line.pv[0], san: san.replace(/^\d+\.(\.\.)?\s*/, ""), line: pv, evalText: formatScore(line.score) });
    }
    const best = cands[0];
    state.arrow = best.uci;
    renderBoard();
    panel.innerHTML = `
      <div class="eval-headline"><span>Best move</span> <strong>${escapeHtml(best.san)}</strong>
        <span class="eval-score">${escapeHtml(best.evalText)}</span></div>
      <p class="muted">Evaluation is from the side-to-move's point of view (positive = better for them).</p>
      <button class="btn btn-play" id="play-btn">&#9654;&nbsp; Play ${escapeHtml(best.san)} on the board</button>
      <ol class="candidates">${cands.map((c, i) => `
        <li><span class="cand-rank">${i + 1}</span><div class="cand-body">
          <div class="cand-head"><span class="cand-san">${escapeHtml(c.san)}</span>
            <span class="cand-eval">${escapeHtml(c.evalText)}</span></div>
          ${c.line ? `<div class="cand-line">${escapeHtml(c.line)}</div>` : ""}</div></li>`).join("")}
      </ol>`;
    $("play-btn").addEventListener("click", async () => {
      const next = await call("play", { fen: state.fen, uci: best.uci });
      if (!next) return;
      state.fen = next;
      state.psc = null;          // the position changed: old confidences are stale
      state.notes = [];
      state.arrow = null;
      $("notes-banner").innerHTML = "";
      panel.innerHTML = "";
      renderBoard();
    });
  } catch (err) {
    panel.innerHTML = `<div class="banner banner-error">Engine error: ${escapeHtml(err.message)}</div>`;
  } finally {
    $("engine-spinner").hidden = true;
    $("best-move-btn").disabled = false;
  }
});

// ---------------------------------------------------------------- corners --

let cornersBound = false;
function initCorners() {
  const wrap = $("photo-wrap");
  let pts = state.corners && state.corners.length === 4 ? state.corners.map((p) => [...p]) : null;
  wrap.classList.toggle("editing", !pts);
  if (!pts) pts = [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]];
  state.corners = pts;
  drawCorners();
  if (cornersBound) return;
  cornersBound = true;

  const handles = wrap.querySelectorAll(".corner-handle");
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const moved = () => { wrap.classList.add("editing"); $("reread-btn").classList.add("btn-primary"); };
  // Window-level listeners rather than pointer capture: with capture, the
  // first tap after a finger drag was swallowed on touch screens.
  let active = -1;
  const onMove = (e) => {
    if (active < 0) return;
    const r = wrap.getBoundingClientRect();
    state.corners[active] = [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
    drawCorners();
    moved();
  };
  const onUp = () => {
    if (active < 0) return;
    handles[active].classList.remove("dragging");
    active = -1;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  };
  handles.forEach((h) => {
    const i = +h.dataset.i;
    h.addEventListener("pointerdown", () => {
      active = i;
      h.classList.add("dragging");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
    });
    h.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 0.02 : 0.004;
      const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
      if (!d) return;
      e.preventDefault();
      state.corners[i] = [clamp(state.corners[i][0] + d[0]), clamp(state.corners[i][1] + d[1])];
      drawCorners();
      moved();
    });
  });
}

function drawCorners() {
  const pts = state.corners;
  $("board-poly").setAttribute("points", pts.map((p) => `${p[0]},${p[1]}`).join(" "));
  document.querySelectorAll(".corner-handle").forEach((h, i) => {
    h.style.left = `${pts[i][0] * 100}%`;
    h.style.top = `${pts[i][1] * 100}%`;
  });
}

boot();

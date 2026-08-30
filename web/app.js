/* Muscle Memory frontend: atlas canvas + inspector + text search + live view.
   Vanilla JS, no build step. */
"use strict";

const SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"];
const OTHER = "#6f6e68";

const S = {
  meta: null, atlas: null,
  pts: [],                    // [x, y, taskIdx, rowIdx]
  taskColor: new Map(),       // taskIdx -> color
  view: { scale: 1, tx: 0, ty: 0 },
  hover: null, selected: null,
  isolate: null,              // taskIdx or null
  matchSet: null,             // Set(rowIdx) from text/similar search
  rowPos: new Map(),          // rowIdx -> [x, y] (atlas points only)
};

const $ = (id) => document.getElementById(id);
const map = $("map"), ctx = map.getContext("2d");

async function boot() {
  S.meta = await (await fetch("/api/meta")).json();
  S.atlas = await (await fetch("/api/atlas")).json();
  S.pts = S.atlas.points;
  for (const p of S.pts) S.rowPos.set(p[3], p);

  $("meta-stats").textContent =
    ` ${S.atlas.n_total.toLocaleString()} moments · ${S.meta.n_clips} clips · ${S.meta.tasks.length} tasks`;
  $("space-badge").textContent = `motion: ${S.meta.motion_space}`;
  $("phone-url").textContent = `${location.protocol}//${location.host}/phone`;

  // top-8 tasks by window count get the validated categorical slots, rest fold to gray
  const order = S.atlas.tasks.map((_, i) => i)
    .sort((a, b) => S.atlas.task_counts[b] - S.atlas.task_counts[a]);
  order.forEach((ti, rank) => S.taskColor.set(ti, rank < 8 ? SLOTS[rank] : OTHER));
  buildLegend(order);

  resize(); fitView(); draw();
  connectLive();
}

/* ---------------- atlas rendering ---------------- */

function resize() {
  const r = map.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  map.width = r.width * dpr; map.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function fitView() {
  const r = map.parentElement.getBoundingClientRect();
  if (!r.width || !r.height) {
    // layout not settled yet (fast localhost fetches can beat first layout)
    requestAnimationFrame(() => { resize(); fitView(); draw(); });
    return;
  }
  S.view.scale = Math.min(r.width, r.height) * 0.42;
  S.view.tx = r.width / 2; S.view.ty = r.height / 2;
}

const toScreen = (x, y) => [x * S.view.scale + S.view.tx, y * S.view.scale + S.view.ty];

function draw() {
  const r = map.parentElement.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  const hasMatch = S.matchSet !== null;
  // pass 1: base points
  for (const [x, y, ti, row] of S.pts) {
    const dimIso = S.isolate !== null && ti !== S.isolate;
    const dimMatch = hasMatch && !S.matchSet.has(row);
    const [sx, sy] = toScreen(x, y);
    if (sx < -4 || sy < -4 || sx > r.width + 4 || sy > r.height + 4) continue;
    ctx.globalAlpha = (dimIso || dimMatch) ? 0.05 : 0.75;
    ctx.fillStyle = S.taskColor.get(ti);
    ctx.fillRect(sx - 1.1, sy - 1.1, 2.2, 2.2);
  }
  ctx.globalAlpha = 1;
  // pass 2: match rings
  if (hasMatch) {
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
    for (const row of S.matchSet) {
      const p = S.rowPos.get(row);
      if (!p) continue;
      const [sx, sy] = toScreen(p[0], p[1]);
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 7); ctx.stroke();
    }
  }
  // selected marker
  if (S.selected && S.rowPos.has(S.selected.row)) {
    const p = S.rowPos.get(S.selected.row);
    const [sx, sy] = toScreen(p[0], p[1]);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, 7); ctx.stroke();
  }
}

function nearest(mx, my, maxPx = 10) {
  let best = null, bd = maxPx * maxPx;
  for (const p of S.pts) {
    if (S.isolate !== null && p[2] !== S.isolate) continue;
    const [sx, sy] = toScreen(p[0], p[1]);
    const d = (sx - mx) ** 2 + (sy - my) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

/* ---------------- interactions ---------------- */

let dragging = false, moved = false, lx = 0, ly = 0;
map.addEventListener("mousedown", (e) => { dragging = true; moved = false; lx = e.clientX; ly = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
map.addEventListener("mousemove", (e) => {
  const rect = map.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (dragging) {
    if (Math.abs(e.clientX - lx) + Math.abs(e.clientY - ly) > 2) moved = true;
    S.view.tx += e.clientX - lx; S.view.ty += e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    draw(); return;
  }
  const p = nearest(mx, my);
  const tip = $("tooltip");
  if (p) {
    tip.hidden = false;
    tip.style.left = (mx + 14) + "px"; tip.style.top = (my + 14) + "px";
    tip.innerHTML = `<div class="tt-task"></div><div class="tt-sub"></div>`;
    tip.children[0].textContent = pretty(S.atlas.tasks[p[2]]);
    tip.children[1].textContent = `row ${p[3]}`;
  } else tip.hidden = true;
});
map.addEventListener("mouseleave", () => { $("tooltip").hidden = true; });
map.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = map.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const f = Math.exp(-e.deltaY * 0.0015);
  S.view.tx = mx - (mx - S.view.tx) * f;
  S.view.ty = my - (my - S.view.ty) * f;
  S.view.scale *= f;
  draw();
}, { passive: false });
map.addEventListener("click", (e) => {
  if (moved) return;
  const rect = map.getBoundingClientRect();
  const p = nearest(e.clientX - rect.left, e.clientY - rect.top);
  if (p) selectRow(p[3]);
});
window.addEventListener("resize", () => { resize(); draw(); });
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    S.matchSet = null; S.isolate = null; S.selected = null;
    $("search").value = "";
    showPanel("empty"); refreshLegend(); draw();
  }
});

/* ---------------- legend ---------------- */

function buildLegend(order) {
  const lg = $("legend");
  lg.innerHTML = "";
  for (const ti of order) {
    const row = document.createElement("div");
    row.className = "lg-row"; row.tabIndex = 0; row.dataset.ti = ti;
    const dot = document.createElement("span");
    dot.className = "lg-dot"; dot.style.background = S.taskColor.get(ti);
    const name = document.createElement("span");
    name.textContent = pretty(S.atlas.tasks[ti]);
    const count = document.createElement("span");
    count.className = "lg-count";
    count.textContent = S.atlas.task_counts[ti].toLocaleString();
    row.append(dot, name, count);
    row.addEventListener("click", () => {
      S.isolate = (S.isolate === ti) ? null : ti;
      refreshLegend(); draw();
    });
    lg.appendChild(row);
  }
}
function refreshLegend() {
  for (const row of document.querySelectorAll(".lg-row"))
    row.classList.toggle("off", S.isolate !== null && Number(row.dataset.ti) !== S.isolate);
}

/* ---------------- inspector panel ---------------- */

function showPanel(which) {
  $("panel-empty").hidden = which !== "empty";
  $("panel-detail").hidden = which !== "detail";
  $("panel-results").hidden = which !== "results";
}

async function selectRow(row) {
  const info = await (await fetch(`/api/window/${row}`)).json();
  S.selected = info;
  showPanel("detail");
  $("d-frame").src = `/api/frame?row=${row}&w=480`;
  $("d-task").textContent = pretty(info.task);
  $("d-sub").textContent = `${info.clip} · t=${info.t}s`;
  const chaps = $("d-chapters");
  chaps.innerHTML = "";
  for (const c of (info.chapters || []).slice(0, 12)) {
    const b = document.createElement("button");
    b.className = "chap"; b.textContent = `⏵ ${c.toFixed(1)}s`;
    b.addEventListener("click", () => jumpTo(info.clip, c));
    chaps.appendChild(b);
  }
  drawSpark(await (await fetch(`/api/imu?row=${row}`)).json());
  draw();
}

async function jumpTo(clipId, t) {
  const r = await (await fetch(`/api/row_at?clip=${encodeURIComponent(clipId)}&t=${t}`)).json();
  if (r.row !== undefined) selectRow(r.row);
}

function drawSpark(d) {
  const c = $("spark");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 70;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  if (!d.t || d.t.length < 2) return;
  const t0 = d.t[0], t1 = d.t[d.t.length - 1];
  const lo = Math.min(...d.mag), hi = Math.max(...d.mag), span = (hi - lo) || 1;
  const X = (t) => 6 + (t - t0) / (t1 - t0) * (W - 12);
  const Y = (v) => H - 10 - (v - lo) / span * (H - 20);
  // contact spike ticks (orange) under the line
  g.strokeStyle = "#d95926"; g.lineWidth = 2;
  for (const ts of d.spikes || []) {
    g.beginPath(); g.moveTo(X(ts), H - 8); g.lineTo(X(ts), H - 2); g.stroke();
  }
  // |a| line (blue, 2px)
  g.strokeStyle = "#3987e5"; g.lineWidth = 2; g.beginPath();
  d.t.forEach((t, i) => { i ? g.lineTo(X(t), Y(d.mag[i])) : g.moveTo(X(t), Y(d.mag[i])); });
  g.stroke();
  // center-of-window marker
  g.strokeStyle = "#c3c2b7"; g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(X(d.center), 4); g.lineTo(X(d.center), H - 4); g.stroke();
  g.setLineDash([]);
}

$("d-similar").addEventListener("click", async () => {
  if (!S.selected) return;
  const res = await (await fetch(`/api/search/window?row=${S.selected.row}`)).json();
  renderResults(`similar to ${pretty(S.selected.task)} @ ${S.selected.t}s`, res.results, false);
});

/* ---------------- text search ---------------- */

async function runTextSearch(q) {
  const res = await (await fetch(`/api/search/text?q=${encodeURIComponent(q)}`)).json();
  renderResults(`“${q}”`, res.results, res.stub);
}

$("search").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const q = e.target.value.trim();
  if (q) runTextSearch(q);
});

/* ---- voice search: browser speech recognition; with smart glasses paired as
   the Bluetooth mic, this is literally "ask your glasses" ---- */
const SEARCH_PLACEHOLDER = $("search").placeholder;
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SR) {
  const mic = $("mic");
  mic.hidden = false;
  let rec = null;
  mic.addEventListener("click", () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = true;
    mic.classList.add("listening");
    $("search").value = "";
    $("search").placeholder = "listening…";
    rec.addEventListener("result", (e) => {
      const txt = Array.from(e.results).map((r) => r[0].transcript).join(" ").trim();
      $("search").value = txt;
      if (txt && e.results[e.results.length - 1].isFinal) runTextSearch(txt);
    });
    rec.addEventListener("error", (e) => {
      $("search").placeholder = `voice error: ${e.error}`;
      setTimeout(() => { $("search").placeholder = SEARCH_PLACEHOLDER; }, 2500);
    });
    rec.addEventListener("end", () => {
      rec = null;
      mic.classList.remove("listening");
      if ($("search").placeholder === "listening…")
        $("search").placeholder = SEARCH_PLACEHOLDER;
    });
    rec.start();
  });
}

function renderResults(title, results, stub) {
  showPanel("results");
  $("r-title").textContent = `results — ${title}`;
  const list = $("r-list");
  list.innerHTML = stub
    ? `<div class="stub-note">fixture text search is a placeholder — real semantics need --embedder clip</div>` : "";
  S.matchSet = new Set(results.map((r) => r.row));
  for (const r of results) {
    const item = document.createElement("div");
    item.className = "r-item"; item.tabIndex = 0;
    const img = document.createElement("img");
    img.loading = "lazy"; img.src = `/api/frame?row=${r.row}&w=168`; img.alt = "";
    const txt = document.createElement("div");
    const a = document.createElement("div"); a.className = "r-task"; a.textContent = pretty(r.task);
    const b = document.createElement("div"); b.className = "r-sub";
    b.textContent = `${r.clip} · ${r.t}s · ${r.score.toFixed(3)}`;
    txt.append(a, b); item.append(img, txt);
    item.addEventListener("click", () => selectRow(r.row));
    list.appendChild(item);
  }
  draw();
}

/* ---------------- live view ---------------- */

const tabs = document.querySelectorAll(".tab");
tabs.forEach((t) => t.addEventListener("click", () => {
  tabs.forEach((x) => x.classList.toggle("active", x === t));
  const which = t.dataset.tab;
  $("view-atlas").hidden = which !== "atlas";
  $("view-live").hidden = which !== "live";
  $("view-vision").hidden = which !== "vision";
  if (which === "atlas") { resize(); draw(); }
}));

function matchCard(m) {
  const card = document.createElement("div");
  card.className = "m-card";
  const img = document.createElement("img");
  img.src = `/api/frame?row=${m.row}&w=320`; img.alt = "";
  const body = document.createElement("div"); body.className = "m-body";
  const a = document.createElement("div"); a.className = "r-task"; a.textContent = pretty(m.task);
  const b = document.createElement("div"); b.className = "m-score";
  b.textContent = `${m.clip} · ${m.t}s · ${m.score.toFixed(3)}`;
  body.append(a, b); card.append(img, body);
  return card;
}

function connectLive() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.addEventListener("message", (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "phone")
      $("live-status").textContent = msg.connected ? "phone connected — move it!" : "phone disconnected";
    if (msg.type === "matches") {
      $("live-status").textContent = `matching in ${msg.space} space`;
      drawEnergy(msg.energy);
      const grid = $("live-matches");
      grid.innerHTML = "";
      for (const m of msg.matches) grid.appendChild(matchCard(m));
    }
  });
  ws.addEventListener("close", () => setTimeout(connectLive, 2000));
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
}

function drawEnergy(v) {
  const c = $("energy");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 26;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#3987e5";
  g.fillRect(0, 4, Math.min(1, v / 8) * W, H - 8);
}

/* ---------------- live vision search ----------------
   Sources: device camera, or a screen-captured window — e.g. a WhatsApp video
   call streamed from Ray-Ban smart glasses, answered on this machine. Every
   ~1.2s the current frame goes to /api/search/image (CLIP image tower). */

let vStream = null, vTimer = null, vBusy = false;

async function visionStart(kind) {
  visionStop();
  try {
    vStream = kind === "cam"
      ? await navigator.mediaDevices.getUserMedia({ video: { width: 960 }, audio: false })
      : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
  } catch (e) {
    $("vision-status").textContent = `could not open source: ${e.message}`;
    return;
  }
  const v = $("v-preview");
  v.srcObject = vStream;
  v.hidden = false;
  $("v-stop").hidden = false;
  $("vision-status").textContent = "searching by sight…";
  vStream.getVideoTracks()[0].addEventListener("ended", visionStop);
  vTimer = setInterval(visionTick, 1200);
}

async function visionTick() {
  const v = $("v-preview");
  if (vBusy || !vStream || !v.videoWidth) return;
  vBusy = true;
  try {
    const c = document.createElement("canvas");
    const w = 480, h = Math.max(2, Math.round((v.videoHeight * w) / v.videoWidth));
    c.width = w; c.height = h;
    c.getContext("2d").drawImage(v, 0, 0, w, h);
    const res = await (await fetch("/api/search/image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: c.toDataURL("image/jpeg", 0.7), k: 9 }),
    })).json();
    const grid = $("vision-matches");
    grid.innerHTML = res.stub
      ? `<div class="stub-note">fixture vision search is a placeholder — real semantics need --embedder clip</div>`
      : "";
    for (const m of res.results || []) grid.appendChild(matchCard(m));
  } catch (e) {
    $("vision-status").textContent = `search error: ${e.message}`;
  } finally {
    vBusy = false;
  }
}

function visionStop() {
  if (vTimer) { clearInterval(vTimer); vTimer = null; }
  if (vStream) { vStream.getTracks().forEach((t) => t.stop()); vStream = null; }
  const v = $("v-preview");
  v.srcObject = null; v.hidden = true;
  $("v-stop").hidden = true;
  $("vision-status").textContent = "no source yet";
}

$("v-cam").addEventListener("click", () => visionStart("cam"));
$("v-screen").addEventListener("click", () => visionStart("screen"));
$("v-stop").addEventListener("click", visionStop);

const pretty = (s) => String(s).replace(/[_-]+/g, " ");

boot();

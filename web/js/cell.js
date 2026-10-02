// Mnx Hive — a Bee's Cell: terminal, browser, jobs and files that run 24/7 on the server.

import { Terminal } from "../vendor/xterm/xterm.mjs";
import { FitAddon } from "../vendor/xterm/addon-fit.mjs";
import { ICONS, api, beeHex, connLabel, fillIcons, gatewayBase, getSettings, h, installTokenPrompt, timeShort } from "./shared.js";

const $ = (id) => document.getElementById(id);
fillIcons();

const beeId = new URLSearchParams(location.search).get("bee") || "browser";
const base = `/api/cells/${encodeURIComponent(beeId)}`;
let info = null;

// ---------- header ----------
async function load() {
  try {
    info = await api(base);
    connLabel($("conn"), "online");
  } catch (e) {
    connLabel($("conn"), e.status === 401 ? "auth" : "offline");
    if (e.status === 404) $("title").textContent = "No such Bee";
    return;
  }
  const { bee, cell, jobs, disk } = info;
  $("title").textContent = `${bee.name} Bee · Cell`;
  document.title = `${bee.name} Cell · Mnx Hive`;
  $("beeHex").innerHTML = beeHex(bee.name, bee.color, "hex head-hex");
  const state = cell.container;
  $("chips").replaceChildren(...[
    h("span", { class: `status ${cell.running ? "busy" : cell.failed ? "failed" : jobs.some((j) => j.enabled && j.schedule !== "@manual") ? "scheduled" : "idle"}` },
      cell.running ? `${cell.running} job${cell.running > 1 ? "s" : ""} running` : cell.failed ? "A job failed" : "Idle"),
    h("span", { class: "chip" }, cell.mode === "docker" ? (state === "missing" ? "Container not started yet" : `Container: ${state}`) : "Local folder (no container)"),
    cell.image ? h("span", { class: "chip" }, cell.image) : null,
    h("span", { class: "chip" }, `Storage ${fmtSize(disk)}`),
    cell.mode === "docker" && state !== "running"
      ? h("button", { class: "btn primary", type: "button", onclick: startCell }, "Start Cell") : null,
  ].filter(Boolean));
  $("jobCount").textContent = jobs.length ? `(${jobs.length})` : "";
  renderJobs(jobs);
}

async function startCell(e) {
  e.target.disabled = true;
  e.target.textContent = "Starting…";
  try { await api(`${base}/start`, { method: "POST" }); } catch (err) { alert(err.message); }
  load();
}

function fmtSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

// ---------- tabs ----------
const shown = new Set();
function showTab(name) {
  document.querySelectorAll(".cell-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  for (const n of ["terminal", "browser", "jobs", "files"]) $(`tab-${n}`).classList.toggle("hidden", n !== name);
  if (name === "terminal") { openTerminal(); requestAnimationFrame(() => fit?.fit()); }
  if (name === "browser" && !shown.has("browser")) $("browserFrame").src = `browser.html?bee=${encodeURIComponent(beeId)}&embed=1`;
  if (name === "files") loadFiles(cwd);
  shown.add(name);
  history.replaceState(null, "", `?bee=${encodeURIComponent(beeId)}#${name}`);
}
document.querySelectorAll(".cell-tabs .tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// ---------- terminal ----------
let term, fit, ws, wsRetry = 0;
function openTerminal() {
  if (term) return;
  term = new Terminal({
    cursorBlink: true, fontSize: 13, scrollback: 5000,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
    theme: { background: "#0b0b0d", foreground: "#e6e6ea", cursor: "#f5a623", selectionBackground: "#f5a62355" },
  });
  fit = new FitAddon();
  term.loadAddon(fit);
  term.open($("term"));
  fit.fit();
  term.onData((d) => send({ t: "in", d }));
  new ResizeObserver(() => { try { fit.fit(); } catch { /* hidden */ } }).observe($("term"));
  term.onResize(({ cols, rows }) => send({ t: "resize", cols, rows }));
  connectTerm();
}
function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}
function connectTerm() {
  const u = new URL(gatewayBase());
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = `/ws/term/${encodeURIComponent(beeId)}`;
  u.search = `?token=${encodeURIComponent(getSettings().token)}`;
  ws = new WebSocket(u);
  ws.binaryType = "arraybuffer";
  let first = true;
  ws.onopen = () => { wsRetry = 0; $("termNote").textContent = "The shell keeps running when you leave; come back and you'll see what happened."; };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      const m = JSON.parse(e.data);
      if (m.t === "error") term.write(`\r\n\x1b[31m${m.d}\x1b[0m\r\n`);
      return;
    }
    if (first) { term.reset(); first = false; send({ t: "resize", cols: term.cols, rows: term.rows }); }
    term.write(new Uint8Array(e.data));
  };
  ws.onclose = (e) => {
    if (e.code === 4401) { window.dispatchEvent(new CustomEvent("mnx-auth")); return; }
    $("termNote").textContent = "Reconnecting to the terminal… (the shell is still running on the server)";
    setTimeout(connectTerm, Math.min(10000, 1000 * 2 ** wsRetry++));
  };
}

// ---------- jobs ----------
const form = $("jobForm");
form.when.addEventListener("change", () => {
  for (const k of ["every", "daily", "cron"]) form.querySelector(`.when-${k}`).classList.toggle("hidden", form.when.value !== k);
});
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("jobErr").textContent = "";
  let schedule;
  switch (form.when.value) {
    case "always": schedule = "@always"; break;
    case "manual": schedule = "@manual"; break;
    case "every": schedule = `*/${Math.max(1, Math.min(59, Number(form.every.value) || 15))} * * * *`; break;
    case "daily": { const [hh, mm] = (form.time.value || "08:00").split(":").map(Number); schedule = `${mm} ${hh} * * *`; break; }
    default: schedule = form.cron.value.trim();
  }
  try {
    await api(`${base}/jobs`, { method: "POST", body: { name: form.name.value.trim(), command: form.command.value, schedule } });
    form.reset();
    form.when.dispatchEvent(new Event("change"));
    load();
  } catch (err) {
    $("jobErr").textContent = err.message;
  }
});

function describe(s) {
  if (s === "@always") return "Always on";
  if (s === "@manual") return "When you press Run";
  let m = s.match(/^\*\/(\d+) \* \* \* \*$/);
  if (m) return `Every ${m[1]} min`;
  m = s.match(/^(\d+) (\d+) \* \* \*$/);
  if (m) return `Every day at ${m[2].padStart(2, "0")}:${m[1].padStart(2, "0")}`;
  return `Cron: ${s}`;
}

const openLogs = new Set();
function renderJobs(jobs) {
  const box = $("jobs");
  const keepScroll = {};
  box.querySelectorAll("pre[data-job]").forEach((p) => { keepScroll[p.dataset.job] = p.scrollTop; });
  box.replaceChildren();
  if (!jobs.length) {
    box.append(h("p", { class: "empty-note" }, "No jobs yet. Add one above: it keeps running here even when your phone is off."));
    return;
  }
  for (const j of jobs) {
    const status = j.running ? h("span", { class: "status busy" }, "Running")
      : !j.enabled ? h("span", { class: "status" }, "Paused")
      : j.last_exit != null && j.last_exit !== 0 ? h("span", { class: "status failed" }, `Exit ${j.last_exit}`)
      : h("span", { class: `status ${j.schedule === "@manual" ? "" : "scheduled"}` }, j.last_exit === 0 ? "OK" : "Waiting");
    const logBox = h("pre", { class: "result-box hidden", "data-job": j.id });
    const toggleLog = async () => {
      if (openLogs.has(j.id)) { openLogs.delete(j.id); logBox.classList.add("hidden"); return; }
      openLogs.add(j.id);
      logBox.classList.remove("hidden");
      await fillLog(j.id, logBox);
    };
    if (openLogs.has(j.id)) { logBox.classList.remove("hidden"); fillLog(j.id, logBox); }
    const call = (path, opts) => async (e) => {
      e.target.disabled = true;
      try { await api(`${base}/jobs/${j.id}${path}`, { method: "POST", ...opts }); } catch (err) { alert(err.message); }
      setTimeout(load, 300);
    };
    box.append(h("div", { class: "card job-card" },
      h("div", { class: "card-head" },
        h("div", { style: "min-width:0;flex:1" }, h("div", {}, j.name), h("div", { class: "hint mono ellipsis" }, j.command)),
        status),
      h("div", { class: "hint" }, `${describe(j.schedule)} · ${j.runs} run${j.runs === 1 ? "" : "s"}${j.last_start ? ` · last ${timeShort(new Date(j.last_start * 1000).toISOString())}` : ""}`),
      h("div", { class: "card-actions" },
        j.running
          ? h("button", { class: "btn", type: "button", onclick: call("/stop") }, j.schedule === "@always" ? "Stop" : "Stop run")
          : h("button", { class: "btn primary", type: "button", onclick: call("/run") }, j.schedule === "@always" ? "Start" : "Run now"),
        j.schedule !== "@manual" && j.schedule !== "@always"
          ? h("button", { class: "btn", type: "button", onclick: call("/enabled", { body: { enabled: !j.enabled } }) }, j.enabled ? "Pause" : "Resume") : null,
        h("button", { class: "btn", type: "button", onclick: toggleLog }, openLogs.has(j.id) ? "Hide log" : "Log"),
        h("button", { class: "btn danger", type: "button", onclick: async () => {
          if (!confirm(`Delete job "${j.name}" and its log?`)) return;
          try { await api(`${base}/jobs/${j.id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
          openLogs.delete(j.id);
          load();
        } }, "Delete")),
      logBox));
    if (keepScroll[j.id] != null) logBox.scrollTop = keepScroll[j.id];
  }
}

async function fillLog(id, el) {
  try {
    const res = await fetch(`${gatewayBase()}${base}/jobs/${id}/log`, { headers: { Authorization: `Bearer ${getSettings().token}` } });
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    el.textContent = (await res.text()) || "(no output yet)";
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch { el.textContent = "Couldn't load the log."; }
}

// ---------- files ----------
let cwd = "";
async function loadFiles(path) {
  cwd = path;
  const crumbs = $("crumbs");
  crumbs.replaceChildren(h("button", { class: "crumb", type: "button", onclick: () => loadFiles("") }, "/work"));
  let acc = "";
  for (const part of path.split("/").filter(Boolean)) {
    acc = acc ? `${acc}/${part}` : part;
    const p = acc;
    crumbs.append(" / ", h("button", { class: "crumb", type: "button", onclick: () => loadFiles(p) }, part));
  }
  const box = $("files");
  let list;
  try { list = await api(`${base}/files?path=${encodeURIComponent(path)}`); }
  catch (e) { box.replaceChildren(h("p", { class: "empty-note err" }, e.message)); return; }
  box.replaceChildren();
  if (!list.length) box.append(h("p", { class: "empty-note" }, "Empty. Upload files or create them from the terminal."));
  for (const f of list) {
    const href = `${gatewayBase()}${base}/file?path=${encodeURIComponent(f.path)}&token=${encodeURIComponent(getSettings().token)}`;
    box.append(h("div", { class: "file-row" },
      f.dir
        ? h("button", { class: "file-name", type: "button", onclick: () => loadFiles(f.path) }, `📁 ${f.name}`)
        : h("a", { class: "file-name", href, download: f.name }, f.name),
      h("span", { class: "hint" }, f.dir ? "" : fmtSize(f.size)),
      h("span", { class: "hint" }, new Date(f.mtime * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })),
      h("button", { class: "icon-btn", type: "button", "aria-label": `Delete ${f.name}`, html: ICONS.trash, onclick: async () => {
        if (!confirm(`Delete ${f.name}${f.dir ? " and everything in it" : ""}?`)) return;
        try { await api(`${base}/file?path=${encodeURIComponent(f.path)}`, { method: "DELETE" }); } catch (e) { alert(e.message); }
        loadFiles(cwd);
      } })));
  }
}
$("refreshFiles").addEventListener("click", () => loadFiles(cwd));
$("upload").addEventListener("change", async (e) => {
  for (const file of e.target.files) {
    const path = cwd ? `${cwd}/${file.name}` : file.name;
    const res = await fetch(`${gatewayBase()}${base}/file?path=${encodeURIComponent(path)}`, {
      method: "PUT", body: file, headers: { Authorization: `Bearer ${getSettings().token}` },
    });
    if (!res.ok) alert(`${file.name}: ${(await res.json().catch(() => ({}))).detail || res.statusText}`);
  }
  e.target.value = "";
  loadFiles(cwd);
});

// ---------- live refresh ----------
installTokenPrompt(() => { load(); if (ws) ws.close(); });
setInterval(() => {
  if (document.visibilityState !== "visible") return;
  load();
  for (const id of openLogs) {
    const el = document.querySelector(`pre[data-job="${id}"]`);
    if (el) fillLog(id, el);
  }
}, 4000);

load();
showTab(["terminal", "browser", "jobs", "files"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "terminal");

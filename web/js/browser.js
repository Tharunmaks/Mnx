// Mnx Hive — Browser Bee: watch and drive the Hive's own Chromium.

import { api, connLabel, fillIcons, h, installTokenPrompt } from "./shared.js";

const $ = (id) => document.getElementById(id);
fillIcons();
installTokenPrompt(() => refresh());

let state = { running: false };
let busy = 0;

async function act(action, args = {}) {
  busy++;
  $("busy").classList.remove("hidden");
  $("err").textContent = "";
  try {
    const s = await api("/api/browser", { method: "POST", body: { action, ...args } });
    connLabel($("conn"), "online");
    show(s);
    if (action !== "screenshot" && !$("tab-elements").classList.contains("hidden")) loadElements();
    return s;
  } catch (e) {
    if (e.status === 401) connLabel($("conn"), "auth");
    $("err").textContent = e.message;
  } finally {
    if (--busy === 0) $("busy").classList.add("hidden");
  }
}

function show(s) {
  if (!s) return;
  state = { ...state, ...s };
  const running = !!s.running;
  $("empty").classList.toggle("hidden", running);
  $("shot").classList.toggle("hidden", !running);
  if (s.image) $("shot").src = s.image;
  if (s.url && document.activeElement !== $("url")) $("url").value = s.url;
  $("title").textContent = running ? (s.title || s.url || "") : " ";
}

async function refresh() {
  try {
    const s = await api("/api/browser");
    connLabel($("conn"), "online");
    if (s.running) show(await api("/api/browser", { method: "POST", body: { action: "screenshot" } }));
    else show(s);
  } catch (e) {
    connLabel($("conn"), e.status === 401 ? "auth" : "offline");
  }
}

// ---------- controls ----------
$("urlForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = $("url").value.trim();
  if (url) { act("goto", { url }); $("url").blur(); }
});
$("bBack").addEventListener("click", () => act("back"));
$("bFwd").addEventListener("click", () => act("forward"));
$("bReload").addEventListener("click", () => act("reload"));
$("closeBrowser").addEventListener("click", () => act("close"));

// Click on the picture → click the page at the same spot.
$("shot").addEventListener("click", (e) => {
  const r = e.target.getBoundingClientRect();
  const x = ((e.clientX - r.left) / r.width) * (state.width || 1280);
  const y = ((e.clientY - r.top) / r.height) * (state.height || 800);
  act("click", { x, y });
});
$("shot").addEventListener("wheel", (e) => {
  e.preventDefault();
  clearTimeout(wheelTimer);
  wheelDy += e.deltaY;
  wheelTimer = setTimeout(() => { act("scroll", { dy: wheelDy }); wheelDy = 0; }, 150);
}, { passive: false });
let wheelTimer, wheelDy = 0;

$("typeForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("typeText").value;
  if (!text) return;
  $("typeText").value = "";
  act("type", { text });
});
document.querySelectorAll("[data-key]").forEach((b) => b.addEventListener("click", () => act("press", { key: b.dataset.key })));
document.querySelectorAll("[data-scroll]").forEach((b) => b.addEventListener("click", () => act("scroll", { dy: Number(b.dataset.scroll) })));

// ---------- side tabs ----------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $("tab-elements").classList.toggle("hidden", t.dataset.tab !== "elements");
  $("tab-text").classList.toggle("hidden", t.dataset.tab !== "text");
  if (t.dataset.tab === "text") loadText();
  else loadElements();
}));

async function loadElements() {
  if (!state.running) return;
  try {
    const r = await api("/api/browser", { method: "POST", body: { action: "elements" } });
    const box = $("tab-elements");
    box.replaceChildren();
    if (!r.elements.length) box.append(h("p", { class: "hint" }, "Nothing clickable in view."));
    for (const el of r.elements) {
      box.append(h("button", { class: "el-row", type: "button", onclick: () => act("click", { x: el.x, y: el.y }) },
        h("span", { class: "chip" }, el.tag + (el.type && el.tag === "input" ? `:${el.type}` : "")),
        h("span", { class: "ellipsis" }, el.label || "(no label)")));
    }
  } catch { /* shown elsewhere */ }
}

async function loadText() {
  if (!state.running) return;
  $("pageText").textContent = "Reading…";
  try {
    const r = await api("/api/browser", { method: "POST", body: { action: "read" } });
    $("pageText").textContent = r.text || "(empty page)";
  } catch (e) {
    $("pageText").textContent = e.message;
  }
}

// ---------- live view ----------
setInterval(() => {
  if ($("live").checked && state.running && !busy && document.visibilityState === "visible") {
    api("/api/browser", { method: "POST", body: { action: "screenshot" } }).then(show).catch(() => {});
  }
}, 2500);

const start = new URLSearchParams(location.search).get("url");
if (start) { $("url").value = start; act("goto", { url: start }); }
else refresh();

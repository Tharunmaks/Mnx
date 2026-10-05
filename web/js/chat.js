// Mnx Hive — Chat screen.
// Sends your message to the gateway and renders the events it streams back.
// No model runs in the browser; when the Mnx brain is not wired up, the gateway says so.

import {
  ICONS, HiveClient, api, beeHex, connLabel, getSettings, h, installTokenPrompt, logoSVG, saveSettings, store, uid,
} from "./shared.js";
import { buildModel, renderRun } from "./render.js";

const $ = (id) => document.getElementById(id);
const CHATS_KEY = "mnx.chats";
const ACTIVE_KEY = "mnx.activeChat";

// ---------- icons ----------
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });
$("brandLogo").innerHTML = logoSVG();
$("welcomeLogo").innerHTML = logoSVG("wordmark-hex");
$("pillLogo").innerHTML = logoSVG();

// ---------- state ----------
let chats = store.get(CHATS_KEY, []);
let activeId = store.get(ACTIVE_KEY, null);
if (!chats.find((c) => c.id === activeId)) activeId = null;

// A task still "running" after a reload lost its stream; close it out.
for (const c of chats) {
  for (const t of c.turns) {
    if (t.role === "mnx" && isRunning(t)) t.events.push({ task: t.task, type: "task_error", text: "Interrupted — the page was reloaded." });
  }
}

const runs = new Map(); // task id → { el, cardCache }
let saveTimer = null;

function saveChats() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    store.set(CHATS_KEY, chats);
    store.set(ACTIVE_KEY, activeId);
  }, 150);
}
const activeChat = () => chats.find((c) => c.id === activeId) || null;

function findTurn(task) {
  for (const c of chats) {
    const t = c.turns.find((x) => x.task === task);
    if (t) return { chat: c, turn: t };
  }
  return null;
}

function runningTask() {
  const c = activeChat();
  if (!c) return null;
  const last = c.turns[c.turns.length - 1];
  return last && last.role === "mnx" && isRunning(last) ? last.task : null;
}
function isRunning(turn) {
  return !turn.events.some((e) => e.type === "task_done" || e.type === "task_error");
}

// ---------- gateway ----------
const client = new HiveClient();

client.addEventListener("state", (e) => {
  connLabel($("conn"), e.detail);
  if (e.detail === "online") { loadTools(); loadBees(); }
});
installTokenPrompt(() => client.reconnectNow());

client.addEventListener("event", (e) => {
  const ev = e.detail;
  if (ev.type === "bee_status" || ev.type === "bees_changed") { loadBees(); return; }
  if (ev.type === "tools_changed" || ev.type === "phones_changed") { loadTools(); return; }
  if (!ev.task) return;
  const found = findTurn(ev.task);
  if (!found) return;
  found.turn.events.push(ev);
  if (ev.type === "bee_created") loadBees();
  found.chat.updated = Date.now();
  saveChats();
  if (found.chat.id === activeId) {
    rerenderTurn(found.turn);
    updateSendState();
  }
});

client.connect();
$("conn").addEventListener("click", () => client.reconnectNow());

// ---------- sending ----------
function send(text) {
  text = text.trim();
  if (!text || runningTask()) return;

  let chat = activeChat();
  if (!chat) {
    chat = { id: uid("c_"), title: titleFrom(text), created: Date.now(), updated: Date.now(), turns: [] };
    chats.unshift(chat);
    activeId = chat.id;
  }
  const task = uid("t_");
  chat.turns.push({ role: "user", text, at: new Date().toISOString() });
  const turn = { role: "mnx", task, events: [] };
  chat.turns.push(turn);
  chat.updated = Date.now();

  const ok = client.send({ type: "user_message", chat: chat.id, task, text, name: getSettings().name || undefined });
  if (!ok) {
    turn.events.push(
      { task, type: "error", text: "Can't reach the Mnx Hive gateway", detail: "check that the server is running, or set its URL in Settings" },
      { task, type: "task_error", text: "Message not sent." },
    );
  }
  saveChats();
  renderAll();
}

function titleFrom(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 42 ? t.slice(0, 40) + "…" : t;
}

function onAction(turn) {
  return (ref, action, values, label) => {
    turn.events.push({ task: turn.task, type: "_action", ref, action, label, values });
    saveChats();
    rerenderTurn(turn);
    const ok = client.send({ type: "action", task: turn.task, ref, action, values: values || {} });
    if (!ok) {
      turn.events.push({ task: turn.task, type: "task_error", text: "Lost connection to the gateway." });
      rerenderTurn(turn);
      updateSendState();
    }
  };
}

function stop() {
  const task = runningTask();
  if (!task) return;
  client.send({ type: "stop", task });
  const found = findTurn(task);
  found.turn.events.push({ task, type: "task_error", text: "Stopped." });
  saveChats();
  rerenderTurn(found.turn);
  updateSendState();
}

// ---------- rendering ----------
function renderAll() {
  const chat = activeChat();
  const hasTurns = chat && chat.turns.length;
  $("welcome").classList.toggle("hidden", !!hasTurns);
  $("scroll").classList.toggle("hidden", !hasTurns);
  $("bottomSlot").classList.toggle("hidden", !hasTurns);
  // The composer lives in one place at a time: the welcome screen (no turns) or the bottom bar. Move it from wherever it is.
  const composer = $("composerHome");
  composer.classList.remove("hidden");
  const slot = hasTurns ? $("bottomSlot") : $("welcomeSlot");
  slot.append(...composer.childNodes, ...[...$("welcomeSlot").childNodes, ...$("bottomSlot").childNodes].filter((n) => n.parentNode !== slot));

  $("chatTitle").textContent = chat ? chat.title : "New chat";
  document.title = chat ? `${chat.title} · Mnx Hive` : "Mnx Hive";

  const thread = $("thread");
  thread.replaceChildren();
  runs.clear();
  if (chat) {
    for (const turn of chat.turns) {
      if (turn.role === "user") {
        thread.append(h("div", { class: "msg-user" }, h("div", { class: "bubble" }, turn.text)));
      } else {
        const slot = h("div");
        thread.append(slot);
        runs.set(turn.task, { el: slot, cardCache: new Map() });
        rerenderTurn(turn, false);
      }
    }
  }
  renderHistory();
  updateSendState();
  scrollDown(true);
}

function rerenderTurn(turn, scroll = true) {
  const run = runs.get(turn.task);
  if (!run) return;
  const nearBottom = isNearBottom();
  const el = renderRun(buildModel(turn.events), { onAction: onAction(turn), cardCache: run.cardCache });
  run.el.replaceChildren(el);
  if (scroll && nearBottom) scrollDown();
}

function isNearBottom() {
  const s = $("scroll");
  return s.scrollHeight - s.scrollTop - s.clientHeight < 120;
}
function scrollDown(force) {
  const s = $("scroll");
  if (force || isNearBottom()) requestAnimationFrame(() => { s.scrollTop = s.scrollHeight; });
}

function renderHistory() {
  const box = $("history");
  box.replaceChildren();
  const sorted = [...chats].sort((a, b) => b.updated - a.updated);
  if (!sorted.length) {
    box.append(h("div", { class: "empty-note" }, "No chats yet"));
    return;
  }
  for (const c of sorted) {
    const del = h("button", { class: "del", "aria-label": "Delete chat", html: ICONS.trash, onclick: (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${c.title}"?`)) return;
      chats = chats.filter((x) => x.id !== c.id);
      if (activeId === c.id) activeId = null;
      saveChats();
      renderAll();
    } });
    del.querySelector("svg").style.cssText = "width:15px;height:15px";
    box.append(h("button", {
      class: `rail-btn${c.id === activeId ? " active" : ""}`,
      onclick: () => { activeId = c.id; saveChats(); renderAll(); closeRail(); },
    }, h("span", {}, c.title), del));
  }
}

// ---------- composer ----------
const input = $("input");
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}
function updateSendState() {
  const btn = $("send");
  const running = !!runningTask();
  btn.classList.toggle("stop", running);
  btn.innerHTML = running ? ICONS.stop : ICONS.send;
  btn.setAttribute("aria-label", running ? "Stop" : "Send");
  btn.disabled = !running && !input.value.trim();
}
input.addEventListener("input", () => { autosize(); updateSendState(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && getSettings().enterToSend) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (runningTask()) { stop(); return; }
  const text = input.value;
  if (!text.trim()) return;
  input.value = "";
  autosize();
  send(text);
  input.focus();
});

$("newChat").addEventListener("click", () => {
  activeId = null;
  saveChats();
  renderAll();
  closeRail();
  input.focus();
});

// ---------- profile + settings ----------
function applyProfile() {
  const s = getSettings();
  const name = s.name.trim();
  $("greeting").textContent = name ? `Hi ${name}, let's start?` : "Hi there, let's start?";
  $("avatar").textContent = name ? name.charAt(0).toUpperCase() : "?";
  $("profileName").textContent = name || "Profile";
  $("modelName").textContent = s.model || "Mnx 3B";
}
applyProfile();

function openSettings(focusName) {
  const f = $("settingsForm");
  const s = getSettings();
  f.name.value = s.name;
  f.model.value = s.model;
  f.gateway.value = s.gateway;
  f.token.value = s.token;
  f.enterToSend.checked = s.enterToSend;
  $("settingsDlg").showModal();
  if (focusName) f.name.focus();
}
$("openSettings").addEventListener("click", () => openSettings(false));
$("openProfile").addEventListener("click", () => openSettings(true));
$("modelPill").addEventListener("click", () => openSettings(false));
$("settingsDlg").addEventListener("close", () => {
  if ($("settingsDlg").returnValue !== "save") return;
  const f = $("settingsForm");
  const prev = getSettings();
  saveSettings({
    token: f.token.value.trim(),
    name: f.name.value.trim(),
    model: f.model.value.trim() || "Mnx 3B",
    gateway: f.gateway.value.trim(),
    enterToSend: f.enterToSend.checked,
  });
  applyProfile();
  const now = getSettings();
  if (prev.gateway !== now.gateway || prev.token !== now.token || client.state !== "online") {
    client.reconnectNow();
    loadTools();
    loadBees();
  }
});
$("clearChats").addEventListener("click", () => {
  if (!confirm("Delete every chat saved on this device?")) return;
  chats = [];
  activeId = null;
  saveChats();
  $("settingsDlg").close();
  renderAll();
});

// ---------- connected MCP tools (rail) ----------
async function loadTools() {
  let data = null;
  try { data = await api("/api/connectors"); } catch { /* offline */ }
  const list = $("toolList");
  list.replaceChildren();
  const conns = data ? data.connected : [];
  $("toolCount").textContent = conns.length ? String(conns.length) : "";
  $("phoneCount").textContent = data && data.phones ? String(data.phones) : "";
  if (!conns.length) list.append(h("div", { class: "empty-note" }, data ? "None connected yet" : "Gateway offline"));
  for (const c of conns.slice(0, 6)) {
    list.append(h("a", { class: "rail-btn", href: "connectors.html" },
      h("span", { class: `tool-dot${c.status === "ok" ? " on" : ""}` }), h("span", {}, c.name)));
  }
}

// ---------- Bees side panel ----------
async function loadBees() {
  let bees = [];
  try { bees = await api("/api/bees"); } catch { /* offline */ }
  const list = $("beeList");
  list.replaceChildren();
  $("beeCount").textContent = bees.length || "";
  if (!bees.length) list.append(h("div", { class: "empty-note" }, "No Bees yet."));
  for (const b of bees) {
    const row = h("a", { class: "bee-row", href: `cell.html?bee=${encodeURIComponent(b.id)}`, title: `Open ${b.name}'s Cell` });
    row.innerHTML = beeHex(b.name, b.color);
    row.append(
      h("div", { class: "meta" }, h("div", { class: "name" }, b.name), h("div", { class: "skill" }, b.skill || "")),
      h("span", { class: `status ${b.status}` }, b.status));
    list.append(row);
  }
}
$("hiveToggle").addEventListener("click", () => {
  const p = $("hivePanel");
  if (getComputedStyle(p).position === "absolute" || !p.offsetParent) p.classList.toggle("open");
  else location.href = "hive.html";
});

// ---------- mobile rail ----------
function closeRail() { $("app").classList.remove("rail-open"); }
$("menuBtn").addEventListener("click", () => $("app").classList.add("rail-open"));
$("scrim").addEventListener("click", closeRail);

renderAll();
loadTools();
loadBees();

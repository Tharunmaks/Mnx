// Mnx Hive — Chat screen.
// Sends your message to the gateway and renders the events it streams back.
// No model runs in the browser; when the Mnx brain is not wired up, the gateway says so.

import {
  EFFORT, ICONS, HiveClient, api, beeHex, connLabel, countTokens, effortInfo, fmtTokens, getSettings, h, installTokenPrompt, logoSVG,
  saveSettings, store, toast, uid, whoami,
} from "./shared.js";
import { buildModel, renderRun } from "./render.js";
import { askNotifications, installVoice, speak, startTalk, stopSpeaking } from "./voice.js";
import { installModelPicker, refreshModelPill, selectedModel } from "./models.js";

const $ = (id) => document.getElementById(id);
const CHATS_KEY = "mnx.chats";
const ACTIVE_KEY = "mnx.activeChat";

// ---------- icons ----------
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });
$("brandLogo").innerHTML = logoSVG();
$("welcomeLogo").innerHTML = logoSVG("wordmark-hex");

// ---------- state ----------
let chats = store.get(CHATS_KEY, []);
// Rush: all your Bees work together on one page. Peace: a separate chat with each of your Bees (like Grok's bots).
let mode = store.get("mnx.mode", "rush") === "peace" ? "peace" : "rush";
let rushId = store.get(ACTIVE_KEY, null);
let peaceBee = store.get("mnx.peaceBee", null);
let activeId = null;
let bees = [];
const boardState = new Map(); // Rush board: Bee name → { text, state }
if (!chats.find((c) => c.id === rushId && !c.bee)) rushId = null;
syncActive();

function syncActive() { activeId = mode === "peace" ? (peaceBee ? `peace_${peaceBee}` : null) : rushId; }

// A task still "running" after a reload keeps running on the Hive: once connected, ask where each one is now (resumeTasks).
function runningTurns() {
  const out = [];
  for (const c of chats) for (const t of c.turns) if (t.role === "mnx" && isRunning(t)) out.push(t.task);
  return out;
}
function resumeTasks() {
  const tasks = runningTurns();
  if (tasks.length) client.send({ type: "resume", tasks });
}

const runs = new Map(); // task id → { el, cardCache }
let saveTimer = null;

function writeChats() {
  clearTimeout(saveTimer);
  store.set(CHATS_KEY, chats);
  store.set(ACTIVE_KEY, rushId);
  store.set("mnx.peaceBee", peaceBee);
  store.set("mnx.mode", mode);
}
function saveChats() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(writeChats, 150);
}
// a reload or a closed tab right after a change: write it now, or that chat is lost
addEventListener("pagehide", writeChats);
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
  if (e.detail === "online") { loadTools(); loadBees(); resumeTasks(); sayFromLink(); }
});

// index.html?say=connect%20blinkit (from the Apps page): send that message once, as if typed
function sayFromLink() {
  const say = new URLSearchParams(location.search).get("say");
  if (!say) return;
  history.replaceState(null, "", location.pathname);
  setTimeout(() => send(say), 400);
}
installTokenPrompt(() => client.reconnectNow());

client.addEventListener("event", (e) => {
  const ev = e.detail;
  if (ev.type === "bee_status" || ev.type === "bees_changed") { loadBees(); return; }
  if (ev.bee) boardEvent(ev);
  if (ev.type === "task_done" || ev.type === "task_error") boardSettle();
  if (ev.type === "tools_changed" || ev.type === "phones_changed") { loadTools(); return; }
  if (ev.type === "models_changed") { refreshModelPill(); return; }
  if (!ev.task) return;
  const found = findTurn(ev.task);
  if (!found) return;
  if (ev.type === "task_state") {  // the Hive's answer to resumeTasks(): the task's whole story so far
    if (ev.unknown) {
      found.turn.events.push({ task: ev.task, type: "task_error", text: "Interrupted: the Hive restarted while this was running. Ask again." });
    } else {
      found.turn.events = ev.events;
    }
    saveChats();
    if (found.chat.id === activeId) scheduleRender(found.turn);
    return;
  }
  // A streamed reply arrives in pieces (answer_delta); once the whole answer is in, only it is kept on this device.
  if (ev.type === "answer") found.turn.events = found.turn.events.filter((e) => e.type !== "answer_delta");
  found.turn.events.push(ev);
  if (ev.type === "bee_created") loadBees();
  if (ev.type === "tokens" && found.chat.id === activeId) updateChatTokens(found.chat);
  if (ev.type === "answer" && ev.text && getSettings().speak && found.chat.id === activeId) speak(ev.text, found.chat.bee);
  found.chat.updated = Date.now();
  saveChats();
  if (found.chat.id === activeId) {
    scheduleRender(found.turn);
  }
});

installVoice(client);
client.connect();
$("conn").addEventListener("click", () => client.reconnectNow());

// ---------- sending ----------
function send(text) {
  text = text.trim();
  if (!text || runningTask()) return;

  let chat = activeChat();
  if (!chat && mode === "peace") {
    const b = bees.find((x) => x.id === peaceBee);
    if (!b) { openNewBee(); return; }
    chat = { id: `peace_${b.id}`, bee: b.id, title: b.name, created: Date.now(), updated: Date.now(), turns: [] };
    chats.unshift(chat);
  } else if (!chat) {
    chat = { id: uid("c_"), title: titleFrom(text), created: Date.now(), updated: Date.now(), turns: [] };
    chats.unshift(chat);
    rushId = chat.id;
  }
  syncActive();
  const task = uid("t_");
  chat.turns.push({ role: "user", text, at: new Date().toISOString() });
  const turn = { role: "mnx", task, events: [], bee: chat.bee || undefined };
  chat.turns.push(turn);
  chat.updated = Date.now();

  const ok = client.send({ type: "user_message", chat: chat.id, task, text, name: getSettings().name || undefined, effort: getSettings().effort,
    mode: chat.bee ? "peace" : "rush", bee: chat.bee || undefined, history: historyOf(chat),
    model: selectedModel() || undefined });
  if (!ok) {
    turn.events.push(
      { task, type: "error", text: "Can't reach the Mnx Hive gateway", detail: "check that the server is running, or set its URL in Settings" },
      { task, type: "task_error", text: "Message not sent." },
    );
  }
  saveChats();
  renderAll(true);
}

// The last turns of this chat, so the model knows what was said before.
function historyOf(chat) {
  const out = [];
  for (const t of chat.turns.slice(0, -2)) {  // everything before the message being sent now
    if (t.role === "user") out.push({ role: "user", text: t.text });
    else { const a = (t.events || []).find((e) => e.type === "answer"); if (a && a.text) out.push({ role: "bot", text: a.text }); }
  }
  return out.slice(-12);
}

function titleFrom(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 42 ? t.slice(0, 40) + "…" : t;
}

// Answers to secret boxes (passwords, OTPs, keys) go to the Hive once and are never kept in the chat on this device.
function maskSecrets(turn, ref, values) {
  const card = turn.events.find((e) => e.id === ref);
  const secret = new Set(((card && card.fields) || []).filter((f) => f.secret).map((f) => f.name));
  if (!values || !secret.size) return values;
  return Object.fromEntries(Object.entries(values).map(([k, v]) => [k, secret.has(k) ? "••••" : v]));
}

function onAction(turn) {
  return (ref, action, values, label) => {
    turn.events.push({ task: turn.task, type: "_action", ref, action, label, values: maskSecrets(turn, ref, values) });
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
function renderAll(smooth = false) {
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

  const pb = mode === "peace" ? bees.find((b) => b.id === peaceBee) : null;
  const title = pb ? pb.name : mode === "peace" ? "Peace" : chat ? chat.title : "Rush";
  $("chatTitle").textContent = title;
  document.title = `${title} · Mnx Hive`;
  for (const btn of $("modeSwitch").querySelectorAll("button")) btn.classList.toggle("on", btn.dataset.mode === mode);
  $("board").classList.toggle("hidden", mode !== "rush" || !bees.length);
  $("newChat").classList.toggle("hidden", mode === "peace");
  $("newBee").classList.toggle("hidden", mode !== "peace");
  $("historyLabel").textContent = mode === "peace" ? "Your Bees" : "Rush chats";
  input.placeholder = pb ? `Message ${pb.name}…` : mode === "peace" ? "Make a Bee first…" : bees.length ? "Ask all your Bees…" : "Ask anything…";
  $("talkBtn").title = pb ? `Talk with ${pb.name} (Fish Audio voice)` : "Talk with the Hive (Fish Audio voice)";
  renderWelcome(pb);
  renderBoard();
  updateChatTokens(chat);

  const thread = $("thread");
  thread.replaceChildren();
  runs.clear();
  if (chat) {
    for (const turn of chat.turns) {
      const fresh = turn.at && Date.now() - Date.parse(turn.at) < 1500 || (turn.task && !turn.events.length);
      if (turn.role === "user") {
        thread.append(h("div", { class: `msg-user${fresh ? " enter" : ""}` }, h("div", { class: "bubble" }, turn.text)));
      } else {
        const slot = h("div", { class: fresh ? "enter" : "" });
        thread.append(slot);
        runs.set(turn.task, { el: slot, cardCache: new Map() });
        rerenderTurn(turn, false);
      }
    }
  }
  renderHistory();
  updateSendState();
  scrollDown(true, smooth);
}

// Replies stream many events a second: draw each turn at most once per frame, so the page stays smooth.
const pendingTurns = new Set();
let frame = 0;
function scheduleRender(turn) {
  pendingTurns.add(turn);
  if (!frame) frame = requestAnimationFrame(() => {
    frame = 0;
    for (const t of pendingTurns) rerenderTurn(t);
    pendingTurns.clear();
    updateSendState();
  });
}

function rerenderTurn(turn, scroll = true) {
  const run = runs.get(turn.task);
  if (!run) return;
  const el = renderRun(buildModel(turn.events), { onAction: onAction(turn), cardCache: run.cardCache, beeNames: new Set(bees.map((b) => b.name)) });
  const owner = turn.bee && bees.find((b) => b.id === turn.bee);
  if (owner) el.querySelector(".logo")?.replaceWith(Object.assign(document.createElement("span"), { className: "logo", innerHTML: beeHex(owner.name, owner.color) }));
  run.el.replaceChildren(el);
  if (scroll) scrollDown();
}

// ---------- smooth following ----------
// "Stuck" means you're reading the bottom: the view then follows the reply as it grows (one jump per layout change,
// no flicker). Scroll up and it lets go; a "Jump to latest" button brings you back.
let stuck = true;
let followFrame = 0;
let programmatic = 0;
const NEAR = 90;

function isNearBottom() {
  const s = $("scroll");
  return s.scrollHeight - s.scrollTop - s.clientHeight < NEAR;
}
function toBottom(smooth = false) {
  const s = $("scroll");
  programmatic = performance.now() + (smooth ? 600 : 80);
  s.scrollTo({ top: s.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}
function scrollDown(force, smooth = false) {
  if (force) stuck = true;
  if (!stuck) return;
  cancelAnimationFrame(followFrame);
  followFrame = requestAnimationFrame(() => toBottom(smooth));
}
let syncFrame = 0;
function syncJumpButton() {  // judged once the scroll has settled, so it never flashes while we follow a reply
  cancelAnimationFrame(syncFrame);
  syncFrame = requestAnimationFrame(() => $("toBottom").classList.toggle("show", !isNearBottom() && !stuck));
}
// Only real input (wheel, touch, keys, dragging the scrollbar) can let go of the bottom, never our own scrolling or
// the page growing under us.
let lastInput = 0;
let touching = false;
const note = (ms) => () => { lastInput = performance.now() + ms; };
const sc = $("scroll");
sc.addEventListener("wheel", note(250), { passive: true });
sc.addEventListener("keydown", note(250));
sc.addEventListener("pointerdown", note(400), { passive: true });
sc.addEventListener("touchstart", () => { touching = true; lastInput = Infinity; }, { passive: true });
sc.addEventListener("touchend", () => { touching = false; lastInput = performance.now() + 1500; }, { passive: true });  // momentum
sc.addEventListener("touchcancel", () => { touching = false; lastInput = performance.now() + 1500; }, { passive: true });
sc.addEventListener("scroll", () => {
  if (touching || performance.now() < lastInput) stuck = isNearBottom();
  syncJumpButton();
}, { passive: true });
if (window.ResizeObserver) new ResizeObserver(() => { if (stuck) scrollDown(); syncJumpButton(); }).observe($("thread"));
$("toBottom").addEventListener("click", () => { stuck = true; toBottom(true); $("toBottom").classList.remove("show"); });

function renderHistory() {
  const box = $("history");
  box.replaceChildren();
  if (mode === "peace") { renderBeeList(box); return; }
  const sorted = chats.filter((c) => !c.bee).sort((a, b) => b.updated - a.updated);
  if (!sorted.length) {
    box.append(h("div", { class: "empty-note" }, "No chats yet"));
    return;
  }
  for (const c of sorted) {
    const del = h("button", { class: "del", "aria-label": "Delete chat", html: ICONS.trash, onclick: (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${c.title}"?`)) return;
      chats = chats.filter((x) => x.id !== c.id);
      if (rushId === c.id) rushId = null;
      syncActive();
      saveChats();
      renderAll();
    } });
    del.querySelector("svg").style.cssText = "width:15px;height:15px";
    box.append(h("button", {
      class: `rail-btn${c.id === activeId ? " active" : ""}`,
      onclick: () => { rushId = c.id; syncActive(); saveChats(); renderAll(); closeRail(); },
    }, h("span", {}, c.title), del));
  }
}

// ---------- token reading ----------
function chatTokens(chat) {
  return chat ? chat.turns.reduce((n, t) => n + (((t.events || []).find((e) => e.type === "tokens") || {}).total || 0), 0) : 0;
}
function updateChatTokens(chat) {
  const used = chatTokens(chat);
  $("chatTokens").textContent = used ? `${used.toLocaleString()} tokens` : "";
}

// ---------- effort (low · med · high · ultra · maxxxx) ----------
function fillEffortSelects() {
  $("effortPill").replaceChildren(...EFFORT.map((e) => h("option", { value: e.id }, `Effort · ${e.label}`)));
  $("setEffort").replaceChildren(...EFFORT.map((e) => h("option", { value: e.id }, `${e.label} · ${e.tokens.toLocaleString()} tokens`)));
}
function applyEffort() {
  const s = getSettings();
  const pill = $("effortPill");
  pill.value = s.effort;
  pill.classList.toggle("maxxxx", s.effort === "maxxxx");
  pill.title = `Effort: ${effortInfo(s.effort).note}`;
  updateTokenCount();
}
function updateTokenCount() {
  const info = effortInfo(getSettings().effort);
  const n = countTokens(input.value);
  const el = $("tokenCount");
  el.textContent = `~${n.toLocaleString()} / ${fmtTokens(info.tokens)} tokens`;
  el.classList.toggle("over", n > info.tokens);
}

// ---------- composer ----------
const input = $("input");
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
  updateTokenCount();
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
  rushId = null;
  syncActive();
  saveChats();
  renderAll();
  closeRail();
  input.focus();
});

// ---------- profile + settings ----------
function applyProfile() {
  const s = getSettings();
  const name = s.name.trim();
  $("avatar").textContent = name ? name.charAt(0).toUpperCase() : "?";
  $("profileName").textContent = name || "Profile";
  refreshModelPill();
}
applyProfile();

// Credentials (Brain, Voice keys, Twilio) are the owner's: users don't see them.
async function applyRole() {
  const owner = (await whoami()) === "owner";
  document.querySelectorAll(".owner-only").forEach((el) => el.classList.toggle("hidden", !owner));
  return owner;
}
client.addEventListener("state", (e) => { if (e.detail === "online") applyRole(); });

function openSettings(focusName) {
  const f = $("settingsForm");
  const s = getSettings();
  f.name.value = s.name;
  f.gateway.value = s.gateway;
  f.token.value = s.token;
  f.enterToSend.checked = s.enterToSend;
  f.effort.value = s.effort;
  $("effortHint").textContent = effortInfo(s.effort).note;
  f.speak.checked = !!s.speak;
  applyRole().then((owner) => { if (owner) { loadVoice(); loadBrain(); } });
  $("settingsDlg").showModal();
  if (focusName) f.name.focus();
}
$("openSettings").addEventListener("click", () => openSettings(false));
$("openProfile").addEventListener("click", () => openSettings(true));
installModelPicker();
$("settingsDlg").addEventListener("close", () => {
  if ($("settingsDlg").returnValue !== "save") return;
  const f = $("settingsForm");
  const prev = getSettings();
  saveSettings({
    token: f.token.value.trim(),
    name: f.name.value.trim(),
    gateway: f.gateway.value.trim(),
    enterToSend: f.enterToSend.checked,
    effort: f.effort.value,
    speak: f.speak.checked,
  });
  if (voiceSeen) saveVoice();
  if ($("bUrl").value !== undefined && !$("bUrl").closest(".owner-only").classList.contains("hidden")) saveBrain();
  applyProfile();
  applyEffort();
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
  rushId = null;
  syncActive();
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
  try { bees = await api("/api/bees"); } catch { /* offline */ }
  if (mode === "peace" && !bees.find((b) => b.id === peaceBee)) { peaceBee = bees.length ? bees[0].id : null; syncActive(); }
  renderAll();
  const list = $("beeList");
  list.replaceChildren();
  $("beeCount").textContent = bees.length || "";
  if (!bees.length) list.append(h("div", { class: "empty-note" }, "No Bees yet. Make one in Peace."));
  for (const b of bees) {
    const row = h("a", { class: "bee-row", href: `cell.html?bee=${encodeURIComponent(b.id)}`, title: `Open ${b.name}'s computer` });
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
  else location.href = "bots.html";
});

// ---------- mobile rail ----------
function closeRail() { $("app").classList.remove("rail-open"); }
$("menuBtn").addEventListener("click", () => $("app").classList.add("rail-open"));
$("scrim").addEventListener("click", closeRail);

renderAll();
loadTools();
loadBees();

fillEffortSelects();
applyEffort();
$("effortPill").addEventListener("change", () => { saveSettings({ effort: $("effortPill").value }); applyEffort(); });
$("setEffort").addEventListener("change", () => { $("effortHint").textContent = effortInfo($("setEffort").value).note; });


// ---------- Rush · Peace ----------
for (const btn of $("modeSwitch").querySelectorAll("button")) {
  btn.addEventListener("click", () => {
    if (mode === btn.dataset.mode) return;
    mode = btn.dataset.mode;
    if (mode === "peace" && !peaceBee && bees.length) peaceBee = bees[0].id;
    syncActive();
    saveChats();
    renderAll();
  });
}

// Peace: each of your Bees in the rail, with the last thing said in its chat.
function renderBeeList(box) {
  if (!bees.length) { box.append(h("div", { class: "empty-note" }, "No Bees yet")); return; }
  const last = (b) => (chats.find((c) => c.id === `peace_${b.id}`) || {}).updated || 0;
  for (const b of [...bees].sort((x, y) => last(y) - last(x))) {
    const chat = chats.find((c) => c.id === `peace_${b.id}`);
    const last = chat && [...chat.turns].reverse().find((t) => t.role === "user" || (t.events || []).some((e) => e.type === "answer"));
    const preview = !last ? b.skill : last.role === "user" ? `You: ${last.text}` : (last.events.find((e) => e.type === "answer") || {}).text || "";
    const icon = h("span", { class: "bee-mini", html: beeHex(b.name, b.color) });
    box.append(h("button", {
      class: `rail-btn${b.id === peaceBee ? " active" : ""}`, title: `${b.name}: ${b.skill}`,
      onclick: () => { peaceBee = b.id; syncActive(); saveChats(); renderAll(); closeRail(); input.focus(); },
    }, icon, h("span", { class: "two" }, h("span", {}, b.name), h("span", { class: "sub" }, preview || ""))));
  }
}

function renderWelcome(pb) {
  const head = $("peaceHead");
  const empty = mode === "peace" && !pb;  // Peace with no Bees yet: make the first one
  head.classList.toggle("hidden", !pb && !empty);
  $("welcomeLogo").classList.toggle("hidden", !!pb);
  $("wordmark").classList.toggle("hidden", !!pb || empty);
  $("greeting").classList.toggle("hidden", !!pb);
  const name = getSettings().name.trim();
  if (empty) {
    $("greeting").textContent = "Peace: a separate chat with each of your Bees.";
    head.replaceChildren(h("div", { class: "skill" }, "Make a Bee and give it a responsibility, like “Scout: watches prices and tells me when they drop”."),
      h("div", { class: "starters" }, h("button", { type: "button", onclick: () => openNewBee() }, "Make your first Bee")));
    return;
  }
  if (!pb) {
    $("greeting").textContent = bees.length ? `${name ? `Hi ${name}` : "Hi there"}, Rush: your ${bees.length} Bee${bees.length > 1 ? "s" : ""} work on what you ask, together.`
      : `${name ? `Hi ${name}` : "Hi there"}, what can I help with?`;
    return;
  }
  head.replaceChildren();
  head.insertAdjacentHTML("beforeend", beeHex(pb.name, pb.color));
  const p = pb.bot || {};
  head.append(h("h2", { class: "wordmark" }, pb.name), h("div", { class: "skill" }, pb.skill),
    p.greeting ? h("div", { class: "greet" }, p.greeting) : "",
    (p.starters || []).length ? h("div", { class: "starters" }, ...p.starters.map((t) => h("button", { type: "button", onclick: () => send(t) }, t))) : "",
    h("div", { class: "comp" }, h("a", { href: `bots.html#edit=${encodeURIComponent(pb.id)}` }, "Edit"), "  ·  ",
      h("a", { href: `cell.html?bee=${encodeURIComponent(pb.id)}` }, "Its computer"), `  ·  ${pb.computer_text || ""}`));
}

// Rush board: one tile per Bee of yours, showing what it's doing right now.
function renderBoard() {
  const board = $("board");
  if (mode !== "rush") return;
  const names = bees.map((b) => b.name);
  board.replaceChildren(...names.map((name) => {
    const b = bees.find((x) => x.name === name);
    const st = boardState.get(name) || { text: b && b.status !== "idle" ? b.status : "Ready", state: b && b.status === "busy" ? "busy" : "" };
    const tile = h("div", { class: `tile ${st.state}`, "data-bee": name, title: st.text },
      h("div", { class: "top", html: beeHex(name, b && b.color) }, h("span", {}, name)),
      h("div", { class: "what" }, st.text));
    return tile;
  }));
}

function boardEvent(ev) {
  if (mode !== "rush" || !bees.some((b) => b.name === ev.bee)) return;  // the Hive's own functions aren't Bees
  const prev = boardState.get(ev.bee) || {};
  let state = prev.state || "";
  if (["step", "planning", "searching", "reading", "running", "opening", "fetching", "writing", "waiting", "thinking"].includes(ev.type)) {
    state = ev.status === "error" ? "error" : ev.status === "done" ? "done" : "busy";
  } else if (ev.type === "output") state = ev.code ? "error" : "done";
  else if (ev.type === "result" || ev.type === "done") state = "done";
  else if (ev.type === "error") state = "error";
  const text = ev.type === "output" ? `$ ${ev.command} → exit ${ev.code}` : ev.type === "result" ? (ev.title || "Reported") : (ev.text || prev.text || "");
  boardState.set(ev.bee, { text, state });
  const tile = $("board").querySelector(`[data-bee="${CSS.escape(ev.bee)}"]`);
  if (tile) { tile.className = `tile ${state}`; tile.title = text; tile.querySelector(".what").textContent = text; }
  else renderBoard();
}

function boardSettle() {
  for (const [name, st] of boardState) if (st.state === "busy") boardState.set(name, { ...st, state: "done" });
  renderBoard();
}

// ---------- voice ----------
$("talkBtn").addEventListener("click", () => { stopSpeaking(); startTalk(mode === "peace" ? peaceBee : null); });

// ---------- New Bee: a name and a responsibility, then straight into its own chat ----------
let planView = null;
async function loadPlan() {
  try { planView = await api("/api/plan"); } catch { return; }
  $("proLabel").textContent = planView.plan === "pro" ? "Pro" : "Get Pro";
  $("proBtn").classList.toggle("is-pro", planView.plan === "pro");
}
function openNewBee() {
  const f = $("beeForm");
  f.reset();
  $("beeErr").textContent = "";
  $("beeNote").textContent = planView ? `It gets its own chat, memory and computer (${planView.computer}). ${bees.length} of ${planView.max_bees} Bees on ${planView.label}.` : "";
  $("beeDlg").showModal();
  closeRail();
  f.name.focus();
}
$("newBee").addEventListener("click", openNewBee);
$("beeForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "make") return;
  e.preventDefault();
  const f = e.target;
  $("beeErr").textContent = "Making it…";
  try {
    const b = await api("/api/bots", { method: "POST", body: { name: f.name.value.trim(), tagline: f.tagline.value.trim(),
      profile: { personality: f.personality.value.trim() } } });
    $("beeDlg").close();
    mode = "peace";
    peaceBee = b.id;
    syncActive();
    saveChats();
    await loadBees();
    input.focus();
  } catch (err) {
    $("beeErr").textContent = err.message;
  }
});
client.addEventListener("state", (e) => { if (e.detail === "online") loadPlan(); });
client.addEventListener("event", (e) => { if (e.detail.type === "plan_changed") loadPlan(); });
loadPlan();

// Settings → Voice: kept on the server (the key never comes back to the page).
let voiceSeen = null;
const VOICE_FIELDS = { vFishKey: "fish_key", vModel: "model", vLang: "language", vPhone: "phone", vSid: "twilio_sid", vTok: "twilio_token", vFrom: "twilio_from", vTz: "timezone" };
async function loadVoice() {
  let v;
  try { v = await api("/api/voice"); } catch { $("vHint").textContent = "The gateway is offline, so voice settings can't load."; return; }
  voiceSeen = v;
  $("vModel").replaceChildren(...v.models.map((m) => h("option", { value: m }, m)));
  for (const [id, key] of Object.entries(VOICE_FIELDS)) {
    const el = $(id);
    if (key === "fish_key") el.value = "", el.placeholder = v.fish_key_set ? "Saved · type a new key to replace it" : "From fish.audio → API keys";
    else if (key === "twilio_token") el.value = "", el.placeholder = v.twilio_token_set ? "Saved" : "";
    else el.value = v[key] ?? "";
  }
  const sel = $("vVoice");
  sel.replaceChildren(h("option", { value: "" }, "Fish Audio's default voice"));
  if (v.voice_id) sel.append(h("option", { value: v.voice_id, "data-name": v.voice_name || v.voice_id }, v.voice_name || v.voice_id));
  sel.value = v.voice_id || "";
  $("vHint").textContent = v.fish_ready ? "Replies and calls use the Fish Audio voice." : "No Fish Audio key yet: the browser's own voice stands in.";
  $("vPhoneHint").textContent = v.twilio_ready ? `Calls ring your phone ${v.phone}.` : `Phone calls ${v.phone_calls}. Until then, calls ring in the Hive app on every open page.`;
}
async function saveVoice() {
  if (!voiceSeen) return;
  const change = {};
  for (const [id, key] of Object.entries(VOICE_FIELDS)) {
    const val = $(id).value.trim();
    if (key === "fish_key" || key === "twilio_token") { if (val) change[key] = val; continue; }
    if (val !== String(voiceSeen[key] ?? "")) change[key] = val;
  }
  const sel = $("vVoice");
  if (sel.value !== (voiceSeen.voice_id || "")) {
    change.voice_id = sel.value;
    change.voice_name = sel.value ? (sel.selectedOptions[0]?.dataset.name || sel.selectedOptions[0]?.textContent || "") : "";
  }
  if (!Object.keys(change).length) return;
  try { await api("/api/voice", { method: "PUT", body: change }); toast("Voice settings saved"); askNotifications(); }
  catch (e) { toast(`Voice: ${e.message}`, "error"); }
}
$("vFind").addEventListener("click", async () => {
  const keyTyped = $("vFishKey").value.trim();
  if (keyTyped) await api("/api/voice", { method: "PUT", body: { fish_key: keyTyped } }).catch(() => {});
  try {
    const list = await api(`/api/voice/voices?q=${encodeURIComponent($("vSearch").value.trim())}`);
    const sel = $("vVoice");
    const keep = sel.value;
    sel.replaceChildren(h("option", { value: "" }, "Fish Audio's default voice"),
      ...list.map((x) => h("option", { value: x.id, "data-name": x.title }, `${x.title}${x.languages.length ? ` · ${x.languages.join(", ")}` : ""}`)));
    sel.value = list[0] ? list[0].id : keep;
    $("vHint").textContent = list.length ? `${list.length} voices found; pick one and press Save.` : "No voices matched.";
  } catch (e) { $("vHint").textContent = `Couldn't search: ${e.message}`; }
});
$("vTest").addEventListener("click", async () => {
  await saveVoice();
  voiceSeen = null;
  await loadVoice();
  speak(`Hi${getSettings().name ? " " + getSettings().name : ""}, this is your Hive speaking with the Fish Audio voice.`);
});

// Settings → Brain: the model the Hive and your bots think with.
async function loadBrain() {
  try {
    const b = await api("/api/brain");
    $("bUrl").value = b.url || "";
    $("bModel").value = b.model || "";
    $("bKey").value = "";
    $("bKey").placeholder = b.key_set ? "Saved · type a new key to replace it" : "Only if the server needs one";
    $("bPreset").replaceChildren(h("option", { value: "" }, "Pick a free option…"), ...b.presets.map((p) => h("option", { value: p.id }, p.label)));
    $("bPreset").onchange = () => { const p = b.presets.find((x) => x.id === $("bPreset").value); if (p) { $("bUrl").value = p.url; $("bModel").value = p.model; } };
    $("bHint").textContent = b.connected ? `Connected: ${b.model || ""} at ${b.url}` : "No brain yet: bots answer from their knowledge and memory until you connect one.";
  } catch { /* offline */ }
}
async function saveBrain() {
  const body = { url: $("bUrl").value.trim() || "-", model: $("bModel").value.trim() || "-" };
  if ($("bKey").value.trim()) body.key = $("bKey").value.trim();
  try { await api("/api/brain", { method: "PUT", body }); } catch (e) { toast(`Brain: ${e.message}`, "error"); }
}
$("bTest").addEventListener("click", async () => {
  await saveBrain();
  $("bHint").textContent = "Asking the brain…";
  try { const r = await api("/api/brain/test", { method: "POST" }); $("bHint").textContent = `It answered: “${r.reply}”`; }
  catch (e) { $("bHint").textContent = `No answer: ${e.message}`; }
});

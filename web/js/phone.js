// Mnx Hive — Phone Drone: see and control your paired Android phone.

import { HiveClient, api, connLabel, fillIcons, gatewayBase, getSettings, h, installTokenPrompt, toast } from "./shared.js";

const $ = (id) => document.getElementById(id);
fillIcons();

let phones = [];
let phone = null; // selected phone view
let size = null; // real screen size {width, height}
let busy = 0;
let apps = [];
let battery = null;

const can = (cmd) => !!phone && phone.capabilities.includes(cmd);

async function cmd(name, args = {}, { quiet = false } = {}) {
  if (!phone) return null;
  if (phone.risky.includes(name)) {
    const what = name === "sms_send" ? `Send this SMS to ${args.number}?\n\n${args.text}` : name === "call" ? `Call ${args.number}?` : `Run ${name}?`;
    if (!confirm(what)) return null;
  }
  busy++;
  if (!quiet) $("busy").classList.remove("hidden");
  $("err").textContent = "";
  try {
    const r = await api(`/api/phones/${encodeURIComponent(phone.id)}/cmd`, {
      method: "POST", body: { cmd: name, args, confirmed: phone.risky.includes(name) },
    });
    return r.result;
  } catch (e) {
    if (!quiet) $("err").textContent = e.message;
    throw e;
  } finally {
    if (--busy === 0) $("busy").classList.add("hidden");
  }
}

// ---------- phones ----------
async function loadPhones() {
  try {
    phones = await api("/api/phones");
    connLabel($("conn"), "online");
  } catch (e) {
    connLabel($("conn"), e.status === 401 ? "auth" : "offline");
    phones = [];
  }
  const pick = $("phonePick");
  pick.replaceChildren(...phones.map((p) => h("option", { value: p.id }, p.name || p.model || p.id)));
  pick.classList.toggle("hidden", phones.length < 2);
  const keep = phone && phones.find((p) => p.id === phone.id);
  select(keep || phones[0] || null);
}
$("phonePick").addEventListener("change", (e) => select(phones.find((p) => p.id === e.target.value)));

function select(p) {
  const changed = (p && p.id) !== (phone && phone.id);
  phone = p;
  $("setup").classList.toggle("hidden", !!p);
  $("control").classList.toggle("hidden", !p);
  if (!p) { renderSetup(); return; }
  $("phonePick").value = p.id;

  const info = $("info");
  info.replaceChildren(
    h("span", { class: "chip" }, p.name || "Phone"),
    p.model ? h("span", { class: "chip" }, p.model) : null,
    p.android ? h("span", { class: "chip" }, `Android ${p.android}`) : null,
    h("span", { class: `chip ${p.backend === "none" ? "ask" : ""}` }, p.backend === "none" ? "No screen control" : `Screen: ${p.backend}`),
    can("battery") ? h("span", { class: "chip", id: "batt" }, "Battery …") : null);
  if (battery) showBattery(battery);

  const screen = can("screenshot");
  $("noScreen").classList.toggle("hidden", screen);
  $("shot").classList.toggle("hidden", !screen);
  for (const b of document.querySelectorAll("[data-key]")) b.disabled = !can("key");
  $("typeForm").querySelector("button[type=submit]").disabled = !can("text");
  for (const b of document.querySelectorAll("[data-cmd]")) b.disabled = !can(b.dataset.cmd);
  $("smsForm").querySelector("button[type=submit]").disabled = !can("sms_send");
  $("callBtn").disabled = !can("call");

  if (changed) {
    size = null;
    if (screen) shot();
    if (can("apps")) loadApps();
    else $("apps").replaceChildren(h("p", { class: "hint" }, "Listing apps needs screen control (ADB or Shizuku)."));
    battery = null;
    if (can("battery")) cmd("battery", {}, { quiet: true }).then(showBattery).catch(() => {});
  }
}

function showBattery(b) {
  if (b && typeof b === "object") battery = b;
  const el = $("batt");
  if (el && b && typeof b === "object") el.textContent = `Battery ${b.percentage}%${b.status === "CHARGING" ? " ⚡" : ""}`;
}

function renderSetup() {
  const base = gatewayBase();
  $("droneUrl").textContent = `${base}/drone.py`;
  $("runCmd").textContent = `python drone.py --gateway ${base} --token <your token> --name "My phone"`;
}
$("copyRun").addEventListener("click", async () => {
  const text = `python drone.py --gateway ${gatewayBase()} --token ${getSettings().token} --name "My phone"`;
  try { await navigator.clipboard.writeText(text); $("copyRun").textContent = "Copied"; }
  catch { prompt("Copy this command:", text); }
});

// ---------- screen ----------
async function shot(quiet = false) {
  if (!can("screenshot")) return;
  try {
    const r = await cmd("screenshot", { width: 540 }, { quiet });
    if (!r) return;
    size = { width: r.width, height: r.height };
    $("shot").src = r.image;
  } catch { /* shown in #err */ }
}
const after = (p) => p.then(() => setTimeout(() => shot(true), 600)).catch(() => {});

function toPhone(e) {
  const r = $("shot").getBoundingClientRect();
  return {
    x: Math.round(((e.clientX - r.left) / r.width) * (size?.width || 1080)),
    y: Math.round(((e.clientY - r.top) / r.height) * (size?.height || 2400)),
  };
}

let down = null;
$("shot").addEventListener("pointerdown", (e) => {
  if (!size) return;
  e.preventDefault();
  down = { ...toPhone(e), t: Date.now() };
  $("shot").setPointerCapture(e.pointerId);
});
$("shot").addEventListener("pointerup", (e) => {
  if (!down) return;
  const up = toPhone(e);
  const dist = Math.hypot(up.x - down.x, up.y - down.y);
  const held = Date.now() - down.t;
  if (dist > 40) after(cmd("swipe", { x1: down.x, y1: down.y, x2: up.x, y2: up.y, ms: Math.min(Math.max(held, 150), 1500) }));
  else if (held > 600) after(cmd("long_press", { x: down.x, y: down.y }));
  else after(cmd("tap", { x: down.x, y: down.y }));
  down = null;
});

document.querySelectorAll("[data-key]").forEach((b) => b.addEventListener("click", () => after(cmd("key", { key: b.dataset.key }))));
$("typeForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("typeText").value;
  if (!text) return;
  $("typeText").value = "";
  after(cmd("text", { text }));
});

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  for (const name of ["apps", "screen", "device", "messages", "ai"]) $(`tab-${name}`).classList.toggle("hidden", t.dataset.tab !== name);
  if (t.dataset.tab === "ai") loadAi();
}));

// Apps
async function loadApps() {
  $("apps").replaceChildren(h("p", { class: "hint" }, "Loading apps…"));
  try { apps = (await cmd("apps", {}, { quiet: true })) || []; } catch { apps = []; }
  renderApps();
}
function renderApps() {
  const q = $("appFilter").value.trim().toLowerCase();
  const list = apps.filter((a) => !q || a.label.toLowerCase().includes(q) || a.package.toLowerCase().includes(q));
  const box = $("apps");
  box.replaceChildren();
  if (!list.length) box.append(h("p", { class: "hint" }, apps.length ? "No match." : "No apps found."));
  for (const a of list.slice(0, 200)) {
    box.append(h("button", { class: "el-row", type: "button", title: a.package, onclick: () => after(cmd("open_app", { package: a.package })) },
      h("span", { class: "app-ico" }, a.label.charAt(0)),
      h("span", { class: "meta" }, h("span", { class: "name" }, a.label), h("span", { class: "skill" }, a.package))));
  }
}
$("appFilter").addEventListener("input", renderApps);

// On screen (accessibility tree)
$("loadTree").addEventListener("click", async () => {
  const box = $("tree");
  box.replaceChildren(h("p", { class: "hint" }, "Reading the screen…"));
  let nodes;
  try { nodes = await cmd("ui_tree"); } catch { box.replaceChildren(); return; }
  box.replaceChildren();
  if (!nodes?.length) box.append(h("p", { class: "hint" }, "Nothing readable on screen."));
  for (const n of nodes || []) {
    const label = n.text || n.desc || n.id || n.cls;
    box.append(h("button", { class: "el-row", type: "button", onclick: () => after(cmd("tap", { x: n.x, y: n.y })) },
      h("span", { class: `chip${n.clickable ? " ask" : ""}` }, n.editable ? "field" : n.clickable ? "button" : "text"),
      h("span", { class: "ellipsis" }, label)));
  }
});
$("tapTextForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("tapText").value.trim();
  if (text) after(cmd("tap_text", { text }));
});

// Device
function out(id, value) {
  const el = $(id);
  if (el.tagName === "PRE") el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  else renderMessages(el, value);
}
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", async () => {
  const args = b.dataset.args ? JSON.parse(b.dataset.args) : {};
  const target = b.dataset.out || "devOut";
  try {
    const r = await cmd(b.dataset.cmd, args);
    if (r !== null) out(target, r);
    if (b.dataset.cmd === "battery") showBattery(r);
  } catch (e) { out(target, e.message); }
}));
$("sayForm").addEventListener("submit", (e) => { e.preventDefault(); const t = $("sayText").value.trim(); if (t) cmd("tts", { text: t }).catch(() => {}); });
$("toastForm").addEventListener("submit", (e) => { e.preventDefault(); const t = $("toastText").value.trim(); if (t) cmd("toast", { text: t }).catch(() => {}); });
$("volForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try { out("devOut", await cmd("volume", { stream: $("volStream").value, level: Number($("volLevel").value) })); } catch { /* shown */ }
});

// Messages
function renderMessages(box, items) {
  box.replaceChildren();
  if (!Array.isArray(items) || !items.length) { box.append(h("p", { class: "hint" }, typeof items === "string" ? items : "Nothing here.")); return; }
  for (const m of items) {
    const title = m.title || m.number || m.sender || m.packageName || "";
    const body = m.content || m.body || m.text || "";
    box.append(h("div", { class: "msg-item" },
      h("div", { class: "name" }, title, m.received ? h("span", { class: "hint" }, ` · ${m.received}`) : null, m.when ? h("span", { class: "hint" }, ` · ${m.when}`) : null),
      h("div", { class: "skill" }, body)));
  }
}
$("smsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const number = $("smsNumber").value.trim(), text = $("smsText").value.trim();
  if (!number || !text) return;
  try { if (await cmd("sms_send", { number, text })) { $("smsText").value = ""; toast("Sent."); } } catch { /* shown */ }
});
$("callBtn").addEventListener("click", () => {
  const number = $("smsNumber").value.trim();
  if (number) cmd("call", { number }).catch(() => {});
});

// ---------- live ----------
setInterval(() => {
  if ($("live").checked && phone && can("screenshot") && !busy && document.visibilityState === "visible") shot(true);
}, 3000);

const client = new HiveClient();
client.addEventListener("state", (e) => { connLabel($("conn"), e.detail); if (e.detail === "online") loadPhones(); });
client.addEventListener("event", (e) => { if (e.detail.type === "phones_changed") loadPhones(); });
client.connect();
installTokenPrompt(() => { client.reconnectNow(); loadPhones(); });
loadPhones();

// ---------- AI on this phone: layer-by-layer streaming ----------
let aiJob = null;
let aiTimer = null;
async function loadAi() {
  $("aiNeeds").classList.toggle("hidden", can("stream_start"));
  let models = [];
  try { models = (await api("/api/lab")).models || []; } catch { models = []; }
  models = models.filter((m) => m.kind === "llm" && m.result);
  const sel = $("aiModel");
  const prev = sel.value;
  sel.replaceChildren(...models.map((m) => h("option", { value: m.id }, m.name)));
  if (!models.length) sel.append(h("option", { value: "" }, "No trained model yet (create one in the chat)"));
  if (prev) sel.value = prev;
  showAiPlan();
}
async function showAiPlan() {
  const mid = $("aiModel").value;
  if (!mid || !phone) { $("aiPlan").textContent = ""; return; }
  try {
    const q = new URLSearchParams({ ram_gb: phone.ram_gb || 8, cores: phone.cores || 8 });
    if (phone.storage_free_gb) q.set("storage_free_gb", phone.storage_free_gb);
    const p = await api(`/api/lab/models/${encodeURIComponent(mid)}/layers/plan?${q}`);
    $("aiPlan").textContent = p.text;
    $("aiPlan").classList.toggle("err", !p.fits);
  } catch (e) { $("aiPlan").textContent = e.message; }
}
$("aiModel").addEventListener("change", showAiPlan);
async function aiStart(mode, extra) {
  const mid = $("aiModel").value;
  if (!mid) return toast("Pick a model first", "error");
  try {
    const r = await cmd("stream_start", { mode, model: mid, ...extra });
    aiJob = r.job;
    $("aiOut").textContent = `Started ${mode} on the phone (job ${r.job})…`;
    clearInterval(aiTimer);
    aiTimer = setInterval(aiPoll, 3000);
  } catch { /* shown */ }
}
async function aiPoll() {
  if (!aiJob) return;
  let st;
  try { st = await cmd("stream_status", { job: aiJob }, { quiet: true }); } catch { return; }
  const m = st.metric || {};
  const line = st.mode === "run"
    ? (m.token ? `token ${m.token} · ${m.layers} layers streamed in ${m.seconds} s · ${m.fetched_mb} MB fetched` : "fetching the first layer…")
    : (m.step ? `step ${m.step} · loss ${m.loss} · ${m.seconds} s per step · ${m.sent_mb} MB sent back` : "fetching the first layer…");
  let out = `${st.running ? "Running" : `Finished (exit ${st.exit_code})`} · ${line}\n\n${st.tail || ""}`;
  if (st.result && st.mode === "run") out += `\n\n→ ${st.result.text}`;
  if (st.result && st.mode === "train") out += `\n\nTrained ${st.result.steps} steps, loss ${st.result.loss}; layers merged back into the Hive's model.`;
  $("aiOut").textContent = out;
  if (!st.running) { clearInterval(aiTimer); aiTimer = null; }
}
$("aiRunForm").addEventListener("submit", (e) => { e.preventDefault(); aiStart("run", { prompt: $("aiPrompt").value.trim() || "Once upon a time", max_new: 40 }); });
$("aiTrainForm").addEventListener("submit", (e) => { e.preventDefault(); aiStart("train", { steps: Number($("aiSteps").value) || 5 }); });
$("aiStop").addEventListener("click", async () => { if (aiJob) { try { await cmd("stream_stop", { job: aiJob }); } catch { /* shown */ } aiPoll(); } });

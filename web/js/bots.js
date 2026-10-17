// Mnx Hive — Your Bees: make your own (name, responsibility, instructions, knowledge, memory, voice), each with its
// own chat in Peace. No premade Bees. Shared: Bees other people on this Hive let you remix.

import { ICONS, HiveClient, api, beeHex, connLabel, gatewayBase, getSettings, h, installTokenPrompt, store, toast } from "./shared.js";
import { installVoice } from "./voice.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

let BOTS = [];
let SHARED = [];
let META = { tones: {} };
let PLAN = null;
let editing = null;     // the Bee being edited (null: a new one)
let fromIdea = null;    // what a remix starts from
let pendingFiles = [];  // knowledge picked before a new Bee exists

// ---------- tabs ----------
for (const t of document.querySelectorAll(".bot-tab")) {
  t.addEventListener("click", () => {
    for (const x of document.querySelectorAll(".bot-tab")) x.classList.toggle("on", x === t);
    $("tab-mine").classList.toggle("hidden", t.dataset.tab !== "mine");
    $("tab-explore").classList.toggle("hidden", t.dataset.tab !== "explore");
  });
}

// ---------- loading ----------
async function load() {
  try {
    [BOTS, META, SHARED, PLAN] = await Promise.all([api("/api/bots"), api("/api/bots/templates"), api("/api/bots/shared"), api("/api/plan")]);
  } catch { $("mine").replaceChildren(h("p", { class: "hint" }, "Couldn't reach the Hive.")); return; }
  $("planNote").replaceChildren(
    `${BOTS.length} of ${PLAN.max_bees} Bees on ${PLAN.label}. Each gets its own chat and a computer with ${PLAN.computer}. `,
    PLAN.plan === "pro" ? "" : h("a", { href: "pro.html" }, `Get Pro for ${PLAN.plans.find((p) => p.id === "pro").computer} and up to ${PLAN.plans.find((p) => p.id === "pro").max_bees} Bees →`));
  render();
  const m = location.hash.match(/edit=([a-z0-9_]+)/);
  if (m) { const b = BOTS.find((x) => x.id === m[1]); if (b) openEditor(b); history.replaceState(null, "", "bots.html"); }
  if (location.hash === "#new") { history.replaceState(null, "", "bots.html"); openEditor(null); }
}

function card(b, line, buttons, extra) {
  return h("div", { class: "bot-card" },
    h("div", { class: "bot-top" }, h("span", { class: "bot-hex", html: beeHex(b.name, b.color) }), h("div", {}, h("b", {}, b.name), h("div", { class: "hint" }, line || ""))),
    extra || null,
    h("div", { class: "bot-actions" }, ...buttons));
}

function chatWith(b) {
  store.set("mnx.mode", "peace");
  store.set("mnx.peaceBee", b.id);
  location.href = "index.html";
}

function render() {
  const mine = $("mine");
  mine.replaceChildren();
  if (!BOTS.length) mine.append(h("div", { class: "usage-card" }, h("b", {}, "No Bees yet."),
    h("p", { class: "hint" }, "Give a Bee a name and a responsibility, like “Scout: watches prices and tells me when they drop”. It gets its own chat."),
    h("button", { class: "btn primary", type: "button", onclick: () => openEditor(null) }, "Make your first Bee")));
  for (const b of BOTS) {
    const p = b.bot;
    const facts = [p.tone, `${b.knowledge.length} knowledge file${b.knowledge.length === 1 ? "" : "s"}`, `${b.memories} memories`,
      p.voice_name ? `voice: ${p.voice_name}` : null, p.visibility === "hive" ? "shared" : null,
      b.parent_name && b.parent_name !== "you" ? `made by ${b.parent_name}` : null].filter(Boolean);
    mine.append(card(b, b.skill, [
      h("button", { class: "btn primary", type: "button", onclick: () => chatWith(b) }, "Chat"),
      h("button", { class: "btn", type: "button", onclick: () => openEditor(b) }, "Edit"),
      h("a", { class: "btn", href: `cell.html?bee=${encodeURIComponent(b.id)}` }, "Computer"),
      h("button", { class: "btn", type: "button", onclick: () => exportBot(b) }, "Export"),
      h("button", { class: "btn danger", type: "button", onclick: () => removeBot(b) }, "Delete"),
    ], h("div", { class: "bot-facts" }, ...facts.map((f) => h("span", { class: "chip" }, f)))));
  }
  $("shared").replaceChildren(...(SHARED.length ? SHARED.map((b) => card(b, b.skill, [
    h("button", { class: "btn primary", type: "button", onclick: () => remix(b) }, "Remix"),
  ], b.starters.length ? h("div", { class: "hint idea-starters" }, b.starters.slice(0, 2).map((x) => `“${x}”`).join("  ")) : null))
    : [h("p", { class: "hint" }, "Nobody has shared a Bee yet. Set a Bee's Sharing to “Others on this Hive can remix it” to share yours.")]));
}

// ---------- the editor ----------
function openEditor(b) {
  editing = b;
  pendingFiles = [];
  const f = $("botForm");
  f.reset();
  $("botErr").textContent = "";
  $("toneSel").replaceChildren(...Object.entries(META.tones).map(([k, v]) => h("option", { value: k, title: v }, k[0].toUpperCase() + k.slice(1))));
  const src = b ? { ...b.bot, name: b.name, tagline: b.skill } : fromIdea ? { ...fromIdea } : {};
  f.name.value = src.name || "";
  f.tagline.value = src.tagline || "";
  f.personality.value = src.personality || "";
  f.tone.value = src.tone || "friendly";
  f.visibility.value = src.visibility || "private";
  f.greeting.value = src.greeting || "";
  f.starters.value = (src.starters || []).join("\n");
  f.memory.checked = src.memory !== false;
  f.computer.checked = src.computer !== false;
  const sel = $("bvVoice");
  sel.replaceChildren(h("option", { value: "" }, "The Hive's voice"));
  if (src.voice_id) sel.append(h("option", { value: src.voice_id, "data-name": src.voice_name || src.voice_id }, src.voice_name || src.voice_id));
  sel.value = src.voice_id || "";
  renderKnowledge();
  $("botTitle").textContent = b ? `Edit ${b.name}` : "New Bee";
  $("compNote").textContent = PLAN ? `Its computer: ${PLAN.computer} (your ${PLAN.label} plan; nobody picks sizes).` : "";
  $("botDlg").showModal();
}

function renderKnowledge() {
  const list = $("knowList");
  list.replaceChildren();
  const files = editing ? editing.knowledge : pendingFiles.map((f) => ({ name: f.name, size: f.size }));
  for (const k of files) {
    list.append(h("div", { class: "know-row" }, h("span", {}, k.name), h("span", { class: "hint" }, `${Math.ceil(k.size / 1024)} KB`),
      h("button", { class: "btn danger", type: "button", onclick: async () => {
        if (editing) { await api(`/api/bots/${editing.id}/knowledge/${encodeURIComponent(k.name)}`, { method: "DELETE" }); editing.knowledge = editing.knowledge.filter((x) => x.name !== k.name); }
        else pendingFiles = pendingFiles.filter((x) => x.name !== k.name);
        renderKnowledge();
      } }, "Remove")));
  }
  if (!files.length) list.append(h("span", { class: "hint" }, "None yet."));
}

async function upload(botId, file) {
  const res = await fetch(`${gatewayBase()}/api/bots/${botId}/knowledge?name=${encodeURIComponent(file.name)}`, {
    method: "POST", body: file, headers: { Authorization: `Bearer ${getSettings().token}` } });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
  return res.json();
}

$("knowFile").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  if (!editing) { pendingFiles.push(...files); renderKnowledge(); return; }
  for (const file of files) {
    try { editing.knowledge.push(await upload(editing.id, file)); } catch (err) { toast(`${file.name}: ${err.message}`, "error"); }
  }
  renderKnowledge();
});

$("bvFind").addEventListener("click", async () => {
  try {
    const list = await api(`/api/voice/voices?q=${encodeURIComponent($("bvSearch").value.trim())}`);
    const sel = $("bvVoice");
    sel.replaceChildren(h("option", { value: "" }, "The Hive's voice"),
      ...list.map((x) => h("option", { value: x.id, "data-name": x.title }, `${x.title}${x.languages.length ? ` · ${x.languages.join(", ")}` : ""}`)));
    if (list[0]) sel.value = list[0].id;
  } catch (e) { toast(`Voices: ${e.message}`, "error"); }
});

$("botForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  const f = e.target;
  const sel = $("bvVoice");
  const profile = {
    personality: f.personality.value.trim(), tone: f.tone.value, visibility: f.visibility.value,
    greeting: f.greeting.value.trim(), starters: f.starters.value.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 4),
    memory: f.memory.checked, computer: f.computer.checked, voice_id: sel.value,
    voice_name: sel.value ? (sel.selectedOptions[0]?.dataset.name || sel.selectedOptions[0]?.textContent || "") : "",
  };
  $("botErr").textContent = "Saving…";
  try {
    if (editing) {
      await api(`/api/bots/${editing.id}`, { method: "PUT", body: { name: f.name.value.trim(), tagline: f.tagline.value.trim(), profile } });
    } else {
      const b = await api("/api/bots", { method: "POST", body: { name: f.name.value.trim(), tagline: f.tagline.value.trim(), profile, remix_of: f.dataset.remix || null } });
      for (const file of pendingFiles) await upload(b.id, file).catch((err) => toast(`${file.name}: ${err.message}`, "error"));
      toast(`${b.name} is ready. Chat with it in Peace.`);
      document.querySelector('.bot-tab[data-tab="mine"]').click();  // show it in My Bees
    }
    delete f.dataset.remix;
    fromIdea = null;
    $("botDlg").close();
    load();
  } catch (err) {
    $("botErr").textContent = err.message;
  }
});
$("botDlg").addEventListener("close", () => { fromIdea = null; });

function remix(b) {
  fromIdea = { tone: b.tone, starters: b.starters, name: `${b.name}`.slice(0, 24), tagline: b.skill };
  openEditor(null);
  $("botForm").dataset.remix = b.id;
  $("botTitle").textContent = `Remix ${b.name}`;
}

async function exportBot(b) {
  const data = await api(`/api/bots/${b.id}/export`);
  const a = h("a", { href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })), download: `${b.id}.mnxbot.json` });
  a.click();
}

async function removeBot(b) {
  if (!confirm(`Delete ${b.name}? Its memories, knowledge and computer go too. Its chat stays on this device.`)) return;
  try { await api(`/api/bees/${b.id}`, { method: "DELETE" }); load(); } catch (e) { toast(e.message, "error"); }
}

$("importFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  try {
    const b = await api("/api/bots/import", { method: "POST", body: JSON.parse(await file.text()) });
    toast(`Imported ${b.name}`);
    load();
  } catch (err) { toast(`Import: ${err.message}`, "error"); }
});

$("newBot").addEventListener("click", () => { fromIdea = null; openEditor(null); });

const client = new HiveClient();
installVoice(client);
client.addEventListener("state", (e) => { connLabel($("conn"), e.detail); if (e.detail === "online") load(); });
client.addEventListener("event", (e) => { if (e.detail.type === "bees_changed" && !$("botDlg").open) load(); });
installTokenPrompt(() => client.reconnectNow());
client.connect();

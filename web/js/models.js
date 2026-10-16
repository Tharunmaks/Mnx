// Mnx Hive — the model picker, like ChatGPT's: click the model name, pick a model, and use it. Names only.
// Everyone just picks. Only the owner can open "Set up models" (Drive links, downloading, start commands),
// and that stays behind the owner's token: users never download or configure anything.

import { api, getSettings, h, saveSettings, toast, whoami } from "./shared.js";

const $ = (id) => document.getElementById(id);
let choices = [];
let owner = false;
let setupList = [];

export const selectedModel = () => {
  const id = getSettings().modelId;
  return choices.some((c) => c.id === id) ? id : (choices[0] ? choices[0].id : null);
};

export async function refreshModelPill() {
  try { choices = await api("/api/models"); } catch { return; }
  owner = (await whoami()) === "owner";
  const cur = choices.find((c) => c.id === selectedModel());
  $("modelName").textContent = cur ? cur.name : "Mnx";
  $("modelPill").classList.toggle("empty", !choices.length && !owner);
  if (!$("modelMenu").classList.contains("hidden")) renderMenu();
}

// ---------- the picker menu ----------
function renderMenu() {
  const cur = selectedModel();
  const items = choices.map((c) => h("button", { type: "button", role: "option", class: `model-item${c.id === cur ? " on" : ""}`, "aria-selected": c.id === cur,
    onclick: () => { saveSettings({ modelId: c.id }); closeMenu(); refreshModelPill(); } }, h("span", {}, c.name)));
  if (!items.length) items.push(h("div", { class: "model-empty" }, owner ? "No models are set up yet." : "No models are available yet."));
  if (owner) items.push(h("div", { class: "model-sep" }), h("button", { type: "button", class: "model-item setup", onclick: () => { closeMenu(); openSetup(); } }, h("span", {}, "Set up models…")));
  $("modelMenu").replaceChildren(...items);
}

function openMenu() {
  renderMenu();
  $("modelMenu").classList.remove("hidden");
  $("modelPill").setAttribute("aria-expanded", "true");
  requestAnimationFrame(() => $("modelMenu").classList.add("open"));
}
function closeMenu() {
  $("modelMenu").classList.remove("open");
  $("modelPill").setAttribute("aria-expanded", "false");
  setTimeout(() => { if (!$("modelMenu").classList.contains("open")) $("modelMenu").classList.add("hidden"); }, 140);
}

export function installModelPicker() {
  $("modelPill").addEventListener("click", (e) => {
    e.stopPropagation();
    if ($("modelMenu").classList.contains("open")) closeMenu(); else openMenu();
  });
  document.addEventListener("click", (e) => { if (!$("modelMenu").contains(e.target)) closeMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
  refreshModelPill();
}

// ---------- the owner's setup (hidden from everyone else) ----------
const STATUS = {
  "not downloaded": ["Not downloaded", ""], downloading: ["Downloading", "busy"], ready: ["Ready", "ok"],
  starting: ["Starting", "busy"], running: ["Running", "ok"], stopped: ["Ready", "ok"], failed: ["Failed", "bad"],
};

async function openSetup() {
  if ((await whoami()) !== "owner") return;
  setupList = await api("/api/models?setup=1");
  renderSetup();
  $("modelsDlg").showModal();
}

function renderSetup() {
  $("modelsHint").textContent = "Only you see this. Paste each model's Google Drive folder link once (shared as “Anyone with the link”), download it, and it appears in everyone's model list. Users just pick a name and the Hive starts it for them.";
  $("modelList").replaceChildren(...setupList.map(setupCard));
}

function setupCard(m) {
  const [label, cls] = STATUS[m.status] || [m.status, ""];
  const box = h("div", { class: "model-card" },
    h("div", { class: "model-top" }, h("div", { class: "model-names" }, h("b", {}, m.name), h("span", { class: "hint" }, m.about)),
      h("span", { class: `pill ${cls}` }, label)),
    h("div", { class: "bot-facts" }, h("span", { class: "chip" }, `${m.disk_gb} GB`), h("span", { class: "chip" }, `needs ~${m.ram_gb} GB RAM`),
      h("span", { class: "chip" }, `runs with ${m.run}`)));
  if (m.progress) {
    const p = m.progress;
    const pct = p.files ? Math.round((100 * p.files_done) / p.files) : 0;
    box.append(h("div", { class: "model-progress" }, h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })),
      h("span", { class: "hint" }, p.stage === "downloading" ? `Part ${p.files_done + 1} of ${p.files} · ${p.file || ""}` : p.stage === "joining" ? "Joining the parts…" : "Checking MD5s…")));
  }
  if (m.error) box.append(h("p", { class: "hint err" }, m.error));

  if (!m.has_link) {
    const inp = h("input", { class: "input", placeholder: "Google Drive folder link", inputmode: "url" });
    box.append(h("div", { class: "row-inline" }, inp, h("button", { class: "btn primary", type: "button", onclick: async () => {
      try { await api(`/api/models/${m.id}/source`, { method: "PUT", body: { link: inp.value.trim() } }); toast("Link saved privately on the Hive"); reload(); }
      catch (e) { toast(e.message, "error"); }
    } }, "Save link")));
  }
  const acts = h("div", { class: "bot-actions" });
  const act = (text, action, style = "") => h("button", { class: `btn ${style}`, type: "button", onclick: (e) => run(e.currentTarget, m, action) }, text);
  const busy = m.status === "downloading" || m.status === "starting";
  if (!busy) {
    if (m.has_link && ["not downloaded", "failed"].includes(m.status)) acts.append(act(m.status === "failed" ? "Try again" : "Download", "download", "primary"));
    if (m.status === "running") acts.append(act("Stop", "stop"));
    if (m.path && m.status !== "running") acts.append(act("Delete files", "delete", "danger"));
  }
  if (m.kind !== "gguf") {
    const cmd = h("input", { class: "input", value: m.command || "", placeholder: "Start command, e.g. ./run_mnx1.sh" });
    const port = h("input", { class: "input", type: "number", value: m.port || 8000, style: "max-width:90px" });
    box.append(h("details", { class: "model-cmd" }, h("summary", { class: "hint" }, `How to start it (${m.run})`),
      h("p", { class: "hint" }, "A command that starts it with an OpenAI-compatible server (your own code), and that server's port. It runs in the model's folder; MNX_MODEL_DIR points there. Once saved, it appears in the model list and starts by itself when someone picks it."),
      h("div", { class: "row-inline" }, cmd, port, h("button", { class: "btn", type: "button", onclick: async () => {
        try { await api(`/api/models/${m.id}/command`, { method: "PUT", body: { command: cmd.value.trim(), port: +port.value } }); toast("Saved"); reload(); }
        catch (e) { toast(e.message, "error"); }
      } }, "Save"))));
  }
  if (acts.children.length) box.append(acts);
  return box;
}

async function reload() {
  setupList = await api("/api/models?setup=1").catch(() => setupList);
  if ($("modelsDlg").open) renderSetup();
  refreshModelPill();
}

async function run(btn, m, action) {
  if (action === "delete" && !confirm(`Delete ${m.name}'s files from this machine? You can download them again.`)) return;
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    await api(`/api/models/${m.id}/${action}`, { method: "POST" });
    if (action === "download") toast(`Downloading ${m.name}: it keeps going if you close this`);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    reload();
  }
}

export { reload as reloadSetup };

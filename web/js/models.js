// Mnx Hive — your models: the model button in the composer opens them. The owner downloads (from Drive), starts and
// switches models; everyone sees which one the Hive is thinking with.

import { api, h, toast, whoami } from "./shared.js";

const $ = (id) => document.getElementById(id);
let list = [];
let owner = false;

const STATUS = {
  "not downloaded": ["Not downloaded", ""], downloading: ["Downloading", "busy"], ready: ["Downloaded", "ok"],
  starting: ["Starting", "busy"], running: ["Running", "ok"], stopped: ["Stopped", ""], failed: ["Failed", "bad"],
};

export async function refreshModelPill() {
  try { list = await api("/api/models"); } catch { return; }
  const active = list.find((m) => m.in_use);
  $("modelName").textContent = active ? active.name : "No model";
  $("modelPill").title = active ? `Thinking with ${active.name}` : "Pick the model the Hive thinks with";
  if ($("modelsDlg").open) render();
}

export async function openModels() {
  owner = (await whoami()) === "owner";
  await refreshModelPill();
  render();
  $("modelsDlg").showModal();
}

function render() {
  $("modelsHint").textContent = owner
    ? "Paste each model's Google Drive folder link once (share the folder as “Anyone with the link”). The Hive downloads the parts, joins them, checks every MD5 and runs it."
    : "These are the Hive's models. The owner downloads and switches them.";
  $("modelList").replaceChildren(...list.map(card));
}

function card(m) {
  const [label, cls] = STATUS[m.status] || [m.status, ""];
  const head = h("div", { class: "model-top" },
    h("span", { class: "model-emoji" }, m.emoji),
    h("div", { class: "model-names" }, h("b", {}, m.name), h("span", { class: "hint" }, m.about)),
    h("span", { class: `pill ${m.in_use ? "ok" : cls}` }, m.in_use ? "In use" : label));
  const chips = h("div", { class: "bot-facts" },
    h("span", { class: "chip" }, `${m.disk_gb} GB`), h("span", { class: "chip" }, `needs ~${m.ram_gb} GB RAM`),
    h("span", { class: "chip" }, `runs with ${m.run}`));
  const box = h("div", { class: `model-card${m.in_use ? " active" : ""}` }, head, chips);
  if (m.progress) {
    const p = m.progress;
    const pct = p.files ? Math.round((100 * p.files_done) / p.files) : 0;
    box.append(h("div", { class: "model-progress" }, h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })),
      h("span", { class: "hint" }, p.stage === "downloading" ? `Part ${p.files_done + 1} of ${p.files} · ${p.file || ""}` : p.stage === "joining" ? "Joining the parts…" : "Checking MD5s…")));
  }
  if (m.error) box.append(h("p", { class: "hint err" }, m.error));
  if (!owner) return box;

  const acts = h("div", { class: "bot-actions" });
  const act = (label, action, style = "") => h("button", { class: `btn ${style}`, type: "button", onclick: (e) => run(e.currentTarget, m, action) }, label);
  if (!m.has_link) {
    const inp = h("input", { class: "input", placeholder: "Google Drive folder link", inputmode: "url" });
    box.append(h("div", { class: "row-inline" }, inp, h("button", { class: "btn primary", type: "button", onclick: async () => {
      try { await api(`/api/models/${m.id}/source`, { method: "PUT", body: { link: inp.value.trim() } }); toast("Link saved privately on the Hive"); refreshModelPill(); }
      catch (e) { toast(e.message, "error"); }
    } }, "Save link")));
  }
  if (m.status === "running") {
    if (!m.in_use) acts.append(act("Use it", "start", "primary"));
    acts.append(act("Stop", "stop"));
  } else if (m.status !== "downloading" && m.status !== "starting") {
    if (m.has_link && !["ready", "stopped"].includes(m.status)) acts.append(act(m.status === "failed" ? "Try again" : "Download", "download", "primary"));
    if (m.can_run && ["ready", "stopped", "failed"].includes(m.status)) acts.append(act("Start and use", "start", "primary"));
    if (["ready", "stopped", "failed"].includes(m.status) && m.path) acts.append(act("Delete files", "delete", "danger"));
  }
  if (m.kind !== "gguf") {
    const cmd = h("input", { class: "input", value: m.command || "", placeholder: "Start command, e.g. ./run_mnx1.sh" });
    const port = h("input", { class: "input", type: "number", value: m.port || 8000, style: "max-width:90px" });
    box.append(h("details", { class: "model-cmd" }, h("summary", { class: "hint" }, `How to start it (${m.run})`),
      h("p", { class: "hint" }, "A command that starts it with an OpenAI-compatible server (your own code), and that server's port. It runs in the model's folder; MNX_MODEL_DIR points there."),
      h("div", { class: "row-inline" }, cmd, port, h("button", { class: "btn", type: "button", onclick: async () => {
        try { await api(`/api/models/${m.id}/command`, { method: "PUT", body: { command: cmd.value.trim(), port: +port.value } }); toast("Saved"); refreshModelPill(); }
        catch (e) { toast(e.message, "error"); }
      } }, "Save"))));
  }
  if (acts.children.length) box.append(acts);
  return box;
}

async function run(btn, m, action) {
  if (action === "delete" && !confirm(`Delete ${m.name}'s files from this machine? You can download them again.`)) return;
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    await api(`/api/models/${m.id}/${action}`, { method: "POST" });
    if (action === "download") toast(`Downloading ${m.name}: it keeps going if you close this`);
    if (action === "start") toast(`${m.name} is running: chats and bots think with it now`);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    refreshModelPill();
  }
}

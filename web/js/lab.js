// Mnx Hive — Cloud Bee · Model Lab: train models, keep them, run them as APIs.

import {
  HiveClient, ICONS, api, beeHex, connLabel, fillIcons, fmtDuration, fmtSize, gatewayBase, getSettings, h,
  installTokenPrompt, toast, uploadFile,
} from "./shared.js";
import { drawCharts, playgroundFor, forgetPlayground } from "./labui.js";

const $ = (id) => document.getElementById(id);
fillIcons();
$("beeHex").innerHTML = beeHex("Cloud", "#4c9bff", "hex head-hex");

let data = null; // GET /api/lab
let sys = null; // GET /api/system
let recipe = null; // recipe selected in the Train form
let pending = []; // files chosen for the next run
let openRun = null;
const keys = {}; // last-rendered JSON per section, to skip redraws that change nothing
const runDetail = new Map(); // rid → {el, logSize, metricCount}

const STATUS_CLASS = { queued: "scheduled", preparing: "busy", running: "busy", succeeded: "", failed: "failed", stopped: "" };
const recipeTitle = (id) => data?.recipes.find((r) => r.id === id)?.title || id || "Imported";
const targetName = (id) => data?.targets.find((t) => t.id === id)?.name || id;

// ---------- loading ----------
async function load() {
  try {
    data = await api("/api/lab");
    connLabel($("conn"), "online");
  } catch (e) {
    connLabel($("conn"), e.status === 401 ? "auth" : "offline");
    return;
  }
  $("runCount").textContent = data.runs.length ? `(${data.runs.length})` : "";
  $("modelCount").textContent = data.models.length ? `(${data.models.length})` : "";
  if (!keys.recipes) { renderRecipes(); keys.recipes = true; }
  renderIf("targetsPick", data.targets, renderTargetPick);
  renderIf("runs", [data.runs, openRun], renderRuns);
  renderIf("models", [data.models, data.deployments], renderModels);
  renderIf("targets", [data.targets, data.secrets, data.hive_key, sys], renderServers);
}
function renderIf(name, value, fn) {
  const k = JSON.stringify(value);
  if (keys[name] !== k) { keys[name] = k; fn(); }
}
let loadTimer;
const loadSoon = () => { clearTimeout(loadTimer); loadTimer = setTimeout(load, 300); };

async function loadSys() {
  try { sys = await api("/api/system"); } catch { return; }
  const mem = sys.memory, disk = sys.disk;
  const chip = (label, value, warn) => h("div", { class: `sys-chip${warn ? " warn" : ""}` }, h("span", {}, label), h("b", {}, value));
  $("sys").replaceChildren(
    chip("CPU", `${sys.cpu_percent ?? "…"}% of ${sys.cpus}`, sys.cpu_percent > 90),
    chip("Memory free", mem.total ? `${fmtSize(mem.available)} / ${fmtSize(mem.total)}` : "—", mem.available < mem.total * 0.1),
    chip("Disk free", fmtSize(disk.free), disk.free < 5 * 1024 ** 3),
    chip("GPU", sys.gpus.length ? sys.gpus.map((g) => `${g.name} (${g.util}%)`).join(", ") : "none on this server"),
    chip("Training", String(sys.containers.training)),
    chip("Models running", String(sys.containers.models)));
}

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll(".lab-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  for (const n of ["train", "runs", "models", "servers"]) $(`tab-${n}`).classList.toggle("hidden", n !== name);
  history.replaceState(null, "", `#${name}`);
}
document.querySelectorAll(".lab-tabs .tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// ---------- Train ----------
function renderRecipes() {
  $("recipes").replaceChildren(...data.recipes.map((r) => h("button", { class: "recipe-card", type: "button", onclick: () => pickRecipe(r) },
    h("div", { class: "recipe-top" },
      h("span", { class: "recipe-ico", html: ICONS[{ classifier: "planning", tabular: "memory", llm: "thinking", custom: "writing" }[r.kind] || "running"] }),
      h("b", {}, r.title)),
    h("p", {}, r.summary),
    h("div", { class: "row" },
      r.gpu === "recommended" ? h("span", { class: "chip ask" }, "GPU recommended") : h("span", { class: "chip" }, r.gpu === "no" ? "CPU is enough" : "CPU or GPU"),
      r.has_serve ? h("span", { class: "chip" }, "Runs as an API") : null,
      r.has_sample ? h("span", { class: "chip" }, "Sample data") : null))));
}

function pickRecipe(r) {
  recipe = r;
  pending = [];
  const f = $("trainForm");
  f.reset();
  f.classList.remove("hidden");
  document.querySelectorAll(".recipe-card").forEach((c, i) => c.classList.toggle("selected", data.recipes[i] === r));
  $("formTitle").textContent = r.title;
  $("formSummary").textContent = r.summary;
  $("gpuBadge").classList.toggle("hidden", r.gpu !== "recommended");
  $("dataHint").textContent = r.dataset.hint;
  $("files").accept = r.dataset.accept === "*" ? "" : r.dataset.accept;
  $("sampleWrap").classList.toggle("hidden", !r.has_sample);
  f.use_sample.checked = !!r.has_sample;
  $("trainErr").textContent = "";
  const box = $("paramFields");
  box.replaceChildren();
  for (const p of r.params) {
    let input;
    if (p.type === "bool") {
      input = h("input", { type: "checkbox", name: `p_${p.name}`, checked: !!p.default });
      box.append(h("label", { class: "check", title: p.help || "" }, input, p.label));
      continue;
    }
    if (p.type === "select") input = h("select", { name: `p_${p.name}` }, (p.options || []).map((o) => h("option", { value: o, selected: o === p.default }, o)));
    else input = h("input", {
      name: `p_${p.name}`, value: p.default ?? "", type: p.type === "int" || p.type === "float" ? "number" : "text",
      step: p.type === "float" ? "any" : p.type === "int" ? "1" : null, min: p.min ?? null, max: p.max ?? null,
    });
    box.append(h("label", { class: "field", title: p.help || "" }, p.label, input, p.help ? h("span", { class: "hint" }, p.help) : null));
  }
  if (r.kind === "llm") {
    box.append(h("p", { class: "hint full" }, "Train your own Mnx: put your Mnx model's Hugging Face id (or a folder on the server) as the base model. Private models need HF_TOKEN under Servers → Secrets."));
  }
  syncGpu();
  renderPending();
  f.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTargetPick() {
  const pick = $("targetPick");
  const keep = pick.value;
  pick.replaceChildren(...data.targets.map((t) => h("option", { value: t.id, disabled: t.kind === "ssh" && t.status !== "ok" },
    `${t.name}${t.gpu ? " · GPU" : ""}${t.kind === "ssh" && t.status !== "ok" ? " (not reachable)" : ""}`)));
  const sshOk = data.targets.filter((t) => t.kind === "ssh" && t.status === "ok");
  if (sshOk.length >= 2) pick.append(h("option", { value: "__cluster__" }, `Cluster: all ${sshOk.length} connected servers together`));
  if (keep && data.targets.some((t) => t.id === keep)) pick.value = keep;
  syncGpu();
}
function syncGpu() {
  const v = $("targetPick").value;
  const t = v === "__cluster__" ? { gpu: data?.targets.some((x) => x.kind === "ssh" && x.gpu) } : data?.targets.find((x) => x.id === v);
  const box = $("gpuBox");
  box.disabled = !t?.gpu;
  box.checked = !!t?.gpu && recipe?.gpu !== "no";
  box.parentElement.title = t?.gpu ? "" : "No GPU on this server";
}
$("targetPick").addEventListener("change", syncGpu);
$("trainCancel").addEventListener("click", () => { $("trainForm").classList.add("hidden"); recipe = null; document.querySelectorAll(".recipe-card").forEach((c) => c.classList.remove("selected")); });

$("files").addEventListener("change", (e) => {
  for (const file of e.target.files) pending.push({ file });
  e.target.value = "";
  renderPending();
});
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  for (const file of e.dataTransfer.files) pending.push({ file });
  renderPending();
});
function renderPending() {
  $("fileList").replaceChildren(...pending.map((p, i) => h("span", { class: "chip file-chip" },
    `${p.file.name} · ${fmtSize(p.file.size)}${p.progress != null && p.progress < 100 ? ` · ${p.progress}%` : ""}`,
    h("button", { type: "button", "aria-label": `Remove ${p.file.name}`, onclick: () => { pending.splice(i, 1); renderPending(); } }, "×"))));
  if (pending.length) $("trainForm").use_sample.checked = false;
}

$("trainForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!recipe) return;
  const f = e.target;
  const btn = $("trainBtn");
  $("trainErr").textContent = "";
  const params = {};
  for (const p of recipe.params) {
    const el = f.elements[`p_${p.name}`];
    params[p.name] = p.type === "bool" ? el.checked : el.value;
  }
  btn.disabled = true;
  try {
    const uploads = [];
    for (const p of pending) {
      btn.textContent = `Uploading ${p.file.name}…`;
      const res = await uploadFile(p.file, (pct) => { p.progress = pct; renderPending(); });
      uploads.push(res.id);
    }
    btn.textContent = "Starting…";
    const run = await api("/api/lab/runs", {
      method: "POST",
      body: {
        recipe: recipe.id, name: f.name.value.trim(), params,
        target: f.target.value === "__cluster__" ? "local" : f.target.value,
        nodes: f.target.value === "__cluster__" ? data.targets.filter((t) => t.kind === "ssh" && t.status === "ok").map((t) => t.id) : null,
        uploads,
        use_sample: f.use_sample.checked, gpu: $("gpuBox").checked,
        cpus: f.cpus.value ? Number(f.cpus.value) : null, memory_gb: f.memory_gb.value ? Number(f.memory_gb.value) : null,
      },
    });
    toast(`Training started: ${run.name}`, "ok");
    pending = [];
    renderPending();
    openRun = run.id;
    showTab("runs");
    await load();
  } catch (err) {
    $("trainErr").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Start training";
  }
});

// ---------- Runs ----------
function since(t) { return t ? fmtDuration(Date.now() / 1000 - t) : "—"; }

function metricSummary(m) {
  if (!m) return "";
  return Object.entries(m).filter(([k, v]) => typeof v === "number" && !["sec", "lr"].includes(k))
    .slice(0, 5).map(([k, v]) => `${k.replace(/_/g, " ")} ${Number.isInteger(v) ? v : v.toFixed(4)}`).join(" · ");
}

function renderRuns() {
  const box = $("runs");
  box.replaceChildren();
  if (!data.runs.length) {
    box.append(h("p", { class: "empty-note" }, "No training runs yet. Start one from the Train tab."));
    return;
  }
  for (const r of data.runs) {
    const active = ["queued", "preparing", "running"].includes(r.status);
    const label = r.status === "preparing" ? (r.stage === "installing" ? "Installing packages" : "Preparing") : r.status[0].toUpperCase() + r.status.slice(1);
    const head = h("div", { class: "card-head" },
      h("div", { style: "min-width:0;flex:1" },
        h("div", {}, r.name),
        h("div", { class: "hint" }, `${recipeTitle(r.recipe)} · ${r.nodes ? `cluster of ${r.nodes.length}: ${r.nodes.map(targetName).join(", ")}` : targetName(r.target)}${r.gpu ? " · GPU" : ""} · ${active ? `running ${since(r.started)}` : r.ended ? `took ${fmtDuration(r.ended - r.started)}` : ""}`)),
      h("span", { class: `status ${STATUS_CLASS[r.status] ?? ""}` }, label));
    const summary = r.result ? h("div", { class: "metric-chips" }, Object.entries(r.result)
      .filter(([, v]) => typeof v === "number" || typeof v === "string").slice(0, 6)
      .map(([k, v]) => h("span", { class: "chip" }, `${k.replace(/_/g, " ")}: ${v}`)))
      : r.last_metric ? h("p", { class: "hint mono" }, metricSummary(r.last_metric)) : null;
    const actions = h("div", { class: "card-actions" },
      h("button", { class: "btn", type: "button", onclick: () => { openRun = openRun === r.id ? null : r.id; renderRuns(); } }, openRun === r.id ? "Hide details" : "Details"),
      active ? h("button", { class: "btn", type: "button", onclick: async (e) => {
        e.target.disabled = true;
        try { await api(`/api/lab/runs/${r.id}/stop`, { method: "POST" }); toast("Stopping…"); } catch (err) { toast(err.message, "error"); }
      } }, "Stop") : null,
      r.model ? h("button", { class: "btn primary", type: "button", onclick: () => { showTab("models"); document.getElementById(`model-${r.model}`)?.scrollIntoView({ behavior: "smooth" }); } }, "Open model") : null,
      !active ? h("button", { class: "btn danger", type: "button", onclick: async () => {
        if (!confirm(`Delete run "${r.name}" and its log? ${r.model ? "Its model stays in Models." : ""}`)) return;
        try { await api(`/api/lab/runs/${r.id}`, { method: "DELETE" }); runDetail.delete(r.id); load(); } catch (err) { toast(err.message, "error"); }
      } }, "Delete") : null);
    const card = h("div", { class: `card run-card ${r.status === "failed" ? "failed" : ""}` }, head,
      r.error ? h("p", { class: "card-text err" }, r.error) : null, summary, actions);
    if (openRun === r.id) card.append(detailFor(r));
    box.append(card);
  }
}

function detailFor(r) {
  let d = runDetail.get(r.id);
  if (!d) {
    const charts = h("div", { class: "charts" });
    const log = h("pre", { class: "result-box run-log" }, "Loading log…");
    d = { el: h("div", { class: "run-detail" }, charts, h("div", { class: "hint" }, "Log"), log), charts, log, logSize: 0, metricCount: -1, text: "" };
    runDetail.set(r.id, d);
  }
  refreshDetail(r.id);
  return d.el;
}

async function refreshDetail(rid) {
  const d = runDetail.get(rid);
  if (!d) return;
  try {
    const summary = data.runs.find((x) => x.id === rid);
    if (summary && summary.metric_count !== d.metricCount) {
      const full = await api(`/api/lab/runs/${rid}`);
      d.metricCount = full.metric_count;
      drawCharts(d.charts, full.metrics);
    }
    const res = await fetch(`${gatewayBase()}/api/lab/runs/${rid}/log?since=${d.logSize}`, { headers: { Authorization: `Bearer ${getSettings().token}` } });
    const size = Number(res.headers.get("X-Log-Size") || 0);
    if (size < d.logSize) { d.text = ""; d.logSize = 0; return refreshDetail(rid); } // log restarted
    const chunk = await res.text();
    const atBottom = d.log.scrollHeight - d.log.scrollTop - d.log.clientHeight < 40;
    if (chunk || !d.text) {
      d.text = (d.text + chunk).slice(-400_000);
      d.log.textContent = d.text.replace(/^MNX_(METRIC|STAGE|RESULT) .*\n/gm, "") || "(nothing yet)";
      if (atBottom || !d.logSize) d.log.scrollTop = d.log.scrollHeight;
    }
    d.logSize = size;
  } catch { /* next tick */ }
}

// ---------- Models ----------
function renderModels() {
  const box = $("models");
  box.replaceChildren();
  if (!data.models.length) {
    box.append(h("p", { class: "empty-note" }, "No models yet. Finished training runs appear here, or import a .gguf file (e.g. your Mnx)."));
    return;
  }
  for (const m of data.models) {
    if (m.kind === "blueprint") { box.append(blueprintCard(m)); continue; }
    const dep = data.deployments[m.id];
    const state = dep ? dep.status : null;
    const resultChips = m.result ? Object.entries(m.result).filter(([, v]) => typeof v === "number" || typeof v === "string").slice(0, 6)
      .map(([k, v]) => h("span", { class: "chip" }, `${k.replace(/_/g, " ")}: ${v}`)) : [];
    const gpuBox = data.gpu_local ? h("label", { class: "check" }, h("input", { type: "checkbox", class: "dep-gpu" }), "GPU") : null;
    const actions = h("div", { class: "card-actions" },
      m.servable && !dep ? h("button", { class: "btn primary", type: "button", onclick: async (e) => {
        e.target.disabled = true; e.target.textContent = "Starting…";
        try { await api(`/api/lab/models/${m.id}/deploy`, { method: "POST", body: { gpu: !!e.target.closest(".card").querySelector(".dep-gpu")?.checked } }); toast("Starting the model — first start installs packages", "ok"); }
        catch (err) { toast(err.message, "error"); }
        load();
      } }, "Run as API") : null,
      m.servable && !dep ? gpuBox : null,
      dep ? h("button", { class: "btn", type: "button", onclick: async () => { await api(`/api/lab/models/${m.id}/undeploy`, { method: "POST" }).catch((err) => toast(err.message, "error")); forgetPlayground(m.id); load(); } }, "Stop API") : null,
      dep ? h("button", { class: "btn", type: "button", onclick: async (e) => {
        const pre = e.target.closest(".card").querySelector(".dep-logs");
        pre.classList.toggle("hidden");
        if (!pre.classList.contains("hidden")) pre.textContent = (await fetchText(`/api/lab/models/${m.id}/logs`).catch(() => "")) || "(no output yet)";
      } }, "Server log") : null,
      h("a", { class: "btn", href: `${gatewayBase()}/api/lab/models/${m.id}/download?token=${encodeURIComponent(getSettings().token)}` }, "Download"),
      h("button", { class: "btn danger", type: "button", onclick: async () => {
        if (!confirm(`Delete model "${m.name}"? ${dep ? "Its API is stopped too." : ""}`)) return;
        try { await api(`/api/lab/models/${m.id}`, { method: "DELETE" }); forgetPlayground(m.id); load(); } catch (err) { toast(err.message, "error"); }
      } }, "Delete"));
    const card = h("div", { class: "card model-card", id: `model-${m.id}` },
      h("div", { class: "card-head" },
        h("div", { style: "min-width:0;flex:1" }, h("div", {}, m.name),
          h("div", { class: "hint" }, `${recipeTitle(m.recipe)} · ${fmtSize(m.size)} · ${new Date(m.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`)),
        state ? h("span", { class: `status ${state === "running" ? "busy" : state === "failed" ? "failed" : "scheduled"}` }, state === "running" ? "API running" : state) : null),
      resultChips.length ? h("div", { class: "metric-chips" }, resultChips) : null,
      !m.servable ? h("p", { class: "hint" }, "No serve.py, so it can't run as an API here. You can still download it.") : null,
      dep?.error ? h("p", { class: "card-text err" }, dep.error) : null,
      actions,
      h("pre", { class: "result-box dep-logs hidden" }));
    if (state === "running") card.append(playgroundFor(m));
    else if (state === "starting") card.append(h("p", { class: "hint" }, "Starting… the first start installs packages and loads the model; this can take a few minutes."));
    box.append(card);
  }
}

function blueprintCard(m) {
  const a = m.arch || {}, e = m.estimates || {}, ref = e.reference || {};
  return h("div", { class: "card model-card", id: `model-${m.id}` },
    h("div", { class: "card-head" },
      h("div", { style: "min-width:0;flex:1" }, h("div", {}, m.name),
        h("div", { class: "hint" }, `Blueprint · designed ${new Date(m.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}${m.purpose ? ` · ${m.purpose}` : ""}`)),
      h("span", { class: "badge wait" }, "Not trained")),
    h("div", { class: "metric-chips" },
      h("span", { class: "chip" }, `${a.params_text} parameters`), h("span", { class: "chip" }, `${a.layers} layers`),
      h("span", { class: "chip" }, `width ${a.hidden}`), h("span", { class: "chip" }, `${a.heads} heads`),
      h("span", { class: "chip" }, `vocab ${(a.vocab || 0).toLocaleString()}`), h("span", { class: "chip" }, `context ${a.seq_len}`)),
    h("p", { class: "card-text" }, `Weights ${e.weights || "?"} · training memory about ${e.train_memory_gpu || "?"} · a good setup is ${ref.count}× ${ref.gpu} for ${ref.time} (about ${ref.cost} to rent).`),
    h("div", { class: "card-actions" },
      h("button", { class: "btn primary", type: "button", onclick: () => {
        const r = data.recipes.find((x) => x.id === "llm-pretrain");
        if (!r) return;
        showTab("train"); pickRecipe(r);
        const f = $("trainForm");
        f.elements.p_size.value = "custom"; f.elements.p_custom_params.value = a.params; f.name.value = m.name.replace(/ blueprint$/, "");
      } }, "Train it"),
      h("button", { class: "btn danger", type: "button", onclick: async () => {
        if (!confirm(`Delete blueprint "${m.name}"?`)) return;
        try { await api(`/api/lab/models/${m.id}`, { method: "DELETE" }); load(); } catch (err) { toast(err.message, "error"); }
      } }, "Delete")));
}

async function fetchText(path) {
  const r = await fetch(gatewayBase() + path, { headers: { Authorization: `Bearer ${getSettings().token}` } });
  return r.text();
}

$("ggufFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const name = prompt("Name for this model:", file.name.replace(/\.gguf$/i, "")) || file.name;
  try {
    toast(`Uploading ${file.name}…`);
    const up = await uploadFile(file, () => {});
    await api("/api/lab/models/import", { method: "POST", body: { name, upload: up.id } });
    toast("Imported. Press Run as API to start it.", "ok");
    load();
  } catch (err) { toast(err.message, "error"); }
});

// ---------- Servers ----------
function renderServers() {
  const box = $("targets");
  box.replaceChildren();
  for (const t of data.targets) {
    if (t.kind === "local") {
      box.append(h("div", { class: "card" },
        h("div", { class: "card-head" }, h("span", { class: "svc" }, "◎"), h("div", { style: "flex:1" }, "This server",
          h("div", { class: "hint" }, sys ? `${sys.cpus} CPUs · ${fmtSize(sys.memory.total)} RAM · ${fmtSize(sys.disk.free)} disk free` : "")),
          h("span", { class: "badge ok" }, "Ready")),
        h("p", { class: "card-text" }, sys?.gpus.length
          ? `GPU: ${sys.gpus.map((g) => `${g.name}, ${g.memory_mb} MB`).join("; ")}${data.gpu_local ? "" : " — install the NVIDIA container toolkit so containers can use it"}`
          : "No GPU here: small models train fine on CPU; add a GPU server below for big ones.")));
      continue;
    }
    const info = t.info;
    box.append(h("div", { class: `card ${t.status === "failed" ? "failed" : ""}` },
      h("div", { class: "card-head" }, h("span", { class: "svc" }, t.name.slice(0, 2).toUpperCase()),
        h("div", { style: "flex:1;min-width:0" }, t.name, h("div", { class: "hint ellipsis" }, `${t.user}@${t.host}:${t.port} · ~/${t.workdir}`)),
        h("span", { class: `badge ${t.status === "ok" ? "ok" : t.status === "failed" ? "no" : "wait"}` }, t.status === "ok" ? "Connected" : t.status)),
      info ? h("p", { class: "card-text" }, `${info.system} · ${info.cpus} CPUs · ${info.ram_mb} MB RAM · ${info.python || "no python3"}${info.gpus.length ? ` · GPU: ${info.gpus.join("; ")}` : " · no GPU"}`) : null,
      t.error ? h("p", { class: "card-text err mono" }, t.error) : null,
      h("div", { class: "card-actions" },
        h("button", { class: "btn", type: "button", onclick: async (e) => {
          e.target.disabled = true; e.target.textContent = "Testing…";
          try { const r = await api(`/api/lab/targets/${t.id}/test`, { method: "POST" }); toast(r.status === "ok" ? "Connected" : "Couldn't connect", r.status === "ok" ? "ok" : "error"); } catch (err) { toast(err.message, "error"); }
          load();
        } }, "Test"),
        h("button", { class: "btn danger", type: "button", onclick: async () => {
          if (!confirm(`Remove ${t.name}?`)) return;
          try { await api(`/api/lab/targets/${t.id}`, { method: "DELETE" }); load(); } catch (err) { toast(err.message, "error"); }
        } }, "Remove"))));
  }
  renderProviders();
  $("hiveKeyCmd").textContent = data.hive_key
    ? `mkdir -p ~/.ssh && echo '${data.hive_key}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`
    : "ssh-keygen isn't installed on the Hive server; paste a private key below instead.";
  $("secretList").replaceChildren(...(data.secrets.length ? data.secrets.map((n) => h("span", { class: "chip file-chip" }, n,
    h("button", { type: "button", "aria-label": `Delete ${n}`, onclick: async () => {
      try { await api("/api/lab/secrets", { method: "POST", body: { name: n, value: "" } }); load(); } catch (err) { toast(err.message, "error"); }
    } }, "×"))) : [h("span", { class: "hint" }, "None yet.")]));
}

let providers = null;
async function renderProviders() {
  const box = $("providers");
  try { providers = await api("/api/lab/providers"); } catch { return; }
  const rp = providers.runpod;
  box.replaceChildren();
  const head = h("div", { class: "card-head" }, h("span", { class: "svc" }, "☁"), h("div", { style: "flex:1" }, "Rent a GPU (RunPod)",
    h("div", { class: "hint" }, rp.configured ? "API key saved · billing starts when a machine is ready and stops when you press Stop" : "Save RUNPOD_API_KEY under Secrets to rent GPUs from here or from the chat")),
    h("span", { class: `badge ${rp.configured ? "ok" : "wait"}` }, rp.configured ? "Ready" : "Not set up"));
  const card = h("div", { class: "card" }, head);
  for (const pod of rp.pods) {
    card.append(h("div", { class: "tool-row" },
      h("div", { class: "meta" }, h("div", { class: "name" }, `${pod.count}× ${pod.gpu_type}`),
        h("div", { class: "skill" }, `${pod.status}${pod.cost_per_hour ? ` · $${pod.cost_per_hour}/h` : ""}${pod.error ? ` · ${pod.error}` : ""} · since ${new Date(pod.started * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`)),
      h("button", { class: "btn danger", type: "button", onclick: async (e) => {
        if (!confirm("Stop this machine? Billing ends and it is removed from your servers.")) return;
        e.target.disabled = true;
        try { await api(`/api/lab/providers/runpod/pods/${pod.id}/stop`, { method: "POST" }); toast("Stopped", "ok"); } catch (err) { toast(err.message, "error"); }
        load();
      } }, "Stop")));
  }
  if (rp.error) card.append(h("p", { class: "card-text err" }, rp.error));
  if (rp.configured && rp.gpus.length) {
    const sel = h("select", { class: "input" }, rp.gpus.map((g) => h("option", { value: g.id }, `${g.name} · ${g.memory_gb} GB · $${g.price}/h${g.stock ? ` · ${g.stock}` : ""}`)));
    const count = h("input", { class: "input", type: "number", min: 1, max: 8, value: 1, style: "max-width:80px" });
    card.append(h("div", { class: "row" }, sel, count, h("button", { class: "btn primary", type: "button", onclick: async (e) => {
      const g = rp.gpus.find((x) => x.id === sel.value);
      if (!confirm(`Rent ${count.value}× ${g.name} for $${(g.price * count.value).toFixed(2)} per hour? This spends real money.`)) return;
      e.target.disabled = true;
      try { await api("/api/lab/providers/runpod/rent", { method: "POST", body: { gpu_type: sel.value, count: Number(count.value) } }); toast("Renting… it appears under Servers when SSH is up", "ok"); }
      catch (err) { toast(err.message, "error"); }
      e.target.disabled = false;
      load();
    } }, "Rent")));
  }
  box.append(card);
}

$("copyKey").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("hiveKeyCmd").textContent); toast("Copied", "ok"); }
  catch { toast("Select the text and copy it"); }
});

$("targetForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const btn = $("targetBtn");
  btn.disabled = true; btn.textContent = "Connecting…";
  $("targetErr").textContent = "";
  try {
    const t = await api("/api/lab/targets", {
      method: "POST",
      body: { name: f.name.value.trim(), host: f.host.value.trim(), port: Number(f.port.value) || 22, user: f.user.value.trim(),
              workdir: f.workdir.value.trim() || "mnx-runs", private_key: f.private_key.value.trim() || null },
    });
    if (t.status === "ok") { toast(`${t.name} connected`, "ok"); f.reset(); }
    else $("targetErr").textContent = `Saved, but couldn't connect yet: ${t.error || "unknown error"}. Add the Hive key on the server, then press Test.`;
    load();
  } catch (err) {
    $("targetErr").textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Add and test";
  }
});

$("secretForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  try {
    await api("/api/lab/secrets", { method: "POST", body: { name: f.name.value.trim().toUpperCase(), value: f.value.value } });
    toast("Saved", "ok");
    f.reset();
    load();
  } catch (err) { toast(err.message, "error"); }
});

// ---------- live updates ----------
const client = new HiveClient();
client.addEventListener("state", (e) => { connLabel($("conn"), e.detail); if (e.detail === "online") load(); });
client.addEventListener("event", (e) => { if (e.detail.type === "lab_changed") loadSoon(); });
client.connect();
installTokenPrompt(() => { client.reconnectNow(); load(); });

setInterval(() => {
  if (document.visibilityState !== "visible") return;
  if (openRun) refreshDetail(openRun);
}, 2000);
setInterval(() => { if (document.visibilityState === "visible") { loadSys(); loadSoon(); } }, 6000);

const tabFromHash = () => (["train", "runs", "models", "servers"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "train");
window.addEventListener("hashchange", () => showTab(tabFromHash()));
showTab(tabFromHash());
loadSys().then(load);

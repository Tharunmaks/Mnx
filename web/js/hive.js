// Mnx Hive — Hive control screen: every Bee's state, add/remove Bees, activity log.

import { ICONS, HiveClient, api, beeHex, connLabel, h, installTokenPrompt, timeShort } from "./shared.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

const STATUSES = ["busy", "idle", "scheduled", "failed"];
let BEES = [];
let COMPUTERS = { sizes: {}, servers: [] };

async function loadComputers() {
  try { COMPUTERS = await api("/api/computers"); } catch { /* offline */ }
}

function fill(select, options, value) {
  select.replaceChildren(...options.map(([v, label]) => h("option", { value: v }, label)));
  if (value != null) select.value = value;
}

function sizeOptions() {
  return Object.entries(COMPUTERS.sizes).map(([k, v]) => [k, v.cpus ? `${v.label} · ${v.cpus} CPU${v.cpus > 1 ? "s" : ""}, ${v.memory_gb} GB` : v.label]);
}

function serverOptions() {
  return COMPUTERS.servers.map((s) => [s.id, `${s.name}${s.cpus ? ` · ${s.cpus} CPUs` : ""}${s.memory_gb ? `, ${s.memory_gb} GB` : ""}${s.gpu ? " · GPU" : ""}${s.status && s.status !== "ok" ? ` (${s.status})` : ""}`]);
}

async function loadBees() {
  let bees = [];
  let offline = false;
  try { bees = await api("/api/bees"); } catch { offline = true; }
  BEES = bees;

  for (const s of STATUSES) {
    $("n" + s[0].toUpperCase() + s.slice(1)).textContent = bees.filter((b) => b.status === s).length;
  }

  const grid = $("beeGrid");
  grid.replaceChildren();
  for (const b of bees) {
    const top = h("div", { class: "top" });
    top.innerHTML = beeHex(b.name, b.color);
    top.append(h("div", { style: "flex:1;min-width:0" },
      h("div", { class: "name" }, `${b.name} Bee`),
      h("span", { class: `status ${b.status}` }, b.status)));
    if (!b.builtin) {
      const del = h("button", { class: "icon-btn", "aria-label": `Remove ${b.name} Bee`, html: ICONS.trash, onclick: async () => {
        if (!confirm(`Remove ${b.name} Bee? Its Cell is deleted too: container, files, browser storage and jobs.`)) return;
        try { await api(`/api/bees/${encodeURIComponent(b.id)}`, { method: "DELETE" }); } catch { /* offline */ }
        loadBees();
      } });
      top.append(del);
    }
    const tools = h("div", { class: "tools" },
      (b.tools || []).map((t) => h("span", { class: "chip" }, t)),
      b.approval ? h("span", { class: "chip ask" }, "Ask lane") : null,
      b.schedule ? h("span", { class: "chip" }, `⏱ ${b.schedule}`) : null);
    const family = h("div", { class: "cell-line" },
      h("span", { class: "chip" }, b.builtin ? "Built in" : `Made by ${b.parent_name || "you"}`),
      b.children && b.children.length ? h("span", { class: "chip" }, `${b.children.length} child${b.children.length > 1 ? "ren" : ""}`) : null,
      h("span", { class: "chip", title: "This Bee's computer" }, `🖥 ${b.computer_text}`));
    const c = b.cell;
    const cellLine = h("div", { class: "cell-line" },
      h("span", { class: "chip" }, !c || c.container === "missing" ? "Cell starts on first use" : c.mode === "docker" ? `Container ${c.container}` : "Local Cell"),
      c && c.jobs ? h("span", { class: "chip" }, `${c.jobs} job${c.jobs > 1 ? "s" : ""}${c.running ? ` · ${c.running} running` : ""}`) : null,
      c && c.terminal ? h("span", { class: "chip" }, "Terminal open") : null,
      c && c.browser ? h("span", { class: "chip" }, "Browser open") : null);
    grid.append(h("div", { class: "bee-card" }, top, h("p", {}, b.skill || ""), tools, family, cellLine,
      h("div", { class: "card-actions" },
        h("a", { class: "btn primary", href: `cell.html?bee=${encodeURIComponent(b.id)}` }, "Open Cell"),
        h("button", { class: "btn", onclick: () => openComputer(b) }, "Computer"),
        b.id === "cloud"
          ? h("a", { class: "btn", href: "lab.html" }, "Model Lab")
          : h("a", { class: "btn", href: `cell.html?bee=${encodeURIComponent(b.id)}#jobs` }, "Jobs"))));
  }
  grid.append(h("button", { class: "bee-card add", onclick: openAdd, disabled: offline },
    h("span", { html: ICONS.plus, style: "width:26px;height:26px;display:block" }),
    offline ? "Gateway offline" : "Add Bee"));
}

async function loadLog() {
  let events = [];
  try { events = await api("/api/events?limit=60"); } catch { /* offline */ }
  const log = $("log");
  log.replaceChildren();
  if (!events.length) {
    log.append(h("div", { class: "log-row" }, h("span", { class: "x", style: "grid-column:1/-1" }, "Nothing has happened yet.")));
    return;
  }
  for (const e of events.slice().reverse()) {
    log.append(h("div", { class: "log-row" },
      h("span", { class: "t" }, timeShort(e.at)),
      h("span", { class: "b" }, e.bee || "Queen"),
      h("span", { class: "x" }, `${e.type}${e.status ? " " + e.status : ""}${e.text ? " · " + e.text : ""}`)));
  }
}

async function openAdd() {
  $("beeForm").reset();
  $("beeErr").textContent = "";
  await loadComputers();
  fill($("beeParent"), [["you", "You"], ...BEES.map((b) => [b.id, `${b.name} Bee`])], "you");
  fill($("beeSize"), sizeOptions(), "small");
  fill($("beeWhere"), serverOptions(), "here");
  $("beeDlg").showModal();
}

let compBee = null;
async function openComputer(b) {
  compBee = b;
  await loadComputers();
  $("compForm").reset();
  $("compErr").textContent = "";
  $("compTitle").textContent = `${b.name} Bee's computer`;
  $("compNow").textContent = `Now: ${b.computer_text}.${COMPUTERS.limits ? ` Limits on this server: ${COMPUTERS.limits}.` : ""}`;
  const c = b.computer || {};
  fill($("compSize"), sizeOptions(), c.size in COMPUTERS.sizes ? c.size : "small");
  fill($("compWhere"), serverOptions(), c.where || "here");
  $("compForm").gpu.checked = !!c.gpu;
  $("compDlg").showModal();
}

$("compForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save" || !compBee) return;
  e.preventDefault();
  const f = e.target;
  $("compErr").textContent = "Applying…";
  try {
    await api(`/api/bees/${encodeURIComponent(compBee.id)}/computer`, { method: "PUT", body: { size: f.size.value, where: f.where.value, gpu: f.gpu.checked } });
    $("compDlg").close();
    loadBees();
  } catch (err) {
    $("compErr").textContent = `Couldn't change it (${err.message}).`;
  }
});
$("addBeeTop").addEventListener("click", openAdd);
$("refreshLog").addEventListener("click", loadLog);

$("beeForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  const f = e.target;
  try {
    await api("/api/bees", {
      method: "POST",
      body: {
        name: f.name.value.trim(),
        skill: f.skill.value.trim(),
        tools: f.tools.value.split(",").map((s) => s.trim()).filter(Boolean),
        schedule: f.schedule.value.trim() || null,
        approval: f.approval.checked,
        created_by: f.parent.value || "you",
        computer: { size: f.size.value, where: f.where.value, gpu: f.gpu.checked },
      },
    });
    $("beeDlg").close();
    loadBees();
  } catch (err) {
    $("beeErr").textContent = `Couldn't add the Bee (${err.message}).`;
  }
});

// Live updates
const client = new HiveClient();
client.addEventListener("state", (e) => {
  connLabel($("conn"), e.detail);
  if (e.detail === "online") { loadBees(); loadLog(); }
});
installTokenPrompt(() => client.reconnectNow());
let logTimer = null;
client.addEventListener("event", (e) => {
  const t = e.detail.type;
  if (t === "bee_status" || t === "bees_changed" || t === "bee_created") loadBees();
  clearTimeout(logTimer);
  logTimer = setTimeout(loadLog, 400);
});
client.connect();

loadBees();
loadLog();

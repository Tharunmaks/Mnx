// Mnx Hive — Free cloud: the free Plus plan, the free GPU (Kaggle) and free servers (Oracle Always Free).

import { ICONS, HiveClient, api, connLabel, h, installTokenPrompt, toast, whoami } from "./shared.js";
import { installVoice } from "./voice.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

async function loadUser() {
  const [st, mine] = await Promise.all([api("/api/cloud/status"), api("/api/plan")]);
  const plus = mine.plan === "plus";
  $("planText").textContent = plus ? "You're on Plus. Every Bee has the Plus computer." : `You're on Free (${mine.computer}). Plus is free too.`;
  $("subscribe").textContent = plus ? "On Plus ✓" : "Subscribe to Plus · ₹0";
  $("subscribe").disabled = plus || mine.locked;
  $("userStatus").textContent = (st.gpu_ready ? `Free GPU ready (${st.hours_week.toFixed(1)} of ${st.weekly_limit} hours used this week). ` : "The free GPU isn't set up yet. ")
    + (st.servers ? `${st.servers} Hive server${st.servers === 1 ? "" : "s"} running your Bees.` : "Your Bees run on the Hive's own machine for now.");
}

async function load() {
  if ((await whoami()) !== "owner") {
    document.documentElement.classList.remove("is-owner");
    try { await loadUser(); } catch { $("planText").textContent = "Couldn't reach the Hive."; }
    return;
  }
  $("userNote").classList.add("hidden");
  let s;
  try { s = await api("/api/cloud/free"); } catch { $("planText").textContent = "Couldn't reach the Hive."; return; }
  const plus = s.plan.plan === "plus";
  $("planText").textContent = plus ? "You're on Plus. Every Bee has the Plus computer." : `You're on Free (${s.plan.computer}). Plus is free too.`;
  $("subscribe").textContent = plus ? "On Plus ✓" : "Subscribe to Plus · ₹0";
  $("subscribe").disabled = plus || s.plan.locked;
  $("stepPlan").classList.toggle("done", plus);

  const g = s.gpu;
  $("stepGpu").classList.toggle("done", g.kaggle || g.local);
  $("gpuText").textContent = g.local ? "This machine has its own GPU: Plus Bees use it directly."
    : g.kaggle ? `Connected as ${s.kaggle_user}. ${g.hours_week.toFixed(1)} of ${g.weekly_limit} free GPU hours used this week.` + (g.cli ? "" : " The Kaggle tool isn't installed on the Hive yet: run  pip install kaggle")
    : "Not connected yet.";
  if (s.kaggle_user) $("kaggleForm").username.value = s.kaggle_user;

  $("hiveKey").textContent = s.hive_key || "(the Hive has no SSH key: ssh-keygen is missing)";
  $("stepServer").classList.toggle("done", s.pool.some((p) => p.status === "ok"));
  $("poolList").replaceChildren(...s.pool.map((p) => h("div", { class: "pool-row" },
    h("b", {}, p.name), h("span", { class: "hint" }, `${p.cpus || "?"} CPUs · ${p.ram_gb || "?"} GB · ${p.status}${p.setup ? " · " + p.setup : ""}`),
    h("span", { class: "hint" }, `${p.bees.length} of ${p.slots} Plus Bees: ${p.bees.join(", ") || "none yet"}`),
    h("button", { class: "btn danger", type: "button", onclick: async () => {
      try { await api(`/api/cloud/pool/${p.id}`, { method: "DELETE" }); load(); } catch (e) { toast(e.message, "error"); }
    } }, "Remove"))));

  loadIota();
  const gpu = await api("/api/gpu").catch(() => ({ jobs: [] }));
  $("jobNote").textContent = gpu.jobs.length ? `${gpu.jobs.length} recent` : "";
  const t = $("jobs");
  t.replaceChildren(...(gpu.jobs.length ? gpu.jobs.map((j) => h("div", { class: "urow", style: "grid-template-columns:1fr 2fr 1fr 1.4fr" },
    h("span", {}, j.bee), h("span", {}, j.command), h("span", {}, j.status), h("span", { class: "hint" }, j.error || (j.outputs || []).slice(0, 3).join(", ") || "")))
    : [h("p", { class: "hint" }, "No GPU jobs yet.")]));
}

// ---------- Iotamine ----------
async function loadIota() {
  const d = await api("/api/cloud/iotamine").catch(() => ({ connected: false, servers: [] }));
  $("iotaDeploy").classList.toggle("hidden", !d.connected);
  $("stepIota").classList.toggle("done", d.servers.some((v) => v.in_hive));
  $("iotaText").textContent = !d.connected ? "Not connected yet." : d.error ? d.error
    : d.servers.length ? `${d.servers.length} VPS on your account.` : "No VPS on your account yet: deploy one below.";
  $("iotaList").replaceChildren(...d.servers.map((v) => h("div", { class: "pool-row" },
    h("b", {}, v.name), h("span", { class: "hint" }, `${v.cores ?? "?"} vCPU · ${v.ram_gb ?? "?"} GB · ${v.disk_gb ?? "?"} GB disk · ${v.ip || "no IP yet"} · ${v.status || ""}`),
    v.in_hive ? h("span", { class: "chip" }, "In the Hive ✓") : h("button", { class: "btn primary", type: "button", disabled: !v.ip, onclick: async () => {
      if (!confirm(`Add ${v.name} (${v.ip}) to the Hive's cloud? It needs the Hive's SSH key (step 3) in root's ~/.ssh/authorized_keys. A VPS the Hive deployed has it already.`)) return;
      try { const r = await api("/api/cloud/pool", { method: "POST", body: { host: v.ip, user: "root" } }); toast(r.text); } catch (e) { toast(e.message, "error"); }
    } }, "Use for the Hive"))));
}

$("iotaForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("iotaText").textContent = "Checking the key with Iotamine…";
  try {
    const r = await api("/api/cloud/iotamine/key", { method: "POST", body: { key: e.target.key.value.trim() } });
    e.target.key.value = "";
    if (!r.ok) $("iotaText").textContent = r.error;
    loadIota();
  } catch (err) { $("iotaText").textContent = err.message; }
});

$("iotaDeploy").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = { cores: +f.cores.value, ram_gb: +f.ram_gb.value, disk_gb: +f.disk_gb.value };
  try {
    const { quote: q } = await api("/api/cloud/iotamine/deploy", { method: "POST", body });
    if (!confirm(`Deploy on Iotamine (${q.region}): ${q.cores} vCPU, ${q.ram_gb} GB RAM, ${q.disk_gb} GB disk.\n\nAbout ₹${q.hourly} per hour (≈ ₹${q.monthly} a month, including the IP at ₹${q.ip_monthly}/month), from Iotamine's own rates; your plan's free credit may cover it.\n\nThe Hive puts its SSH key on it and adds it to its cloud.`)) return;
    await api("/api/cloud/iotamine/deploy", { method: "POST", body: { ...body, pop: q.pop, confirm: true } });
    toast("Deploying: it shows up here in a few minutes");
  } catch (err) { toast(err.message, "error"); }
});

$("subscribe").addEventListener("click", async () => {
  try { const r = await api("/api/plan", { method: "PUT", body: { plan: "plus" } }); toast(`You're on Plus. ${r.changed.length} Bees got the Plus computer.`); load(); }
  catch (e) { toast(e.message, "error"); }
});

$("kaggleForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  $("gpuText").textContent = "Checking the Kaggle account…";
  try {
    const r = await api("/api/cloud/kaggle", { method: "POST", body: { username: f.username.value.trim(), token: f.token.value.trim() } });
    f.token.value = "";
    if (!r.ok) $("gpuText").textContent = r.error; else { toast("Free GPU connected"); load(); }
  } catch (err) { $("gpuText").textContent = err.message; }
});

$("poolForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  try { const r = await api("/api/cloud/pool", { method: "POST", body: { host: f.host.value.trim(), user: f.user.value.trim() } }); toast(r.text); }
  catch (err) { toast(err.message, "error"); }
});

$("copyKey").addEventListener("click", () => { navigator.clipboard?.writeText($("hiveKey").textContent).then(() => toast("Copied")); });

const client = new HiveClient();
installVoice(client);
client.addEventListener("state", (e) => { connLabel($("conn"), e.detail); if (e.detail === "online") load(); });
client.addEventListener("event", (e) => {
  const ev = e.detail;
  if (ev.type === "cloud_ready") { toast(ev.text); load(); }
  if (ev.type === "cloud_error") toast(ev.text, "error");
  if (ev.type === "gpu_changed" || ev.type === "plan_changed") load();
});
installTokenPrompt(() => client.reconnectNow());
client.connect();

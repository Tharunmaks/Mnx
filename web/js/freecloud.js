// Mnx Hive — Free cloud: the free Plus plan, the free GPU (Kaggle) and free servers (Oracle Always Free).

import { ICONS, HiveClient, api, connLabel, h, installTokenPrompt, toast } from "./shared.js";
import { installVoice } from "./voice.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

async function load() {
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

  const gpu = await api("/api/gpu").catch(() => ({ jobs: [] }));
  $("jobNote").textContent = gpu.jobs.length ? `${gpu.jobs.length} recent` : "";
  const t = $("jobs");
  t.replaceChildren(...(gpu.jobs.length ? gpu.jobs.map((j) => h("div", { class: "urow", style: "grid-template-columns:1fr 2fr 1fr 1.4fr" },
    h("span", {}, j.bee), h("span", {}, j.command), h("span", {}, j.status), h("span", { class: "hint" }, j.error || (j.outputs || []).slice(0, 3).join(", ") || "")))
    : [h("p", { class: "hint" }, "No GPU jobs yet.")]));
}

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

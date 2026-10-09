// Mnx Hive: Usage and limits. Every number comes from /api/usage and is measured; what can't be measured says so.

import { ICONS, HiveClient, api, connLabel, fmtDuration, fmtSize, h, installTokenPrompt } from "./shared.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

const num = (n) => (n == null ? "—" : Number(n).toLocaleString());
const pct = (a, b) => (b ? Math.min(100, (100 * a) / b) : 0);
const usd = (n) => (n == null ? "price unknown" : `$${n < 10 ? n.toFixed(2) : n.toFixed(0)}`);
const hours = (x) => (x < 1 ? `${Math.round(x * 60)} min` : `${x.toFixed(x < 10 ? 1 : 0)} h`);
let tableView = false;
let last = null;

function tile(label, value, sub) {
  return h("div", { class: "stat" }, h("div", { class: "n" }, value), h("div", { class: "l" }, label), sub ? h("div", { class: "sub" }, sub) : null);
}

function meter(label, used, total, text) {
  const p = total ? pct(used, total) : 0;
  const cls = !total ? "none" : p >= 90 ? "hot" : p >= 75 ? "warn" : "";
  return h("div", { class: "meter-row" },
    h("span", { class: "lbl" }, label),
    h("div", { class: `meter ${cls}`, role: "img", "aria-label": `${label}: ${total ? p.toFixed(0) + "%" : "not measured"}` }, h("i", { style: `width:${p}%` })),
    h("span", { class: "val" }, text + (total ? ` · ${p.toFixed(0)}%` : "")));
}

function table(el, cols, rows) {
  const tpl = cols.map((c) => c.w || "1fr").join(" ");
  el.replaceChildren(
    h("div", { class: "urow uhead", style: `grid-template-columns:${tpl}` }, cols.map((c) => h("span", { class: c.num ? "num" : "" }, c.h))),
    ...rows.map((r) => h("div", { class: "urow", style: `grid-template-columns:${tpl}` },
      cols.map((c, i) => { const v = r[i]; return v instanceof Node ? v : h("span", { class: (c.num ? "num " : "") + (v === "not measured" ? "muted" : "") }, v); }))));
}

function drawChart(t) {
  const box = $("tokChart");
  const max = Math.max(1, ...t.series.map((d) => d.tokens));
  if (tableView) {
    const el = h("div", { class: "utable" });
    table(el, [{ h: "Day", w: "110px" }, { h: "Messages", num: true }, { h: "Tokens in", num: true }, { h: "Tokens out", num: true }, { h: "Total", num: true }],
      t.series.slice().reverse().map((d) => [d.day, num(d.messages), num(d.in), num(d.out), num(d.tokens)]));
    box.replaceChildren(el);
    return;
  }
  const tip = h("div", { class: "tip", style: "display:none" });
  const cols = t.series.map((d) => {
    const label = `${d.day}: ${num(d.tokens)} tokens in ${num(d.messages)} message${d.messages === 1 ? "" : "s"} (${num(d.in)} in, ${num(d.out)} out)`;
    const col = h("div", { class: "col", tabindex: "0", "aria-label": label },
      d.tokens === max && d.tokens > 0 ? h("em", {}, num(d.tokens)) : null,
      h("b", { style: `height:${(82 * d.tokens) / max}%` }));
    const show = () => {
      tip.textContent = label; tip.style.display = "block";
      const r = col.getBoundingClientRect(), w = tip.offsetWidth;
      tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
      tip.style.top = Math.max(8, r.top - 8) + "px";
    };
    col.addEventListener("mouseenter", show); col.addEventListener("focus", show);
    col.addEventListener("mouseleave", () => { tip.style.display = "none"; }); col.addEventListener("blur", () => { tip.style.display = "none"; });
    return col;
  });
  const empty = t.series.every((d) => d.tokens === 0);
  box.replaceChildren(...[h("div", { class: "bars" }, cols),
    h("div", { class: "bar-days" }, t.series.map((d) => h("span", {}, d.day.slice(5)))), tip,
    empty ? h("p", { class: "hint", style: "margin-top:10px" }, "No messages counted yet. Each message you send is added here when it finishes.") : null].filter(Boolean));
}

function render(u) {
  last = u;
  $("stamp").textContent = `Measured ${new Date(u.at * 1000).toLocaleTimeString()}; refreshes every 10 seconds.`;

  // ----- tokens
  const t = u.tokens;
  $("tokNote").textContent = t.estimate;
  $("tokStats").replaceChildren(
    tile("Tokens today", num(t.today.tokens), `${num(t.today.in)} in · ${num(t.today.out)} out`),
    tile("Messages today", num(t.today.messages), t.today.over_budget ? `${t.today.over_budget} went over their limit` : "none over their limit"),
    tile("Last 7 days", num(t.week.tokens), `${num(t.week.messages)} messages`),
    tile("All time", num(t.all.tokens), t.all.since ? `${num(t.all.messages)} messages since ${t.all.since}` : "nothing counted yet"));
  drawChart(t);
  $("viewToggle").textContent = tableView ? "Chart" : "Table";
  const levels = Object.entries(u.limits.efforts);
  table($("effortTable"), [{ h: "Level", w: "90px" }, { h: "Limit per message", num: true }, { h: "Messages today", num: true }, { h: "Biggest today", num: true }, { h: "Longest reply", num: true }],
    levels.map(([id, l]) => {
      const e = t.efforts_today[id];
      return [l.label, num(l.tokens), e ? num(e.messages) : "0", e ? `${num(e.peak)} (${pct(e.peak, l.tokens).toFixed(0)}%)` : "—", `${num(l.reply)} tokens`];
    }));

  // ----- server
  const s = u.server;
  const memTotal = s.memory.total, memUsed = s.memory.total && s.memory.available != null ? s.memory.total - s.memory.available : null;
  const diskUsed = s.disk.total - s.disk.free;
  const meters = [
    meter("CPU", s.cpu_percent ?? 0, s.cpu_percent == null ? 0 : 100, s.cpu_percent == null ? `${s.cpus} CPUs · measuring…` : `${s.cpus} CPUs · load ${s.load[0]}`),
    meter("Memory", memUsed ?? 0, memTotal, memUsed == null ? "—" : `${fmtSize(memUsed)} of ${fmtSize(memTotal)}`),
    meter("Disk", diskUsed, s.disk.total, `${fmtSize(diskUsed)} of ${fmtSize(s.disk.total)}`),
    h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "Hive data folder"), h("span"), h("span", { class: "val" }, fmtSize(s.data_bytes))),
  ];
  for (const g of s.gpus) {
    meters.push(meter(`GPU ${g.name}`, g.used_mb, g.memory_mb, `${fmtSize(g.used_mb * 1024 ** 2)} of ${fmtSize(g.memory_mb * 1024 ** 2)} · busy ${g.util}%`));
  }
  if (!s.gpus.length) meters.push(h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "GPU"), h("span"), h("span", { class: "val" }, "none on this server")));
  $("serverMeters").replaceChildren(...meters);
  $("serverNote").textContent = `Up ${fmtDuration(s.uptime)} · Cells run in ${s.cell_mode} mode`;

  // ----- bees
  const b = u.bees;
  $("beeNote").textContent = b.docker_measured ? "CPU and memory measured by Docker" : "CPU and memory need Docker; disk and jobs are measured";
  $("beeStats").replaceChildren(
    tile("Bees", `${b.count} / ${b.limit}`, `${b.built_in} built in · ${b.yours} made by you · ${b.by_bees} made by Bees`),
    tile("Deepest family", `${b.deepest} / ${b.max_depth}`, "generations"),
    tile("Children per Bee", `up to ${b.max_children}`, "limit"),
    tile("Jobs running", num(b.rows.reduce((a, r) => a + r.running, 0)), `of ${num(b.rows.reduce((a, r) => a + r.jobs, 0))} jobs`));
  table($("beeTable"), [{ h: "Bee", w: "1.1fr" }, { h: "Computer", w: "1.6fr" }, { h: "CPU", num: true, w: ".9fr" }, { h: "Memory", num: true, w: "1.3fr" }, { h: "Disk", num: true, w: ".8fr" }, { h: "Jobs", num: true, w: ".6fr" }],
    b.rows.slice().sort((x, y) => (y.cpu_percent ?? -1) - (x.cpu_percent ?? -1) || (y.disk_bytes ?? 0) - (x.disk_bytes ?? 0)).map((r) => [
      r.name + (r.parent && r.parent !== "you" && !r.builtin ? ` ← ${r.parent}` : ""),
      r.computer,
      r.measured ? `${r.cpu_percent.toFixed(1)}%${r.cpus_limit ? ` of ${r.cpus_limit}` : ""}` : "not measured",
      r.measured ? `${fmtSize(r.memory_bytes)}${r.memory_limit_bytes ? ` of ${fmtSize(r.memory_limit_bytes)}` : ""}` : "not measured",
      r.disk_bytes == null ? "on its server" : fmtSize(r.disk_bytes),
      `${r.running}/${r.jobs}`]));

  // ----- lab
  const l = u.lab;
  $("labStats").replaceChildren(
    tile("Training runs", num(l.runs), Object.entries(l.by_status).map(([k, v]) => `${v} ${k}`).join(" · ") || "none yet"),
    tile("Time training", fmtDuration(l.training_seconds), l.running ? `${l.running} running now` : "none running"),
    tile("Models", num(l.models), `${fmtSize(l.models_bytes)} on disk`),
    tile("Models serving", num(l.deployments_running), `${num(l.servers)} cloud server${l.servers === 1 ? "" : "s"} connected`));

  // ----- rentals
  const r = u.rentals;
  const rows = [];
  if (r.active.length) {
    rows.push(h("div", { class: "usage-head" }, h("b", {}, "Running now (billing)")));
    const el = h("div", { class: "utable" });
    table(el, [{ h: "Machine", w: "2fr" }, { h: "Running for", num: true }, { h: "Price", num: true }, { h: "Cost so far", num: true }],
      r.active.map((p) => [`${p.count}× ${p.gpu}`, hours(p.hours), p.cost_per_hour ? `$${p.cost_per_hour}/h` : "unknown", usd(p.cost)]));
    rows.push(el);
  } else {
    rows.push(h("p", { class: "hint" }, "No GPU machine is rented right now, so nothing is billing."));
  }
  rows.push(h("div", { class: "usage-head", style: "margin-top:14px" }, h("b", {}, "Finished rentals"),
    h("span", { class: "hint" }, `${r.past_count} rental${r.past_count === 1 ? "" : "s"} · ${hours(r.past_hours)} · about ${usd(r.past_cost)}`)));
  if (r.recent.length) {
    const el = h("div", { class: "utable" });
    table(el, [{ h: "Machine", w: "2fr" }, { h: "Ran for", num: true }, { h: "Cost", num: true }],
      r.recent.map((p) => [`${p.count}× ${p.gpu}`, hours(p.hours), usd(p.cost)]));
    rows.push(el);
  }
  rows.push(h("p", { class: "hint", style: "margin-top:10px" }, r.note + (r.unpriced ? ` ${r.unpriced} rental${r.unpriced === 1 ? " has" : "s have"} no recorded price.` : "")));
  $("rentals").replaceChildren(...rows);

  // ----- limits
  const lim = u.limits;
  $("limits").replaceChildren(
    h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "Tokens per message"), h("span"), h("span", { class: "val" }, levels.map(([, x]) => `${x.label} ${num(x.tokens)}`).join(" · "))),
    h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "Bee computers"), h("span"),
      h("span", { class: "val" }, Object.values(lim.sizes).map((x) => x.cpus ? `${x.label} ${x.cpus} CPU, ${x.memory_gb} GB` : x.label).join(" · "))),
    h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "Bees"), h("span"), h("span", { class: "val" }, `${b.limit} in all · ${b.max_depth} generations · ${b.max_children} children each`)),
    h("div", { class: "meter-row" }, h("span", { class: "lbl" }, "File upload"), h("span"), h("span", { class: "val" }, `${lim.upload_mb} MB each`)));
}

async function load() {
  try { render(await api("/api/usage")); } catch { $("stamp").textContent = "Couldn't reach the Hive."; }
}

$("viewToggle").addEventListener("click", () => { tableView = !tableView; if (last) { drawChart(last.tokens); $("viewToggle").textContent = tableView ? "Chart" : "Table"; } });

const client = new HiveClient();
client.addEventListener("state", (e) => { connLabel($("conn"), e.detail); if (e.detail === "online") load(); });
installTokenPrompt(() => client.reconnectNow());
client.connect();
load();
setInterval(load, 10_000);

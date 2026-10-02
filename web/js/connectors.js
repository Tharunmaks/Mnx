// Mnx Hive — Connector Hub: search every layer, connect MCP servers, set lanes, try tools.

import { ICONS, api, connLabel, fillIcons, h, installTokenPrompt } from "./shared.js";

const $ = (id) => document.getElementById(id);
fillIcons();
installTokenPrompt(() => load());

let phones = 0;

// ---------- connected list ----------
async function load() {
  let data;
  try {
    data = await api("/api/connectors");
    connLabel($("conn"), "online");
  } catch (e) {
    connLabel($("conn"), e.status === 401 ? "auth" : "offline");
    $("connected").replaceChildren(h("div", { class: "empty-note" }, "Can't reach the gateway."));
    return;
  }
  phones = data.phones;
  renderLayers(data.layers, data.connected);
  renderConnected(data.connected);
}

function renderLayers(layers, connected) {
  const box = $("layers");
  box.replaceChildren();
  for (const l of layers) {
    let live = "";
    if (l.n === 6) live = phones ? `Browser ready · ${phones} phone${phones > 1 ? "s" : ""}` : "Browser ready · no phone";
    else if (l.n >= 2 && l.n <= 4) {
      const n = connected.filter((c) => c.layer === l.n).length;
      live = n ? `${n} connected` : "";
    } else if (l.n === 1) live = "Planned";
    else if (l.n === 5) live = "Needs the Mnx brain";
    box.append(h("div", { class: "layer" },
      h("div", { class: "layer-n" }, l.n),
      h("div", {},
        h("div", { class: "layer-name" }, l.name),
        h("div", { class: "layer-reach" }, l.reach),
        live ? h("div", { class: "layer-live" }, live) : null)));
  }
}

function renderConnected(list) {
  $("connCount").textContent = list.length ? `· ${list.length}` : "";
  const box = $("connected");
  box.replaceChildren();
  if (!list.length) {
    box.append(h("div", { class: "empty-note" }, "Nothing connected yet. Search above, or add a server by URL."));
    return;
  }
  for (const c of list) box.append(connCard(c));
}

function connCard(c) {
  const ok = c.status === "ok";
  const tools = h("div", { class: "tool-list" });
  for (const t of c.tools) {
    const lane = h("select", { class: "lane", "aria-label": `Lane for ${t.name}`, onchange: async (e) => {
      try { await api(`/api/connectors/${encodeURIComponent(c.id)}/lane`, { method: "POST", body: { tool: t.name, lane: e.target.value } }); }
      catch (err) { alert(err.message); }
    } }, ["go", "ask", "you"].map((v) => h("option", { value: v, selected: t.lane === v }, { go: "Go", ask: "Ask", you: "You" }[v])));
    tools.append(h("div", { class: "tool-row" },
      h("div", { class: "meta" },
        h("div", { class: "name mono" }, t.name),
        h("div", { class: "skill" }, t.description || "")),
      lane,
      h("button", { class: "btn", type: "button", onclick: () => openTry(c, t) }, "Try")));
  }

  const details = h("details", { class: "conn-tools" },
    h("summary", {}, `${c.tools.length} tool${c.tools.length === 1 ? "" : "s"}`), tools);

  return h("div", { class: `card conn-card ${ok ? "" : "failed"}` },
    h("div", { class: "card-head" },
      h("span", { class: "svc" }, c.name.slice(0, 2).toUpperCase()),
      h("div", { style: "min-width:0" },
        h("div", {}, c.name),
        h("div", { class: "hint ellipsis" }, `Layer ${c.layer} · ${c.url}`)),
      h("span", { class: `badge ${ok ? "ok" : c.status === "checking" ? "wait" : "no"}` }, ok ? "Healthy" : c.status)),
    c.error ? h("p", { class: "card-text err" }, c.error) : null,
    c.tools.length ? details : null,
    h("div", { class: "card-actions" },
      h("button", { class: "btn", type: "button", onclick: async (e) => {
        e.target.disabled = true; e.target.textContent = "Testing…";
        try { await api(`/api/connectors/${encodeURIComponent(c.id)}/test`, { method: "POST" }); } catch (err) { alert(err.message); }
        load();
      } }, "Test"),
      h("button", { class: "btn danger", type: "button", onclick: async () => {
        if (!confirm(`Remove ${c.name}? Its saved login is deleted too.`)) return;
        try { await api(`/api/connectors/${encodeURIComponent(c.id)}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
        load();
      } }, "Remove")));
}

// ---------- search ----------
let searchSeq = 0;
async function search(q) {
  const seq = ++searchSeq;
  const box = $("results");
  if (!q.trim()) { box.replaceChildren(); return; }
  box.replaceChildren(h("div", { class: "empty-note" }, "Searching every layer…"));
  let r;
  try { r = await api(`/api/connectors/search?q=${encodeURIComponent(q)}`); }
  catch (e) { box.replaceChildren(h("div", { class: "empty-note err" }, e.message)); return; }
  if (seq !== searchSeq) return;

  box.replaceChildren();
  const group = (title, note, rows) => {
    if (!rows.length) return;
    box.append(h("div", { class: "section-title" }, h("h2", {}, title), note ? h("span", { class: "hint" }, note) : null),
      h("div", { class: "result-list" }, rows));
  };

  group("Connected", null, r.connected.map((c) => resultRow(c.name, `Layer ${c.layer} · ${c.tools.length} tools`, h("span", { class: "badge ok" }, "Connected"))));

  group("Browser Bee — any website", "Layer 6", r.browser.map((b) => resultRow(b.name, b.url,
    h("a", { class: "btn primary", href: `browser.html?url=${encodeURIComponent(b.url)}` }, "Open in browser"))));

  group("On your phone", "Layer 6", r.phone.map((a) => resultRow(a.label, a.package,
    h("button", { class: "btn primary", type: "button", onclick: (e) => openOnPhone(a, e.target) }, "Open on phone"))));
  if (!r.phone.length && !phones) {
    box.append(h("p", { class: "hint" }, "Pair your phone (Phone page) to search the apps installed on it too."));
  }

  group("MCP registry", r.registry_error || `${r.registry.length} found · Layer 2`, r.registry.map((s) => {
    const remote = s.remotes[0];
    const action = remote
      ? h("button", { class: "btn primary", type: "button", onclick: () => openAdd({ name: s.name, url: remote.url, transport: remote.type, headers: remote.headers, layer: 2, source: s.registry_name }) }, "Connect")
      : h("span", { class: "badge wait", title: "Runs locally only; needs the Docker sandbox" }, "Local only");
    const sub = h("span", {}, s.description || s.registry_name);
    if (s.repo) sub.append(" · ", h("a", { href: s.repo, target: "_blank", rel: "noopener noreferrer" }, "source"));
    return resultRow(s.name, sub, action);
  }));
  if (!r.registry.length && !r.registry_error) box.append(h("p", { class: "hint" }, "No MCP servers matched. The Browser Bee can still use the website."));

  group("Native Bees", "Layer 1 · built one by one", r.native.map((n) => resultRow(n.name, n.category, h("span", { class: "badge wait" }, "Planned"))));
}

function resultRow(title, sub, action) {
  return h("div", { class: "result-row" },
    h("div", { class: "meta" }, h("div", { class: "name" }, title), h("div", { class: "skill" }, sub)),
    action);
}

async function openOnPhone(app, btn) {
  btn.disabled = true;
  try {
    await api(`/api/phones/${encodeURIComponent(app.phone)}/cmd`, { method: "POST", body: { cmd: "open_app", args: { package: app.package } } });
    btn.textContent = "Opened";
  } catch (e) {
    alert(e.message);
    btn.disabled = false;
  }
}

let typing;
$("q").addEventListener("input", (e) => { clearTimeout(typing); typing = setTimeout(() => search(e.target.value), 350); });
$("searchForm").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(typing); search($("q").value); });

// ---------- add by URL ----------
let headerDefs = [];
function openAdd(pre = {}) {
  const f = $("addForm");
  f.reset();
  $("addErr").textContent = "";
  $("addTitle").textContent = pre.name ? `Connect ${pre.name}` : "Add an MCP server";
  f.name.value = pre.name || "";
  f.url.value = pre.url || "";
  f.transport.value = pre.transport === "sse" ? "sse" : "streamable-http";
  f.layer.value = String(pre.layer || 2);
  f.dataset.source = pre.source || "";
  headerDefs = pre.headers || [];
  const hf = $("headerFields");
  hf.replaceChildren();
  for (const hd of headerDefs) {
    hf.append(h("label", { class: "field" }, `${hd.name}${hd.isRequired ? " *" : ""}${hd.description ? " — " + hd.description : ""}`,
      h("input", { "data-header": hd.name, type: hd.isSecret ? "password" : "text", required: !!hd.isRequired, autocomplete: "off" })));
  }
  $("addHint").textContent = /\{[^}]+\}/.test(f.url.value)
    ? "Replace the {placeholders} in the URL with your own values."
    : "Any remote MCP server works here, including your Zapier MCP or Pipedream MCP URL.";
  $("addDlg").showModal();
}
$("addUrl").addEventListener("click", () => openAdd());

$("addForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  const f = e.target;
  if (/\{[^}]+\}/.test(f.url.value)) { $("addErr").textContent = "Fill in the {placeholders} in the URL first."; return; }
  const headers = {};
  for (const line of f.headers.value.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) headers[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  for (const input of $("headerFields").querySelectorAll("[data-header]")) {
    const v = input.value.trim();
    if (!v) continue;
    // Registry headers often look like "Bearer {api_key}": add the prefix when only the key was pasted.
    headers[input.dataset.header] = /^authorization$/i.test(input.dataset.header) && !/\s/.test(v) ? `Bearer ${v}` : v;
  }
  const btn = $("addSave");
  btn.disabled = true; btn.textContent = "Connecting…";
  try {
    const c = await api("/api/connectors", {
      method: "POST",
      body: { name: f.name.value.trim(), url: f.url.value.trim(), transport: f.transport.value, layer: Number(f.layer.value), headers, source: f.dataset.source || null },
    });
    $("addDlg").close();
    if (c.status !== "ok") alert(`Saved, but the server didn't answer: ${c.error}`);
    load();
  } catch (err) {
    $("addErr").textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Connect";
  }
});

// ---------- try a tool ----------
let trying = null;
function openTry(conn, tool) {
  trying = { conn, tool };
  $("tryTitle").textContent = `${conn.name} · ${tool.name}`;
  $("tryDesc").textContent = tool.description || "";
  const props = (tool.schema && tool.schema.properties) || {};
  const sample = Object.fromEntries(Object.entries(props).map(([k, v]) => [k, v.default ?? (v.type === "number" || v.type === "integer" ? 0 : v.type === "boolean" ? false : v.type === "array" ? [] : v.type === "object" ? {} : "")]));
  $("tryForm").args.value = JSON.stringify(sample, null, 2);
  $("tryOut").classList.add("hidden");
  $("tryDlg").showModal();
}
$("tryForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "run") return;
  e.preventDefault();
  const out = $("tryOut");
  out.classList.remove("hidden");
  let args;
  try { args = JSON.parse(e.target.args.value || "{}"); } catch { out.textContent = "Arguments must be valid JSON."; return; }
  if (trying.tool.lane !== "go" && !confirm(`Run ${trying.tool.name} now? It's in the ${trying.tool.lane === "you" ? "You" : "Ask"} lane.`)) return;
  out.textContent = "Running…";
  try {
    const r = await api(`/api/connectors/${encodeURIComponent(trying.conn.id)}/call`, { method: "POST", body: { tool: trying.tool.name, args } });
    out.textContent = (r.is_error ? "Tool reported an error:\n" : "") +
      r.content.map((p) => (p.type === "text" ? p.text : `[${p.type}]`)).join("\n\n");
  } catch (err) {
    out.textContent = err.message;
  }
});

load();
const pre = new URLSearchParams(location.search).get("q");
if (pre) { $("q").value = pre; search(pre); }

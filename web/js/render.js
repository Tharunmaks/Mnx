// Turns the event stream of one task into the "answer block" from the sketch:
// Bee chips, an L-shaped step tree, interactive cards and the final answer.
// The renderer is a pure function of the event list, so saved chats re-render exactly.

import { ICONS, h, logoSVG, toast, uploadFile } from "./shared.js";
import { filesCard, modelCard, runCard } from "./labcard.js";

const STEP_TYPES = new Set([
  "step", "thinking", "planning", "searching", "reading", "web", "opening",
  "fetching", "writing", "running", "booking", "waiting", "memory", "done", "error",
]);
const CARD_TYPES = new Set(["connect", "clarify", "confirm", "result", "lab_run", "lab_model", "lab_files"]);
const LIVE_CARDS = new Set(["result", "lab_run", "lab_model", "lab_files"]); // show information; nothing to answer

// ---------- events → model ----------
export function buildModel(events) {
  const items = new Map(); // id → item
  const roots = [];
  const model = { roots, answer: "", status: "running", error: null, resolved: new Map() };
  let auto = 0;

  const place = (item, parent) => {
    const p = parent && items.get(parent);
    if (p) p.children.push(item);
    else roots.push(item);
  };

  for (const ev of events) {
    const t = ev.type;
    if (t === "bee_created") {
      const item = { kind: "bee", id: ev.id || `b${auto++}`, bee: ev.bee, text: ev.text || "created", children: [] };
      items.set(item.id, item);
      place(item, ev.parent);
    } else if (STEP_TYPES.has(t)) {
      const id = ev.id || `s${auto++}`;
      const existing = ev.id && items.get(ev.id);
      if (existing) {
        Object.assign(existing, pickStep(ev, existing));
      } else {
        const item = { kind: "step", id, children: [], ...pickStep(ev, {}) };
        items.set(id, item);
        place(item, ev.parent);
      }
    } else if (CARD_TYPES.has(t)) {
      const id = ev.id || `c${auto++}`;
      const item = { kind: "card", id, card: ev, children: [] };
      items.set(id, item);
      place(item, ev.parent);
    } else if (t === "resolved" || t === "_action") {
      model.resolved.set(ev.ref, ev);
    } else if (t === "answer_delta") {
      model.answer += ev.text || "";
    } else if (t === "answer") {
      model.answer = ev.text || "";
    } else if (t === "tokens") {
      model.tokens = ev;
    } else if (t === "task_done") {
      model.status = "done";
    } else if (t === "task_error") {
      model.status = "error";
      model.error = ev.text || "Something went wrong.";
    }
  }

  // Once a task ends, nothing is "running" any more.
  if (model.status !== "running") {
    for (const it of items.values()) {
      if (it.kind === "step" && it.status === "running") it.status = model.status === "error" ? "error" : "done";
    }
  }
  return model;
}

function pickStep(ev, prev) {
  const type = ev.type === "step" ? (ev.icon || prev.type || "step") : ev.type;
  let status = ev.status || prev.status;
  if (!status) status = type === "error" ? "error" : type === "done" ? "done" : type === "waiting" ? "waiting" : "running";
  return {
    type,
    status,
    bee: ev.bee ?? prev.bee,
    text: ev.text ?? prev.text ?? "",
    detail: ev.detail ?? prev.detail,
  };
}

// ---------- model → DOM ----------
// cardCache keeps live card elements (and anything typed into them) across re-renders.
export function renderRun(model, { onAction, cardCache }) {
  const body = h("div", { class: "body" });

  if (model.roots.length) body.append(renderList(model.roots, model, onAction, cardCache, true));

  if (model.answer) body.append(h("div", { class: "answer" }, model.answer));

  if (model.status === "running" && !hasWaitingCard(model)) {
    body.append(h("div", { class: "pulse", "aria-label": "Mnx is working", html: ICONS.hex + ICONS.hex + ICONS.hex }));
  }
  if (model.status === "error") {
    body.append(h("div", { class: "step error" }, h("span", { class: "ico", html: ICONS.error }), h("span", { class: "txt" }, model.error)));
  }
  if (model.tokens) body.append(tokensLine(model.tokens));

  const wrap = h("div", { class: "msg-mnx" });
  wrap.innerHTML = logoSVG();
  wrap.append(body);
  return wrap;
}

// Token reading for one turn: what you wrote, what the Hive wrote, against the effort level's budget.
const EFFORT_LABELS = { low: "Low", med: "Med", high: "High", ultra: "Ultra", maxxxx: "Maxxxx" };
function tokensLine(t) {
  const budget = Math.max(1, t.budget || 1);
  const pct = Math.min(100, Math.round(100 * (t.total || 0) / budget));
  const level = EFFORT_LABELS[t.effort] || t.effort || "";
  const extra = t.effort === "maxxxx" ? " · no refusals · +20% tokens" : "";
  return h("div", { class: "tokens-line", title: `${(t.total || 0).toLocaleString()} of ${budget.toLocaleString()} tokens used` },
    h("span", {}, `Tokens · in ${(t.in || 0).toLocaleString()} · out ${(t.out || 0).toLocaleString()} · ${(t.total || 0).toLocaleString()} / ${budget.toLocaleString()}`),
    h("span", { class: "tokens-bar" }, h("i", { style: `width:${pct}%` })),
    h("span", { class: `effort-tag ${t.effort || ""}` }, `${level}${extra}`));
}

function hasWaitingCard(model) {
  const walk = (list) => list.some((it) =>
    (it.kind === "card" && !LIVE_CARDS.has(it.card.type) && !model.resolved.has(it.id)) || walk(it.children));
  return walk(model.roots);
}

function renderList(list, model, onAction, cardCache, top) {
  const box = h("div", { class: top ? "steps" : "children steps" });
  for (const it of list) {
    if (it.kind === "card") {
      box.append(renderCard(it, model, onAction, cardCache));
      continue;
    }
    const group = h("div", { class: "step-group" });
    group.append(it.kind === "bee" ? renderBee(it) : renderStep(it));
    if (it.children.length) group.append(renderList(it.children, model, onAction, cardCache, false));
    box.append(group);
  }
  return box;
}

function renderBee(it) {
  return h("div", { class: "step done" },
    h("span", { class: "ico", html: ICONS.hex }),
    h("span", { class: "txt bee-line" },
      h("span", { class: "bee-tag" }, `${it.bee} Bee`),
      h("span", {}, it.text)));
}

function renderStep(it) {
  const icon = ICONS[it.status === "error" ? "error" : it.type] || ICONS.planning;
  const txt = h("span", { class: "txt" });
  if (it.bee) txt.append(h("b", {}, `${it.bee} · `));
  txt.append(it.text);
  if (it.detail) txt.append(h("span", { class: "detail" }, ` — ${it.detail}`));
  return h("div", { class: `step ${it.status}` }, h("span", { class: "ico", html: icon }), txt);
}

// ---------- cards ----------
function renderCard(it, model, onAction, cardCache) {
  const resolved = model.resolved.get(it.id);
  let el = cardCache.get(it.id);
  if (!el) {
    el = buildCard(it.card, (action, values, label) => onAction(it.id, action, values, label));
    cardCache.set(it.id, el);
  }
  if (resolved && !el.classList.contains("resolved")) {
    el.classList.add("resolved");
    const badge = el.querySelector(".badge");
    if (badge) { badge.className = "badge ok"; badge.textContent = "Answered"; }
    const note = el.querySelector(".card-note") || el.appendChild(h("div", { class: "card-note" }));
    note.textContent = resolved.text || (resolved.label ? `You chose: ${resolved.label}` : "Answered");
  }
  return el;
}

function buildCard(c, act) {
  switch (c.type) {
    case "connect": return connectCard(c, act);
    case "clarify": return clarifyCard(c, act);
    case "confirm": return confirmCard(c, act);
    case "result": return resultCard(c);
    case "lab_run": return runCard(c.run);
    case "lab_model": return modelCard(c.model);
    case "lab_files": return filesCard(c);
  }
  return h("div");
}

function head(title, service, badge) {
  const el = h("div", { class: "card-head" });
  if (service) el.append(h("span", { class: "svc" }, service.slice(0, 2).toUpperCase()));
  el.append(h("span", {}, title));
  if (badge) el.append(h("span", { class: `badge ${badge[0]}` }, badge[1]));
  return el;
}

function actionButtons(actions, act, getValues) {
  const row = h("div", { class: "card-actions" });
  actions.forEach((a, i) => {
    row.append(h("button", {
      class: `btn ${a.style || (i === 0 ? "primary" : "")}`,
      type: "button",
      onclick: () => act(a.id, getValues ? getValues() : undefined, a.label),
    }, a.label));
  });
  return row;
}

// "Your account to MMT is not connected, please connect to continue  [Allow & permit] [Sign-in]"
function connectCard(c, act) {
  const service = c.service || "App";
  const actions = c.actions?.length ? c.actions : [
    { id: "allow", label: "Allow & permit" },
    { id: "signin", label: "Sign in" },
  ];
  return h("div", { class: "card attn" },
    head(c.title || `Connect ${service}`, service, ["wait", "Not connected"]),
    h("p", { class: "card-text" }, c.text || `Sign in and allow Mnx Hive access to ${service} to continue.`),
    actionButtons(actions, act));
}

// "type the clarification [where i.e time, date and from where to where]"
// Fields: text / number / date / time / textarea / select (options as strings or {value, label})
// / file (uploaded when you press the first button; the answer carries the upload id).
// A field with show_if: {other_field: value} only appears while that select has that value.
function clarifyCard(c, act) {
  const inputs = [];
  const labels = [];
  const fields = h("div", { class: "fields" });
  for (const f of c.fields || []) {
    let input;
    if (f.input === "select") {
      input = h("select", { name: f.name }, (f.options || []).map((o) => {
        const value = typeof o === "object" ? o.value : o;
        return h("option", { value, selected: f.value != null && String(f.value) === String(value) }, typeof o === "object" ? o.label : o);
      }));
    } else if (f.input === "textarea") {
      input = h("textarea", { name: f.name, rows: 2, placeholder: f.placeholder || "" }, f.value || "");
    } else if (f.input === "file") {
      input = h("input", { name: f.name, type: "file", accept: f.accept || null });
    } else {
      input = h("input", { name: f.name, type: f.input || "text", placeholder: f.placeholder || "", value: f.value ?? "" });
    }
    if (f.required) input.required = true;
    inputs.push(input);
    const long = f.input === "file" || String(f.value ?? "").length > 28
      || (f.options || []).some((o) => String(typeof o === "object" ? o.label : o).length > 28);
    const label = h("label", { class: `field${long ? " wide" : ""}` }, f.label || f.name, input);
    labels.push([label, f.show_if]);
    fields.append(label);
  }
  const byName = Object.fromEntries(inputs.map((i) => [i.name, i]));
  const visible = (cond) => !cond || Object.entries(cond).every(([k, v]) => byName[k] && byName[k].value === String(v));
  const sync = () => labels.forEach(([label, cond]) => label.classList.toggle("hidden", !visible(cond)));
  inputs.forEach((i) => i.addEventListener("change", sync));
  sync();
  const shown = () => inputs.filter((i) => !i.closest(".hidden"));
  const actions = c.actions?.length ? c.actions : [{ id: "submit", label: c.submit || "Send answers" }];
  const card = h("div", { class: "card attn" },
    head(c.title || "A few details needed", null, ["wait", "Waiting for you"]),
    c.text ? h("p", { class: "card-text pre" }, c.text) : null,
    fields,
    actionButtons(actions, async (id, _v, label) => {
      const primary = id === actions[0].id;
      const missing = shown().find((i) => i.required && (i.type === "file" ? !i.files.length : !i.value.trim()));
      if (missing && primary) { missing.focus(); return; }
      const values = {};
      for (const i of shown()) {
        if (i.type !== "file") { values[i.name] = i.value; continue; }
        if (!primary || !i.files.length) continue;
        const btns = card.querySelectorAll(".card-actions .btn");
        btns.forEach((b) => { b.disabled = true; });
        try {
          const up = await uploadFile(i.files[0], (pct) => { btns[0].textContent = `Uploading ${pct}%`; });
          values[i.name] = up.id;
          btns[0].textContent = actions[0].label;
        } catch (e) {
          toast(e.message, "error");
          btns.forEach((b) => { b.disabled = false; });
          btns[0].textContent = actions[0].label;
          return;
        }
      }
      act(id, values, label);
    }));
  return card;
}

// "Asking user to confirm  [Allow & permit]"
function confirmCard(c, act) {
  const actions = c.actions?.length ? c.actions : [
    { id: "allow", label: "Allow & permit" },
    { id: "cancel", label: "Cancel", style: "danger" },
  ];
  return h("div", { class: "card attn" },
    head(c.title || "Confirm to continue", c.service, ["wait", "Needs your OK"]),
    c.text ? h("p", { class: "card-text" }, c.text) : null,
    details(c.details),
    actionButtons(actions, act));
}

// "Successfully booked  [details, driver details, price, etc.]"
function resultCard(c) {
  const ok = c.status !== "failed";
  return h("div", { class: `card ${ok ? "success" : "failed"}` },
    head(c.title || (ok ? "Done" : "Failed"), c.service, ok ? ["ok", c.badge || "Success"] : ["no", c.badge || "Failed"]),
    c.text ? h("p", { class: "card-text" }, c.text) : null,
    details(c.details),
    c.note ? h("div", { class: "card-note" }, c.note) : null);
}

function details(d) {
  if (!d) return null;
  const rows = Array.isArray(d) ? d : Object.entries(d);
  if (!rows.length) return null;
  const dl = h("dl", { class: "details" });
  for (const [k, v] of rows) dl.append(h("dt", {}, k), h("dd", {}, v));
  return dl;
}

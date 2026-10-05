// Live Cloud Bee cards inside the chat: a training run that updates itself, and a
// trained model you can start and talk to right in the conversation.
// They read live state from the server, so they stay correct after a reload.

import { api, fmtDuration, h, toast } from "./shared.js";
import { drawCharts, playgroundFor } from "./labui.js";

const ACTIVE = new Set(["queued", "preparing", "running"]);
const STAGE = { preparing: "Getting the server ready", installing: "Installing PyTorch and friends", training: "Training" };

// Poll while the card is on screen; stop once it's gone for good.
function poll(el, fn, every) {
  let seen = false, timer = null;
  const tick = async () => {
    if (el.isConnected) seen = true;
    else if (seen) return; // card left the page (chat switched); stop
    let again = true;
    try { again = await fn(); } catch { /* keep trying */ }
    if (again !== false) timer = setTimeout(tick, typeof again === "number" ? again : every);
  };
  tick();
  return () => clearTimeout(timer);
}

function headRow(title, badge) {
  return h("div", { class: "card-head" }, h("span", { class: "svc" }, "AI"), h("span", { style: "flex:1;min-width:0" }, title), badge);
}

export function runCard(runId) {
  const title = h("span", {}, "Training");
  const badge = h("span", { class: "badge wait" }, "…");
  const line = h("p", { class: "card-text" }, "Loading…");
  const charts = h("div", { class: "charts compact" });
  const actions = h("div", { class: "card-actions" });
  const modelSlot = h("div");
  const card = h("div", { class: "card lab-card" }, headRow(title, badge), line, charts, actions, modelSlot);
  let lastCount = -1, modelShown = null;

  poll(card, async () => {
    const r = await api(`/api/lab/runs/${runId}`);
    title.textContent = r.name;
    const active = ACTIVE.has(r.status);
    const last = r.metrics[r.metrics.length - 1];
    badge.className = `badge ${r.status === "succeeded" ? "ok" : r.status === "failed" ? "no" : "wait"}`;
    badge.textContent = active ? "Live" : r.status[0].toUpperCase() + r.status.slice(1);
    const took = r.started ? fmtDuration((r.ended || Date.now() / 1000) - r.started) : "";
    if (active) {
      const prog = last ? [last.epoch != null ? `epoch ${last.epoch}` : null, last.step != null ? `step ${last.step}` : null,
        last.loss != null ? `loss ${Number(last.loss).toFixed(4)}` : null].filter(Boolean).join(" · ") : "";
      const detail = r.stage !== "training" && r.last_line ? ` · ${r.last_line.slice(0, 90)}` : "";
      line.textContent = `${STAGE[r.stage] || "Starting"}${prog ? ` · ${prog}` : ""}${detail} · ${took}`;
    } else if (r.status === "succeeded") {
      line.textContent = `Done in ${took}.${r.result?.examples ? ` Learned from ${r.result.examples} examples.` : ""}`;
      if (r.result && (r.result.perplexity != null || r.result.samples) && !card.querySelector(".eval-box")) {
        card.insertBefore(evalBox(r.result), actions);
      }
    } else {
      line.textContent = r.error ? `${r.status === "stopped" ? "Stopped" : "Failed"}: ${r.error}` : `${r.status} after ${took}`;
      line.classList.toggle("err", r.status === "failed");
    }
    if (r.metric_count !== lastCount) {
      lastCount = r.metric_count;
      drawCharts(charts, r.metrics.map(({ step, loss }) => ({ step, loss })).filter((p) => p.loss != null));
      if (!r.metric_count) charts.replaceChildren();
    }
    actions.replaceChildren(...[
      active ? h("button", { class: "btn", type: "button", onclick: async (e) => {
        if (!confirm("Stop this training?")) return;
        e.target.disabled = true;
        try { await api(`/api/lab/runs/${runId}/stop`, { method: "POST" }); } catch (err) { toast(err.message, "error"); }
      } }, "Stop") : null,
      h("a", { class: "btn", href: "lab.html#runs" }, "Full log in the Lab"),
    ].filter(Boolean));
    if (r.model && modelShown !== r.model) {
      modelShown = r.model;
      modelSlot.replaceChildren(modelSection(r.model));
    }
    return active ? 3000 : false;
  }, 3000);
  return card;
}

// Eval Bee's report: how well the model predicts held-out text, and what it writes.
function evalBox(res) {
  const box = h("div", { class: "eval-box" }, h("div", { class: "eval-head" }, h("span", { class: "bee-tag" }, "Eval Bee"),
    h("span", {}, res.perplexity != null ? `perplexity ${res.perplexity} on held-out text (lower is better)` : "checked the model")));
  if (res.params_text) box.append(h("div", { class: "hint" }, `${res.params_text} parameters · trained on ${fmtTokens(res.tokens)} tokens in ${res.minutes} min`));
  for (const smp of res.samples || []) {
    box.append(h("div", { class: "sample" }, h("b", {}, smp.prompt), " ", h("span", {}, smp.text.slice(smp.prompt.length) || smp.text)));
  }
  return box;
}
function fmtTokens(n) {
  if (!n) return "0";
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}K` : String(n);
}

export function modelCard(modelId) {
  return h("div", { class: "card lab-card" }, modelSection(modelId));
}

function modelSection(modelId) {
  const box = h("div", { class: "model-section" });
  const status = h("p", { class: "card-text" }, "Checking the model…");
  const actions = h("div", { class: "card-actions" });
  const playSlot = h("div");
  box.append(status, actions, playSlot);
  let starting = false;

  poll(box, async () => {
    const { model: m, deployment: d, gpu_local } = await api(`/api/lab/models/${modelId}`);
    const state = d ? d.status : null;
    if (m.kind === "blueprint") {
      const a = m.arch || {}, e = m.estimates || {}, ref = e.reference || {};
      status.replaceChildren(h("b", {}, `${m.name}`), ` — designed, not trained. ${a.layers} layers, width ${a.hidden}, ${a.heads} heads, vocabulary ${a.vocab?.toLocaleString()}.`,
        h("div", { class: "hint" }, `Weights ${e.weights || "?"} · training memory ${e.train_memory_gpu || "?"} · a good setup is ${ref.count}× ${ref.gpu} for ${ref.time} (~${ref.cost}).`));
      actions.replaceChildren(h("a", { class: "btn", href: "lab.html#models" }, "See it in the Lab"));
      return false;
    }
    if (!m.servable) {
      status.textContent = "This model can't run as an API here (no serve.py), but you can download it from the Lab.";
      actions.replaceChildren();
      return false;
    }
    if (state === "running") {
      status.replaceChildren(h("b", {}, `${m.name} is ready.`), " Talk to it below; it keeps running 24/7 until you stop it.");
      actions.replaceChildren(h("button", { class: "btn", type: "button", onclick: async () => {
        await api(`/api/lab/models/${modelId}/undeploy`, { method: "POST" }).catch((err) => toast(err.message, "error"));
      } }, "Stop it"));
      if (!playSlot.firstChild) playSlot.replaceChildren(playgroundFor(m));
      return 8000;
    }
    playSlot.replaceChildren();
    if (state === "starting" || starting) {
      status.textContent = m.kind === "llm"
        ? "Starting… installing PyTorch and loading the model can take a few minutes the first time."
        : "Starting…";
      actions.replaceChildren();
      return 3000;
    }
    status.textContent = state === "failed" ? `It stopped: ${d.error || "see the server log in the Lab"}` : `${m.name} is trained. Start it to talk to it here.`;
    actions.replaceChildren(h("button", { class: "btn primary", type: "button", onclick: async (e) => {
      e.target.disabled = true;
      starting = true;
      try { await api(`/api/lab/models/${modelId}/deploy`, { method: "POST", body: { gpu: !!gpu_local } }); }
      catch (err) { toast(err.message, "error"); e.target.disabled = false; }
      starting = false;
    } }, state === "failed" ? "Try again" : "Run it"));
    return 4000;
  }, 4000);
  return box;
}

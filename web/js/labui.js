// Mnx Hive — shared Model Lab pieces: metric charts and model playgrounds.
// Used by the Lab page and by the live cards the Cloud Bee posts in chat.

import { effortInfo, gatewayBase, getSettings, h, toast } from "./shared.js";

const playgrounds = new Map(); // model id → element, so a chat with the model survives redraws

export function forgetPlayground(id) {
  playgrounds.delete(id);
}

// Small line charts, one per metric (loss and accuracy have different scales).
export function drawCharts(box, points) {
  box.replaceChildren();
  if (!points?.length) { box.append(h("p", { class: "hint" }, "Charts appear when training reports its first numbers.")); return; }
  const xKey = "step" in points[0] ? "step" : null;
  const keysAll = [...new Set(points.flatMap((p) => Object.keys(p)))].filter((k) => !["step", "epoch", "sec", "lr"].includes(k) && points.some((p) => typeof p[k] === "number"));
  for (const k of keysAll) {
    const pts = points.map((p, i) => [xKey ? p[xKey] : i, p[k]]).filter(([, y]) => typeof y === "number" && isFinite(y));
    if (!pts.length) continue;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 300, H = 90, sx = (x) => (x1 === x0 ? W / 2 : ((x - x0) / (x1 - x0)) * W), sy = (y) => (y1 === y0 ? H / 2 : H - ((y - y0) / (y1 - y0)) * (H - 8) - 4);
    const dpath = pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
    const good = /acc|f1|r2|score/.test(k);
    const last = ys[ys.length - 1];
    const svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><path d="${dpath}" fill="none" stroke="${good ? "#4c9bff" : "#f5a623"}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
    box.append(h("div", { class: "chart" },
      h("div", { class: "chart-head" }, h("span", {}, k.replace(/_/g, " ")), h("b", {}, Number.isInteger(last) ? last : last.toFixed(4))),
      h("div", { class: "chart-svg", html: svg, role: "img", "aria-label": `${k} from ${y0} to ${y1} over ${pts.length} points` }),
      h("div", { class: "chart-axis" }, h("span", {}, `${xKey || "#"} ${x0}`), h("span", {}, `min ${+y0.toFixed(4)} · max ${+y1.toFixed(4)}`), h("span", {}, String(x1)))));
  }
}

export function endpoint(m) {
  return `${gatewayBase()}/api/lab/serve/${m.id}/`;
}

export function playgroundFor(m) {
  let el = playgrounds.get(m.id);
  if (el) return el;
  const url = endpoint(m);
  const path = m.playground === "chat" ? "v1/chat/completions" : "predict";
  const example = m.playground === "chat"
    ? `curl ${url}${path} \\\n  -H "Authorization: Bearer $MNX_TOKEN" -H "Content-Type: application/json" \\\n  -d '{"messages":[{"role":"user","content":"Hi"}]}'`
    : `curl ${url}${path} \\\n  -H "Authorization: Bearer $MNX_TOKEN" -H "Content-Type: application/json" \\\n  -d '${m.playground === "classify" ? '{"text":"I love it"}' : '{"row":{}}'}'`;
  const body = h("div", { class: "playground" });
  el = h("div", { class: "play-wrap" },
    h("div", { class: "play-head" }, h("b", {}, "Try it"), h("span", { class: "hint" }, m.playground === "chat" ? "OpenAI-compatible API" : "JSON API")),
    body,
    h("details", { class: "api-box" }, h("summary", { class: "hint" }, "Use it from other apps"),
      h("pre", { class: "result-box" }, example),
      h("p", { class: "hint" }, "Send your Hive access token as the Bearer token. Chat models work with any OpenAI-compatible client: set the base URL to ", h("code", {}, `${url}v1`), ".")));
  if (m.playground === "classify") buildClassify(body, m);
  else if (m.playground === "chat") buildChat(body, m);
  else buildJson(body, m);
  playgrounds.set(m.id, el);
  return el;
}

async function call(m, path, payload) {
  const res = await fetch(`${endpoint(m)}${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${getSettings().token}`, "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(out.error || out.detail || res.statusText);
  return out;
}

function buildClassify(box, m) {
  const input = h("textarea", { class: "input", rows: 2, placeholder: "Type a sentence to classify…" });
  const out = h("div", { class: "scores" });
  const go = async () => {
    if (!input.value.trim()) return;
    out.replaceChildren(h("span", { class: "hint" }, "Thinking…"));
    try {
      const r = await call(m, "predict", { text: input.value });
      out.replaceChildren(h("div", { class: "pred" }, "→ ", h("b", {}, r.label)),
        ...Object.entries(r.scores).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
          h("div", { class: "score" }, h("span", {}, k), h("div", { class: "bar" }, h("i", { style: `width:${(v * 100).toFixed(1)}%` })), h("span", { class: "hint" }, `${(v * 100).toFixed(1)}%`))));
    } catch (e) { out.replaceChildren(h("span", { class: "err" }, e.message)); }
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } });
  box.append(input, h("div", { class: "row" }, h("button", { class: "btn primary", type: "button", onclick: go }, "Predict")), out);
}

function buildJson(box, m) {
  const input = h("textarea", { class: "input mono", rows: 6 }, '{\n  "row": {}\n}');
  const out = h("pre", { class: "result-box" }, "The answer shows here.");
  call(m, "info").then((info) => { if (info.example) input.value = JSON.stringify(info.example, null, 2); }).catch(() => {});
  const go = async () => {
    let payload;
    try { payload = JSON.parse(input.value || "{}"); } catch { out.textContent = "That isn't valid JSON."; return; }
    out.textContent = "Thinking…";
    try { out.textContent = JSON.stringify(await call(m, "predict", payload), null, 2); } catch (e) { out.textContent = e.message; }
  };
  box.append(input, h("div", { class: "row" }, h("button", { class: "btn primary", type: "button", onclick: go }, "Predict")), out);
}

function buildChat(box, m) {
  const messages = [];
  const log = h("div", { class: "chat-log" }, h("p", { class: "hint" }, "Say something to your model."));
  const input = h("textarea", { class: "input", rows: 2, placeholder: "Message your model…" });
  const draw = () => log.replaceChildren(...messages.map((x) => h("div", { class: `chat-msg ${x.role}` }, x.content)));
  const go = async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    messages.push({ role: "user", content: text });
    draw();
    const pending = h("div", { class: "chat-msg assistant hint" }, "…");
    log.append(pending);
    try {
      const r = await call(m, "v1/chat/completions", { messages, max_tokens: effortInfo(getSettings().effort).reply, temperature: 0.7 });
      messages.push({ role: "assistant", content: r.choices?.[0]?.message?.content ?? "" });
    } catch (e) {
      messages.pop();
      toast(e.message, "error");
    }
    draw();
    log.scrollTop = log.scrollHeight;
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } });
  box.append(log, h("div", { class: "row" }, input, h("button", { class: "btn primary", type: "button", onclick: go }, "Send"),
    h("button", { class: "btn", type: "button", onclick: () => { messages.length = 0; draw(); } }, "Clear")));
}

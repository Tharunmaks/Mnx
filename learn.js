// Learning from chats. Mnx keeps, on this device only:
//  • replies you mark 👍 (the model's own good answers, exactly as it wrote them),
//  • 👎 replies you correct (your better answer replaces Mnx's),
//  • chats with the online models (Fable, Opus, Sonnet) as "teacher" examples,
//    converted into the local model's format, when "Learn from my chats" is on.
// They're written to data/learned.jsonl in the training format, and
// `npm run learn` adds them to the next training run.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "data");
export const LEARNED_FILE = () => process.env.MNX_LEARNED_FILE || path.join(DIR, "learned.jsonl");

// Python json.dumps style, to match the training data exactly.
export function pyJson(v) {
  if (Array.isArray(v)) return `[${v.map(pyJson).join(", ")}]`;
  if (v && typeof v === "object") return `{${Object.entries(v).filter(([, x]) => x !== undefined).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(", ")}}`;
  return JSON.stringify(v ?? null);
}

/* ───────────── recent traces (so 👍 can find what the model saw) ───────────── */
const traces = new Map();
export function rememberTrace(trace) {
  const id = crypto.randomBytes(9).toString("base64url");
  traces.set(id, { ...trace, at: Date.now() });
  while (traces.size > 300) traces.delete(traces.keys().next().value);
  return id;
}

const lastUserIndex = (msgs) => {
  for (let i = msgs.length - 1; i >= 0; i--) if (msgs[i].role === "user" && !String(msgs[i].content).startsWith("<tool_response>")) return i;
  return -1;
};

/* ───────────── teacher chats: Claude format → local format ───────────── */
function shortThink(thinking) {
  const t = String(thinking || "").trim();
  const head = t.match(/\*\*([^*\n]{3,80})\*\*/)?.[1];
  const body = t.replace(/\*\*[^*\n]{3,80}\*\*\s*/, "").replace(/\s+/g, " ").trim();
  const first = (body.match(/^.{20,280}?[.!?](\s|$)/)?.[0] || body.slice(0, 220)).trim();
  return `<think>**${head || "Answering"}**\n${first || "I'll answer directly."}</think>\n`;
}
const textOf = (content) =>
  typeof content === "string"
    ? content
    : (content || []).map((b) => (b.type === "text" ? b.text : b.type === "image" ? "[The user attached an image, which this local model can't see.]" : b.type === "document" ? "[The user attached a PDF, which this local model can't read.]" : "")).filter(Boolean).join("\n\n");

/**
 * Convert a finished chat with an online model into the local model's
 * training format. Returns null when it can't be taught faithfully (e.g. it
 * used web tools the local model doesn't have, or connectors).
 */
export function teacherToLocal(messages, systemText, knownTools) {
  const isRealUser = (m) => m.role === "user" && (typeof m.content === "string" || m.content.some((b) => b.type === "text"));
  let last = -1;
  for (let i = messages.length - 1; i >= 0; i--) if (isRealUser(messages[i])) {
    last = i;
    break;
  }
  if (last < 0 || last === messages.length - 1) return null;
  const out = [{ role: "system", content: systemText }];
  // Earlier turns: plain text, final answers only (as Mnx sends them).
  for (const m of messages.slice(0, last)) {
    const text = m.role === "user" && !isRealUser(m) ? "" : textOf(m.content);
    if (!text) continue;
    const prev = out[out.length - 1];
    if (prev.role === m.role) prev.content += `\n\n${text}`;
    else if (m.role === "user" || prev.role === "user") out.push({ role: m.role, content: text });
  }
  if (out[out.length - 1].role === "user") out.pop(); // keep strict alternation
  const train_from = out.length;
  out.push({ role: "user", content: textOf(messages[last].content) });
  for (const m of messages.slice(last + 1)) {
    if (m.role === "assistant") {
      const blocks = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content;
      if (blocks.some((b) => b.type === "server_tool_use" || b.type === "mcp_tool_use" || /_tool_result$/.test(b.type))) return null;
      const thinking = blocks.filter((b) => b.type === "thinking").map((b) => b.thinking).join("\n");
      const text = blocks.filter((b) => b.type === "text").map((b) => b.text).join("");
      const calls = blocks.filter((b) => b.type === "tool_use");
      if (calls.some((c) => !knownTools.includes(c.name))) return null;
      out.push({ role: "assistant", content: shortThink(thinking) + text + calls.map((c) => `<tool_call>\n${pyJson({ name: c.name, arguments: c.input })}\n</tool_call>`).join("\n") });
    } else {
      const results = (typeof m.content === "string" ? [] : m.content).filter((b) => b.type === "tool_result");
      if (!results.length) return null;
      const fmt = (c) => {
        try {
          return pyJson(JSON.parse(c));
        } catch {
          return typeof c === "string" ? c : pyJson(c);
        }
      };
      out.push({ role: "user", content: results.map((r) => `<tool_response>\n${fmt(r.content)}\n</tool_response>`).join("\n") });
    }
  }
  if (out[out.length - 1].role !== "assistant" || !/<\/think>\n\S/.test(out[out.length - 1].content)) return null;
  return { messages: out, train_from };
}

/* ───────────── saving ───────────── */
function append(row) {
  fs.mkdirSync(path.dirname(LEARNED_FILE()), { recursive: true });
  fs.appendFileSync(LEARNED_FILE(), `${JSON.stringify(row)}\n`);
}

export function saveTeacher(trace) {
  append({ category: "learned_teacher", source: trace.model || "online", messages: trace.messages, train_from: trace.train_from, at: new Date().toISOString() });
}

/** 👍 / 👎 on a reply. Returns what happened, for the UI. */
export function saveFeedback(traceId, rating, correction) {
  const t = traces.get(traceId);
  if (!t) return { ok: false, error: "That reply is too old to learn from (Mnx was restarted). Try again on a new reply." };
  if (rating === "up") {
    append({ category: "learned_good", source: t.source, messages: t.messages, train_from: t.train_from, at: new Date().toISOString() });
    return { ok: true, learned: true };
  }
  const fix = String(correction || "").trim();
  if (!fix) {
    // Kept for review only; never trained on.
    fs.mkdirSync(path.dirname(LEARNED_FILE()), { recursive: true });
    fs.appendFileSync(LEARNED_FILE().replace(/\.jsonl$/, "-bad.jsonl"), `${JSON.stringify({ messages: t.messages, at: new Date().toISOString() })}\n`);
    return { ok: true, learned: false };
  }
  // The user's corrected answer replaces Mnx's final answer.
  const msgs = t.messages.slice();
  const lastUser = lastUserIndex(msgs);
  const tail = msgs.slice(lastUser + 1);
  const lastTool = tail.map((m) => m.role).lastIndexOf("user");
  const kept = msgs.slice(0, lastUser + 1 + (lastTool >= 0 ? lastTool + 1 : 0));
  kept.push({ role: "assistant", content: `<think>**Answering**\nI'll give the answer the user expects.</think>\n${fix}` });
  append({ category: "learned_corrected", source: t.source, messages: kept, train_from: t.train_from, at: new Date().toISOString() });
  return { ok: true, learned: true };
}

export function learnStats() {
  const out = { total: 0, good: 0, corrected: 0, teacher: 0, file: LEARNED_FILE() };
  try {
    for (const line of fs.readFileSync(LEARNED_FILE(), "utf8").split("\n")) {
      if (!line.trim()) continue;
      out.total++;
      const c = JSON.parse(line).category;
      if (c === "learned_good") out.good++;
      else if (c === "learned_corrected") out.corrected++;
      else if (c === "learned_teacher") out.teacher++;
    }
  } catch {
    /* nothing learned yet */
  }
  return out;
}
export { lastUserIndex };

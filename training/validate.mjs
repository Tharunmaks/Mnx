// Checks every training conversation with Mnx's own runtime parsers, so the
// model is only ever trained on output Mnx can actually understand.
//   node training/validate.mjs [training/data/train.jsonl]
import fs from "node:fs";
import readline from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ThinkSplitter, ToolCallScanner, parseToolCall, LOCAL_TOOLS, STABLE_PROMPT } from "../local.js";
import { MORE_TOOLS } from "../tools-more.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const file = process.argv[2] || path.join(here, "data/train.jsonl");
const specs = Object.fromEntries([
  ...LOCAL_TOOLS.map((t) => [t.function.name, t.function.parameters]),
  ...MORE_TOOLS.map((t) => [t.name, { properties: t.params, required: t.required }]),
]);

// Parse an assistant message exactly as Mnx does, feeding it in small chunks.
function parseAssistant(text) {
  let think = "";
  let answer = "";
  const calls = [];
  let cur = null;
  const scanner = new ToolCallScanner({
    onText: (t) => (answer += t),
    onCallStart: () => (cur = ""),
    onCallChunk: (t) => (cur += t),
    onCallEnd: () => calls.push(cur),
  });
  const splitter = new ThinkSplitter((t) => t && (think += t), (t) => scanner.push(t));
  for (let i = 0; i < text.length; i += 7) splitter.push(text.slice(i, i + 7));
  splitter.end();
  scanner.end();
  return { think, answer, calls };
}

function checkArgs(name, args) {
  const spec = specs[name];
  if (!spec) return `unknown tool ${name}`;
  for (const req of spec.required || []) if (args[req] === undefined || args[req] === "") return `${name}: missing ${req}`;
  for (const [k, v] of Object.entries(args)) {
    const p = spec.properties[k];
    if (!p) return `${name}: unexpected argument ${k}`;
    if (p.enum && !p.enum.includes(v)) return `${name}.${k}: ${v} not in ${p.enum}`;
    if (p.type === "string" && typeof v !== "string") return `${name}.${k} must be a string`;
    if (p.type === "array" && !Array.isArray(v)) return `${name}.${k} must be an array`;
    if (p.type === "integer" && !Number.isInteger(v)) return `${name}.${k} must be an integer`;
  }
  if (name === "create_document") {
    const list = args.format === "pptx" ? args.slides : args.sections;
    if (!Array.isArray(list) || !list.length) return `create_document: ${args.format} needs ${args.format === "pptx" ? "slides" : "sections"}`;
  }
  return null;
}

export function validateConversation(ex) {
  const m = ex.messages;
  if (m[0]?.role !== "system" || !m[0].content.startsWith(STABLE_PROMPT)) return "system prompt doesn't match local.js (re-run export_prompt.mjs)";
  // Earlier turns (follow-ups) are plain user/assistant text, exactly as Mnx
  // sends history. Only the turn after `train_from` is learned.
  const from = ex.train_from ?? 1;
  for (let i = 1; i <= from; i++) {
    const want = (i - from) % 2 === 0 ? "user" : "assistant";
    if (m[i]?.role !== want) return `history message ${i} should be ${want}`;
    if (want === "assistant" && /<\/?(think|tool_call)>/.test(m[i].content)) return `history message ${i} must be a plain answer`;
  }
  for (let i = from + 1; i < m.length; i++) {
    const msg = m[i];
    const expectAssistant = (i - from) % 2 === 1;
    if (msg.role !== (expectAssistant ? "assistant" : "user")) return `message ${i} should be ${expectAssistant ? "assistant" : "user"}`;
    if (!expectAssistant) {
      if (!/^<tool_response>\n[\s\S]+\n<\/tool_response>$/.test(msg.content)) return `message ${i}: tool result not wrapped in <tool_response>`;
      continue;
    }
    const { think, answer, calls } = parseAssistant(msg.content);
    if (!think.trim()) return `message ${i}: missing <think>`;
    const last = i === m.length - 1;
    if (last) {
      if (calls.length) return "final message must not call a tool";
      if (!answer.trim()) return "final message has no answer";
    } else {
      if (calls.length !== 1) return `message ${i}: expected exactly one tool call, got ${calls.length}`;
      const parsed = parseToolCall(calls[0]);
      if (!parsed) return `message ${i}: tool call is not valid JSON`;
      const err = checkArgs(parsed.name, parsed.input);
      if (err) return `message ${i}: ${err}`;
      if (m[i + 1]?.role !== "user") return `message ${i}: tool call without a result`;
    }
  }
  if (m[m.length - 1].role !== "assistant") return "conversation must end with the assistant";
  return null;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  // Stream line by line: a 100,000-conversation file is far too big to load at once.
  let bad = 0;
  let total = 0;
  const byCat = {};
  const lines = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    const n = total++;
    const ex = JSON.parse(line);
    byCat[ex.category] = (byCat[ex.category] || 0) + 1;
    const err = validateConversation(ex);
    if (err) {
      bad++;
      if (bad <= 10) console.log(`line ${n + 1} (${ex.category}): ${err}`);
    }
  }
  console.log(`${total - bad}/${total} conversations valid`);
  console.log(Object.entries(byCat).map(([k, v]) => `${k} ${v}`).join(", "));
  process.exit(bad ? 1 : 0);
}

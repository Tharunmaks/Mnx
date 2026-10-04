// Turns what Mnx learned from your chats (data/learned.jsonl) into training
// data for the next fine-tune, and optionally uploads it so the Colab
// notebook picks it up automatically.
//
//   npm run learn                  → checks it and writes training/data/extra.jsonl
//   HF_TOKEN=hf_... npm run learn -- --upload
//                                  → also uploads it to your private Hugging Face
//                                    dataset (default: <your username>/mnx-learned)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STABLE_PROMPT, ThinkSplitter, ToolCallScanner, parseToolCall, LOCAL_TOOL_NAMES } from "../local.js";
import { LEARNED_FILE } from "../learn.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const outFile = path.join(root, "training/data/extra.jsonl");

function calls(text) {
  const found = [];
  let cur = null;
  const scanner = new ToolCallScanner({ onText: () => {}, onCallStart: () => (cur = ""), onCallChunk: (t) => (cur += t), onCallEnd: () => found.push(cur) });
  const splitter = new ThinkSplitter(() => {}, (t) => scanner.push(t));
  splitter.push(text);
  splitter.end();
  scanner.end();
  return found;
}

// Older examples may have an older system prompt: bring them up to date.
function refreshSystem(m) {
  const tail = m.content.slice(m.content.indexOf("\n\nToday's date is"));
  return { role: "system", content: STABLE_PROMPT + (tail.startsWith("\n\nToday's date is") ? tail : `\n\nToday's date is ${new Date().toISOString().slice(0, 10)}.`) };
}

function check(row) {
  const m = row.messages;
  if (!Array.isArray(m) || m[0]?.role !== "system" || m.length < 3) return "not a conversation";
  if (m[m.length - 1].role !== "assistant") return "doesn't end with an answer";
  for (let i = row.train_from + 1; i < m.length; i++) {
    if (m[i].role !== "assistant") continue;
    for (const c of calls(m[i].content)) {
      const call = parseToolCall(c);
      if (!call) return "a tool call isn't valid JSON";
      if (!LOCAL_TOOL_NAMES.includes(call.name)) return `uses a tool Mnx no longer has (${call.name})`;
    }
  }
  if (!/<\/think>\n\S/.test(m[m.length - 1].content)) return "empty answer";
  return null;
}

let rows = [];
try {
  rows = fs.readFileSync(LEARNED_FILE(), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
} catch {
  console.log(`Nothing learned yet (${LEARNED_FILE()} doesn't exist). Tap 👍 on good answers, correct bad ones with 👎, or chat with an online model with "Learn from my chats" on.`);
  process.exit(0);
}

const seen = new Set();
const keep = [];
const skipped = {};
for (const row of rows) {
  row.messages = [refreshSystem(row.messages[0]), ...row.messages.slice(1)];
  const why = check(row);
  if (why) {
    skipped[why] = (skipped[why] || 0) + 1;
    continue;
  }
  const key = JSON.stringify(row.messages.slice(1));
  if (seen.has(key)) continue;
  seen.add(key);
  keep.push({ category: row.category, messages: row.messages, train_from: row.train_from });
}
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, keep.map((r) => JSON.stringify(r)).join("\n") + (keep.length ? "\n" : ""));
const by = keep.reduce((m, r) => ((m[r.category] = (m[r.category] || 0) + 1), m), {});
console.log(`${keep.length} learned examples ready → ${path.relative(root, outFile)}`);
console.log(`  ${Object.entries(by).map(([k, v]) => `${k.replace("learned_", "")}: ${v}`).join(", ") || "none"}`);
for (const [why, n] of Object.entries(skipped)) console.log(`  skipped ${n}: ${why}`);

if (process.argv.includes("--upload")) {
  const token = process.env.HF_TOKEN;
  if (!token) {
    console.error("Set HF_TOKEN (a Hugging Face token with write access) to upload.");
    process.exit(1);
  }
  const H = { authorization: `Bearer ${token}` };
  const who = await (await fetch("https://huggingface.co/api/whoami-v2", { headers: H })).json();
  if (!who.name) {
    console.error("That HF_TOKEN didn't work:", JSON.stringify(who).slice(0, 200));
    process.exit(1);
  }
  const repo = process.env.MNX_HF_DATASET || `${who.name}/mnx-learned`;
  await fetch("https://huggingface.co/api/repos/create", {
    method: "POST",
    headers: { ...H, "content-type": "application/json" },
    body: JSON.stringify({ type: "dataset", name: repo.split("/")[1], organization: repo.split("/")[0] === who.name ? undefined : repo.split("/")[0], private: true }),
  }); // already exists → 409, fine
  const ndjson = [
    { key: "header", value: { summary: `Mnx learned examples (${keep.length})` } },
    { key: "file", value: { path: "learned.jsonl", content: Buffer.from(fs.readFileSync(outFile)).toString("base64"), encoding: "base64" } },
  ].map((x) => JSON.stringify(x)).join("\n");
  const r = await fetch(`https://huggingface.co/api/datasets/${repo}/commit/main`, { method: "POST", headers: { ...H, "content-type": "application/x-ndjson" }, body: ndjson });
  if (!r.ok) {
    console.error(`Upload failed (HTTP ${r.status}): ${(await r.text()).slice(0, 300)}\nYou can still upload ${outFile} to Colab by hand as extra.jsonl.`);
    process.exit(1);
  }
  console.log(`Uploaded to https://huggingface.co/datasets/${repo} (private). The training notebook downloads it automatically.`);
}

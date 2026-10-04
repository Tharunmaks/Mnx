// Measures how often the local model gets each kind of task right, using
// Mnx's real tool loop (local.js) with fixed tool results from the held-out
// test set. Prints a success rate per category and overall.
//
//   node training/eval.mjs --model models/mnx-q4_k_m.gguf [--limit 100] [--target 98]
//   MNX_LLAMA_SERVER=http://127.0.0.1:8080 node training/eval.mjs   (use a running llama-server)
//
// A case passes only if every check passes: right tool (or no tool), valid
// arguments, the permission step for code, and a final answer that uses the
// tool result. Writes training/data/eval-report.json.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
};
const casesFile = arg("cases", path.join(here, "data/test.jsonl"));
const limit = Number(arg("limit", 0));
const target = Number(arg("target", 98));
const model = arg("model", process.env.MNX_LOCAL_MODEL || "");
if (!process.env.MNX_LLAMA_SERVER && !process.env.MNX_LOCAL_BACKEND) process.env.MNX_LOCAL_BACKEND = "server";

const { handleLocalChat, localModelPath, warmUpLocal } = await import("../local.js");
const file = model ? path.resolve(model) : localModelPath(path.join(here, ".."));
if (!file || !fs.existsSync(file)) {
  console.error("No model found. Pass --model path/to/model.gguf");
  process.exit(2);
}

function score(ex, run) {
  const e = ex.expect;
  const fails = [];
  const calls = run.calls;
  const first = calls[0];
  const re = (s) => new RegExp(s, "i");
  if (run.error) fails.push(`error: ${run.error}`);
  if (run.badCalls) fails.push(`${run.badCalls} invalid tool call(s)`);
  if (e.tool === null) {
    if (calls.length) fails.push(`called ${first.name} but no tool was needed`);
  } else if (!first) fails.push(`expected ${e.tool}, made no tool call`);
  else {
    if (first.name !== e.tool) fails.push(`expected ${e.tool}, called ${first.name}`);
    for (const [k, pattern] of Object.entries(e.args || {})) if (!re(pattern).test(String(first.input[k] ?? ""))) fails.push(`${k}=${JSON.stringify(first.input[k])} doesn't match /${pattern}/`);
    for (const [k, v] of Object.entries(e.args_eq || {})) if (first.input[k] !== v) fails.push(`${k}=${JSON.stringify(first.input[k])}, expected ${JSON.stringify(v)}`);
    if (e.min_items) {
      const list = first.input.slides || first.input.sections || [];
      if (list.length < e.min_items) fails.push(`only ${list.length} slides/sections`);
    }
    if (e.then && !calls.slice(1).some((c) => c.name === e.then)) fails.push(`never called ${e.then}`);
    if (e.tool === "run_code" && !run.approvalAsked) fails.push("code ran without asking permission");
  }
  if (run.unexpected.length) fails.push(`unexpected tools: ${run.unexpected.join(", ")}`);
  const answer = run.answer.trim();
  if (!answer || answer === "…") fails.push("no final answer");
  if (/<\/?tool_call>|<\/?think>/.test(answer)) fails.push("raw tags in the answer");
  for (const pattern of e.answer || []) if (answer && !re(pattern).test(answer)) fails.push(`answer missing /${pattern}/`);
  return fails;
}

async function runCase(ex) {
  const queues = Object.fromEntries(Object.entries(ex.mock).map(([k, v]) => [k, [...v]]));
  const run = { calls: [], unexpected: [], answer: "", approvalAsked: false, badCalls: 0, error: null, chars: 0 };
  const ranIds = new Set();
  const t0 = Date.now();
  await handleLocalChat({
    body: {},
    messages: [{ role: "user", content: ex.messages[1].content }],
    file,
    isClosed: () => false,
    requestApproval: async () => {
      run.approvalAsked = true;
      return !ex.expect.decline;
    },
    runToolImpl: async (name, input, ctx) => {
      run.calls.push({ name, input });
      if (name === "run_code") {
        // Go through the same permission step as the real tool.
        const ok = await ctx.requestApproval({ kind: "run_code", language: input.language, code: input.code });
        if (!ok) return { result: { ran: false, reason: "The user chose not to run this code." } };
      }
      const next = queues[name]?.shift();
      if (!next) {
        run.unexpected.push(name);
        return { error: `No result available for ${name} in this test.` };
      }
      return next.error ? { error: next.error } : { result: next };
    },
    send: (ev) => {
      if (ev.t === "text" || ev.t === "thinking") run.chars += ev.text.length;
      if (ev.t === "assistant_turn") run.answer = ev.content[0].text;
      if (ev.t === "error") run.error = ev.text;
      if (ev.t === "tool_run") ranIds.add(ev.id);
      // A failed call that never reached tool_run was unparseable or unknown.
      if (ev.t === "tool_done" && ev.error && !ranIds.has(ev.id)) run.badCalls++;
    },
  });
  run.seconds = (Date.now() - t0) / 1000;
  return run;
}

const cases = fs.readFileSync(casesFile, "utf8").trim().split("\n").map((l) => JSON.parse(l)).slice(0, limit || undefined);
console.log(`Evaluating ${cases.length} held-out cases on ${path.basename(file)}…`);
await warmUpLocal(file).catch(() => {});
const byCat = {};
const failures = [];
let totalChars = 0;
let totalSeconds = 0;
for (const [i, ex] of cases.entries()) {
  const run = await runCase(ex);
  const fails = score(ex, run);
  totalChars += run.chars;
  totalSeconds += run.seconds;
  const c = (byCat[ex.category] ||= { pass: 0, total: 0 });
  c.total++;
  if (!fails.length) c.pass++;
  else failures.push({ category: ex.category, prompt: ex.messages[1].content, fails, calls: run.calls, answer: run.answer.slice(0, 400) });
  if (process.stdout.isTTY) process.stdout.write(`\r  ${i + 1}/${cases.length}`);
}
const pass = Object.values(byCat).reduce((a, c) => a + c.pass, 0);
const pct = (p, t) => ((100 * p) / t).toFixed(1);
console.log("\n\nCategory            Success");
for (const [k, c] of Object.entries(byCat).sort()) console.log(`${k.padEnd(20)}${pct(c.pass, c.total).padStart(6)}%  (${c.pass}/${c.total})`);
const overall = (100 * pass) / cases.length;
console.log(`${"OVERALL".padEnd(20)}${overall.toFixed(1).padStart(6)}%  (${pass}/${cases.length})   target ${target}%`);
console.log(`Speed: ~${(totalChars / Math.max(totalSeconds, 0.001)).toFixed(0)} characters/s of output, ${(totalSeconds / cases.length).toFixed(1)} s per case`);
for (const f of failures.slice(0, 8)) console.log(`\n✗ [${f.category}] ${f.prompt}\n   ${f.fails.join("; ")}`);
fs.writeFileSync(path.join(path.dirname(casesFile), "eval-report.json"), JSON.stringify({ overall, target, byCat, failures }, null, 2));
process.emit("exit");
process.exit(overall >= target ? 0 : 1);

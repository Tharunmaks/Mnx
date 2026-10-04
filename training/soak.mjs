// Long chat test ("soak test"): chats with Mnx non-stop through its real
// HTTP API, the way the browser does, and reports how it holds up over time.
//
//   npm run soak -- --hours 3 --model models/mnx-q4_k_m.gguf
//   npm run soak -- --hours 3 --url http://localhost:3000     (an Mnx that's already running)
//
// Each chat session asks 1–8 questions in a row (later questions see the
// earlier answers, like a real chat), approves or declines code runs, and
// with --chaos (default 0.1) mixes in cancelled replies and odd requests.
// Every few minutes it prints success per category, response times and
// memory use, and writes training/data/soak-report.json.
//
// Tools run for real (web search, weather, code runner…), so answer
// contents aren't compared; a turn passes when the model picked the right
// tool with valid arguments, asked permission before running code, and
// finished with a clean answer and no errors.
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { score } from "./scoring.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
};
const hours = Number(arg("hours", 3));
const chaos = Number(arg("chaos", 0.1));
const every = Number(arg("report-every", 10)) * 60_000;
const scoreTurns = arg("score", "yes") !== "no";
const casesFile = arg("cases", path.join(here, "data/test.jsonl"));
const reportFile = arg("report", path.join(here, "data/soak-report.json"));
const deadline = Date.now() + hours * 3600_000;
const cases = fs.readFileSync(casesFile, "utf8").trim().split("\n").map((l) => JSON.parse(l));
const singleTurn = cases.filter((c) => c.messages.length === 2);
const rand = (n) => Math.floor(Math.random() * n);
const pick = (a) => a[rand(a.length)];

// Start our own Mnx server unless --url is given.
let base = arg("url", "");
let serverProc = null;
if (!base) {
  const port = 39000 + rand(1000);
  const env = { ...process.env, PORT: String(port), HOST: "127.0.0.1", MNX_LOCAL_BACKEND: process.env.MNX_LOCAL_BACKEND || "server" };
  if (arg("model", "")) env.MNX_LOCAL_MODEL = path.resolve(arg("model"));
  serverProc = spawn(process.execPath, ["server.js"], { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });
  serverProc.stderr.on("data", (d) => fs.appendFileSync(reportFile + ".server.log", d));
  await new Promise((resolve, reject) => {
    serverProc.stdout.on("data", (d) => String(d).includes("Mnx is live") && resolve());
    serverProc.on("exit", (c) => reject(new Error(`Mnx server exited (${c})`)));
  });
  base = `http://127.0.0.1:${port}`;
}
const stop = () => serverProc?.kill();
process.on("SIGINT", () => (stop(), process.exit(130)));

// Unusual requests that exercise error paths (some only do something with a test model).
const CHAOS_PROMPTS = ["bad json please", "call an unknown tool", "loop forever", "unclosed thought", "no think please", "", "   ", "🙂".repeat(50), "x".repeat(4000)];

const stats = { started: new Date().toISOString(), sessions: 0, turns: 0, scored: 0, passed: 0, aborted: 0, chaos: 0, httpErrors: 0, streamErrors: 0, hung: 0, byCat: {}, latencies: [], firstToken: [], memory: [], failures: [] };

function rssMB(pid) {
  try {
    const m = fs.readFileSync(`/proc/${pid}/status`, "utf8").match(/VmRSS:\s+(\d+)/);
    return m ? Math.round(Number(m[1]) / 1024) : null;
  } catch {
    return null;
  }
}
function llamaPids() {
  try {
    return fs.readdirSync("/proc").filter((p) => /^\d+$/.test(p)).filter((p) => {
      try {
        return /llama-server/.test(fs.readFileSync(`/proc/${p}/cmdline`, "utf8"));
      } catch {
        return false;
      }
    });
  } catch {
    return [];
  }
}

// One turn: POST the conversation, read the event stream, answer approvals.
async function turn(messages, { approve = true, abortAfterMs = 0 } = {}) {
  const ctrl = new AbortController();
  const t0 = Date.now();
  const run = { calls: [], unexpected: [], answer: "", approvalAsked: false, badCalls: 0, error: null, history: [], firstToken: null, done: false };
  const ranIds = new Set();
  const watchdog = setTimeout(() => ctrl.abort(new Error("hung")), 10 * 60_000);
  let abortTimer = abortAfterMs ? setTimeout(() => ctrl.abort(), abortAfterMs) : null;
  try {
    const res = await fetch(`${base}/api/chat`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "local", messages }), signal: ctrl.signal });
    if (!res.ok) {
      run.error = `HTTP ${res.status}`;
      return run;
    }
    const dec = new TextDecoder();
    let buf = "";
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if ((ev.t === "text" || ev.t === "thinking") && run.firstToken === null) run.firstToken = Date.now() - t0;
        if (ev.t === "tool_run") {
          ranIds.add(ev.id);
          run.calls.push({ name: ev.name, input: ev.input });
        }
        if (ev.t === "tool_done" && ev.error && !ranIds.has(ev.id)) run.badCalls++;
        if (ev.t === "approval_request") {
          run.approvalAsked = true;
          await fetch(`${base}/api/approve`, { method: "POST", body: JSON.stringify({ id: ev.id, token: ev.token, approve }) });
        }
        if (ev.t === "assistant_turn") {
          run.answer = ev.content.map((b) => b.text || "").join("");
          run.history.push({ role: "assistant", content: ev.content });
        }
        if (ev.t === "error") run.error = ev.text;
        if (ev.t === "done") run.done = true;
      }
    }
  } catch (err) {
    if (ctrl.signal.reason?.message === "hung") run.hung = true;
    else if (err.name !== "AbortError") run.error = String(err.message || err);
    run.aborted = true;
  } finally {
    clearTimeout(watchdog);
    clearTimeout(abortTimer);
  }
  run.ms = Date.now() - t0;
  return run;
}

function pct(a, p) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}
function report(final = false) {
  const mins = ((Date.now() - Date.parse(stats.started)) / 60000).toFixed(0);
  const mem = { mnx: serverProc ? rssMB(serverProc.pid) : null, llama: llamaPids().map(rssMB).reduce((a, b) => a + (b || 0), 0) || null, at: Math.round((Date.now() - Date.parse(stats.started)) / 60000) };
  stats.memory.push(mem);
  const rate = stats.scored ? ((100 * stats.passed) / stats.scored).toFixed(1) : "–";
  console.log(`\n[${mins} min] ${stats.sessions} chats, ${stats.turns} turns | success ${rate}% (${stats.passed}/${stats.scored}) | cancelled ${stats.aborted} | errors: http ${stats.httpErrors}, stream ${stats.streamErrors}, hung ${stats.hung} | ` +
    `latency p50 ${(pct(stats.latencies, 50) / 1000).toFixed(1)}s p95 ${(pct(stats.latencies, 95) / 1000).toFixed(1)}s, first token p50 ${(pct(stats.firstToken, 50) / 1000).toFixed(2)}s | memory: Mnx ${mem.mnx ?? "?"} MB, llama ${mem.llama ?? "?"} MB`);
  if (final) {
    console.log("\nCategory            Success");
    for (const [k, c] of Object.entries(stats.byCat).sort()) console.log(`${k.padEnd(20)}${((100 * c.pass) / c.total).toFixed(1).padStart(6)}%  (${c.pass}/${c.total})`);
    for (const f of stats.failures.slice(-8)) console.log(`\n✗ [${f.category}] ${f.prompt.slice(0, 80)}\n   ${f.fails.join("; ")}`);
  }
  const { latencies, firstToken, ...rest } = stats;
  fs.writeFileSync(reportFile, JSON.stringify({ ...rest, latency: { p50: pct(latencies, 50), p95: pct(latencies, 95), max: Math.max(0, ...latencies) }, firstTokenP50: pct(firstToken, 50) }, null, 2));
}

console.log(`Chatting with Mnx at ${base} for ${hours} h (chaos ${chaos})…`);
let nextReport = Date.now() + every;
while (Date.now() < deadline) {
  stats.sessions++;
  const history = [];
  const length = 1 + rand(8);
  for (let t = 0; t < length && Date.now() < deadline; t++) {
    // Follow-up cases need their own earlier turns, so they only start a new chat.
    const ex = history.length ? pick(singleTurn) : pick(cases);
    const isChaos = Math.random() < chaos;
    const userText = isChaos ? pick(CHAOS_PROMPTS) : ex.messages.at(-1).content;
    // A brand-new chat may start with the test case's own earlier turns.
    if (!history.length && !isChaos) for (const m of ex.messages.slice(1, -1)) history.push({ role: m.role, content: m.content });
    history.push({ role: "user", content: userText });
    const abortAfterMs = isChaos && Math.random() < 0.5 ? 200 + rand(3000) : 0;
    const run = await turn(history, { approve: !ex.expect.decline, abortAfterMs });
    stats.turns++;
    if (isChaos) stats.chaos++;
    if (run.aborted) stats.aborted++;
    if (run.hung) stats.hung++;
    if (run.error?.startsWith("HTTP")) stats.httpErrors++;
    else if (run.error && !(isChaos && /No local model|empty/i.test(run.error))) stats.streamErrors++;
    if (!run.aborted) {
      stats.latencies.push(run.ms);
      if (run.firstToken !== null) stats.firstToken.push(run.firstToken);
    }
    // Keep the conversation going like the browser does.
    history.push(...run.history);
    if (!run.history.length) history.pop();
    if (!isChaos && !run.aborted && scoreTurns) {
      const fails = score(ex, run, { checkAnswers: false });
      if (!run.done) fails.push("stream ended without finishing");
      const c = (stats.byCat[ex.category] ||= { pass: 0, total: 0 });
      c.total++;
      stats.scored++;
      if (!fails.length) {
        c.pass++;
        stats.passed++;
      } else stats.failures.push({ category: ex.category, prompt: userText, fails, at: new Date().toISOString() });
      if (stats.failures.length > 500) stats.failures.shift();
    }
    if (Date.now() >= nextReport) {
      report();
      nextReport = Date.now() + every;
    }
  }
}
report(true);
stop();
process.exit(0);

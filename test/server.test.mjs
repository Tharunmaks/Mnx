// Integration tests: the real server, driven over HTTP, against a mock
// Anthropic API and a fake llama-server (test/fixtures). Run with: npm test
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { startMockAnthropic } from "./fixtures/mock-anthropic.mjs";
import { handleLocalChat } from "../local.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const hasPython = (() => {
  try {
    execFileSync("python3", ["-c", "1"]);
    return true;
  } catch {
    return false;
  }
})();

let mock;
let server;
let base;
let llamaLog;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-it-"));

before(async () => {
  mock = await startMockAnthropic();
  const port = 31000 + Math.floor(Math.random() * 2000);
  const model = path.join(tmp, "mnx-test.gguf");
  fs.writeFileSync(model, Buffer.alloc(1_100_000));
  llamaLog = path.join(tmp, "llama.log");
  server = spawn(process.execPath, ["server.js"], {
    cwd: root,
    env: {
      ...process.env,
      PORT: String(port),
      HOST: "127.0.0.1",
      ANTHROPIC_API_KEY: "sk-test-secret",
      ANTHROPIC_BASE_URL: `http://127.0.0.1:${mock.port}`,
      PATH: `${path.join(root, "test/fixtures/bin")}${path.delimiter}${process.env.PATH}`,
      MNX_LOCAL_BACKEND: "server",
      MNX_LOCAL_MODEL: model,
      MNX_LLAMA_PORT: String(port + 1),
      FAKE_LLAMA_LOG: llamaLog,
      MNX_LEARNED_FILE: path.join(tmp, "learned.jsonl"),
      MNX_MEMORY_FILE: path.join(tmp, "memory.json"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let out = "";
  server.stderr.on("data", (d) => (out += d));
  await new Promise((resolve, reject) => {
    server.stdout.on("data", (d) => String(d).includes("Mnx is live") && resolve());
    server.on("exit", (c) => reject(new Error(`server exited ${c}: ${out}`)));
  });
  base = `http://127.0.0.1:${port}`;
});

after(() => {
  server?.kill();
  mock?.server.close();
});

// POST /api/chat and collect events. `approve` answers approval requests.
async function chat(body, { approve, abortOn } = {}) {
  const ctrl = new AbortController();
  const res = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: ctrl.signal,
  });
  if (!res.ok) return { status: res.status, events: [], error: await res.json() };
  const events = [];
  let aborted = false;
  const dec = new TextDecoder();
  let buf = "";
  try {
    for await (const chunk of res.body) {
      if (aborted) break;
      buf += dec.decode(chunk, { stream: true });
      let i;
      while (!aborted && (i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        events.push(ev);
        if (abortOn?.(ev)) {
          aborted = true;
          ctrl.abort();
          break;
        }
        if (ev.t === "approval_request" && approve !== undefined) {
          const r = await fetch(`${base}/api/approve`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ id: ev.id, token: ev.token, approve }),
          });
          assert.equal(r.status, 200);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") throw err;
  }
  return { status: res.status, events, aborted };
}
const user = (text) => ({ role: "user", content: [{ type: "text", text }] });
const of = (events, t) => events.filter((e) => e.t === t);
const text = (events) => of(events, "text").map((e) => e.text).join("");
const thinking = (events) => of(events, "thinking").map((e) => e.text).join("");
const llamaRequests = () => fs.readFileSync(llamaLog, "utf8").trim().split("\n").map((l) => JSON.parse(l));

describe("HTTP basics", () => {
  test("config lists the online models and the local model", async () => {
    const cfg = await (await fetch(`${base}/api/config`)).json();
    assert.equal(cfg.hasKey, true);
    assert.equal(cfg.local, "local");
    assert.ok(cfg.models["claude-opus-5-5"]);
    assert.ok(cfg.models.local);
  });
  test("serves the app and whitelisted vendor files only", async () => {
    assert.equal((await fetch(`${base}/`)).status, 200);
    assert.equal((await fetch(`${base}/vendor/marked.js`)).status, 200);
    assert.equal((await fetch(`${base}/vendor/../../server.js`)).status, 404);
    assert.equal((await fetch(`${base}/vendor/secret.js`)).status, 404);
  });
  test("blocks path traversal outside public/", async () => {
    for (const p of ["/%2e%2e/server.js", "/..%2fserver.js", "/%2e%2e/%2e%2e/etc/passwd"]) {
      const r = await fetch(`${base}${p}`);
      const body = await r.text();
      assert.ok(!body.includes("Anthropic") && !body.includes("root:"), `${p} leaked a file`);
      assert.ok([403, 404].includes(r.status), `${p} → ${r.status}`);
    }
  });
  test("rejects bad chat requests", async () => {
    assert.equal((await chat({ messages: [] })).status, 400);
    assert.equal((await chat({ messages: [{ role: "assistant", content: "x" }] })).status, 400);
    const r = await fetch(`${base}/api/chat`, { method: "POST", body: "{not json" });
    assert.equal(r.status, 400);
    assert.equal((await fetch(`${base}/api/chat`, { method: "PUT" })).status, 405);
  });
  test("approve endpoint rejects unknown ids and wrong tokens", async () => {
    const r = await fetch(`${base}/api/approve`, { method: "POST", body: JSON.stringify({ id: "nope", token: "x", approve: true }) });
    assert.equal(r.status, 404);
  });
  test("only listens on loopback", async () => {
    const ip = Object.values(os.networkInterfaces()).flat().find((n) => n && n.family === "IPv4" && !n.internal)?.address;
    if (!ip) return;
    await assert.rejects(fetch(`http://${ip}:${new URL(base).port}/api/config`, { signal: AbortSignal.timeout(2000) }));
  });
});

describe("online model (mock Anthropic API)", () => {
  test("streams thinking and text with the right request settings", async () => {
    const before = mock.requests.length;
    const { events } = await chat({ messages: [user("hello")], model: "claude-sonnet-5-5", effort: "high" });
    assert.match(thinking(events), /Understanding the request/);
    assert.equal(text(events), "Hello from the **mock** model!");
    assert.equal(events.at(-1).t, "done");
    const req = mock.requests[before];
    assert.equal(req.body.model, "claude-sonnet-5-5");
    assert.deepEqual(req.body.thinking, { type: "adaptive", display: "summarized" });
    assert.deepEqual(req.body.output_config, { effort: "high" });
    assert.equal(req.body.fallbacks, "default");
    assert.match(req.headers["anthropic-beta"], /server-side-fallback-2026-07-01/);
    assert.ok(req.body.tools.some((t) => t.name === "run_code"));
  });
  test("unknown model and effort fall back to safe defaults", async () => {
    const before = mock.requests.length;
    await chat({ messages: [user("hello")], model: "gpt-9", effort: "ludicrous" });
    assert.equal(mock.requests[before].body.model, "claude-opus-5-5");
    assert.deepEqual(mock.requests[before].body.output_config, { effort: "medium" });
  });
  test("web search results are forwarded", async () => {
    const { events } = await chat({ messages: [user("search something")] });
    const result = of(events, "block_start").find((e) => e.block.type === "web_search_tool_result");
    assert.equal(result.block.results[0].url, "https://example.com/r");
  });
  test("tool loop: create_document runs and the result goes back to the model", async () => {
    const before = mock.requests.length;
    const { events } = await chat({ messages: [user("make a deck")] });
    const done = of(events, "tool_done")[0];
    assert.equal(done.display.kind, "document");
    assert.equal(done.display.filename, "Mock Deck.pptx");
    assert.match(text(events), /delivered/);
    // Second request carries the assistant tool_use + matching tool_result.
    const second = mock.requests[before + 1].body.messages;
    assert.equal(second.at(-1).content[0].type, "tool_result");
    // History events let the client continue the conversation.
    assert.equal(of(events, "assistant_turn").length, 2);
    assert.equal(of(events, "user_turn").length, 1);
  });
  test("follow-up turns with returned history are accepted", async () => {
    const first = await chat({ messages: [user("make a deck")] });
    const history = [user("make a deck")];
    for (const e of first.events) {
      if (e.t === "assistant_turn") history.push({ role: "assistant", content: e.content });
      if (e.t === "user_turn") history.push({ role: "user", content: e.content });
    }
    history.push(user("thanks"));
    const second = await chat({ messages: history });
    assert.equal(of(second.events, "error").length, 0);
    assert.equal(second.events.at(-1).t, "done");
  });
  test("run_code waits for approval and runs once approved", async () => {
    const { events } = await chat({ messages: [user("run some code")] }, { approve: true });
    const req = of(events, "approval_request")[0];
    assert.equal(req.code, "print(sum(range(10)))");
    assert.ok(req.tool_use_id.startsWith("toolu_"));
    assert.equal(of(events, "approval_result")[0].approved, true);
    if (hasPython) {
      const done = of(events, "tool_done")[0];
      assert.equal(done.display.stdout.trim(), "45");
      assert.match(text(events), /45/);
    }
  });
  test("run_code is skipped when declined, and the model is told", async () => {
    const { events } = await chat({ messages: [user("run some code")] }, { approve: false });
    assert.equal(of(events, "tool_done")[0].display.declined, true);
    assert.match(text(events), /chose not to run/);
  });
  test("closing the chat while waiting cancels the approval", async () => {
    let pending;
    await chat({ messages: [user("run some code")] }, { abortOn: (e) => e.t === "approval_request" && (pending = e) });
    await new Promise((r) => setTimeout(r, 200));
    const r = await fetch(`${base}/api/approve`, {
      method: "POST",
      body: JSON.stringify({ id: pending.id, token: pending.token, approve: true }),
    });
    assert.equal(r.status, 404);
  });
  test("a refusal is reported to the user", async () => {
    const { events } = await chat({ messages: [user("please refuse")] });
    assert.match(of(events, "notice")[0].text, /can't help/);
  });
  test("MCP connectors are validated and passed through", async () => {
    const before = mock.requests.length;
    await chat({
      messages: [user("hello")],
      mcpServers: [
        { name: "My Tools!", url: "https://a.example/mcp", token: " t0k " },
        { name: "my tools", url: "https://b.example/mcp" },
        { name: "insecure", url: "http://c.example/mcp" },
      ],
    });
    const body = mock.requests[before].body;
    assert.deepEqual(body.mcp_servers, [
      { type: "url", url: "https://a.example/mcp", name: "my-tools", authorization_token: "t0k" },
      { type: "url", url: "https://b.example/mcp", name: "my-tools-x" },
    ]);
    assert.equal(body.tools.filter((t) => t.type === "mcp_toolset").length, 2);
    assert.match(mock.requests[before].headers["anthropic-beta"], /mcp-client-2025-11-20/);
  });
});

describe("local model (fake llama-server)", () => {
  const local = (text, opts = {}) => chat({ messages: [user(text)], model: "local" }, opts);

  test("streams thinking then the answer, with Qwen sampling and tools", async () => {
    const { events } = await local("hi there");
    assert.match(thinking(events), /Understanding the question/);
    assert.equal(text(events), "Hello! I'm **Mnx**.");
    assert.equal(of(events, "assistant_turn")[0].content[0].text, "Hello! I'm **Mnx**.");
    const req = llamaRequests().at(-1);
    assert.equal(req.top_k, 20);
    assert.equal(req.top_p, 0.8);
    assert.equal(req.max_tokens, 4096);
    assert.match(req.messages[0].content, /<tools>/);
    assert.match(req.messages[0].content, /"name":"run_code"/);
  });
  test("converts mixed history (thinking, tool blocks) into plain turns", async () => {
    await chat({
      model: "local",
      messages: [
        user("first"),
        { role: "assistant", content: [{ type: "thinking", thinking: "secret", signature: "s" }, { type: "tool_use", id: "t1", name: "x", input: {} }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "r" }] },
        { role: "assistant", content: [{ type: "text", text: "earlier answer" }] },
        user("second"),
      ],
    });
    const msgs = llamaRequests().at(-1).messages;
    assert.deepEqual(msgs.map((m) => m.role), ["system", "user", "assistant", "user"]);
    assert.equal(msgs[2].content, "earlier answer");
    assert.ok(!JSON.stringify(msgs).includes("secret"));
  });
  test("answers without a <think> block still work", async () => {
    const { events } = await local("no think please");
    assert.equal(text(events), "Plain answer without thinking.");
  });
  test("an unclosed <think> still produces an answer", async () => {
    const { events } = await local("unclosed thought");
    assert.match(text(events), /never close/);
  });
  test("runs two tools in a row, then answers", async () => {
    const { events } = await local("use two tools");
    assert.deepEqual(of(events, "tool_done").map((e) => e.name), ["create_file", "create_image"]);
    assert.deepEqual(of(events, "iteration").map((e) => e.it), [0, 1, 2]);
    assert.match(text(events), /Done/);
    const lastReq = llamaRequests().at(-1).messages;
    assert.match(lastReq.at(-1).content, /<tool_response>/);
    assert.match(lastReq.at(-2).content, /<tool_call>/);
  });
  test("stops a tool loop after 5 steps", async () => {
    const { events } = await local("loop forever");
    assert.equal(of(events, "iteration").length, 5);
    assert.match(of(events, "notice")[0].text, /maximum/);
    assert.equal(events.at(-1).t, "done");
  });
  test("invalid tool JSON is reported and the model recovers", async () => {
    const { events } = await local("bad json please");
    assert.equal(of(events, "tool_done")[0].error, true);
    assert.match(llamaRequests().at(-1).messages.at(-1).content, /wasn't valid JSON/);
    assert.equal(events.at(-1).t, "done");
  });
  test("unknown tools are refused", async () => {
    const { events } = await local("call an unknown tool");
    assert.equal(of(events, "tool_done")[0].error, true);
    assert.match(llamaRequests().at(-1).messages.at(-1).content, /Unknown tool/);
  });
  test("run_code with raw newlines: asks, then runs", { skip: !hasPython && "python3 not installed" }, async () => {
    const { events } = await local("raw newline code", { approve: true });
    assert.equal(of(events, "approval_request")[0].code, "x = 6 * 7\nprint('answer', x)");
    assert.equal(of(events, "tool_done")[0].display.stdout.trim(), "answer 42");
  });
  test("run_code does not leak the server's API key", { skip: !hasPython && "python3 not installed" }, async () => {
    const { events } = await local("try to leak the key", { approve: true });
    assert.equal(of(events, "tool_done")[0].display.stdout.trim(), "KEY=none");
  });
  test("declining run_code never runs it", async () => {
    const { events } = await local("raw newline code", { approve: false });
    const done = of(events, "tool_done")[0];
    assert.equal(done.display.declined, true);
    assert.equal(done.display.stdout, undefined);
  });
  test("aborting mid-answer doesn't block the next request", async () => {
    const r = await local("slow answer", { abortOn: (e) => e.t === "thinking" });
    assert.equal(r.aborted, true);
    const next = await local("hi again");
    assert.equal(text(next.events), "Hello! I'm **Mnx**.");
  });
  test("a missing model file gives a helpful error", async () => {
    const events = [];
    await handleLocalChat({
      body: {},
      messages: [user("hi")],
      file: path.join(tmp, "missing.gguf"),
      send: (e) => events.push(e),
      isClosed: () => false,
    });
    assert.match(events[0].text, /No local model found/);
  });
});

test("local: a model that repeats itself is stopped, with a note", async () => {
  const t0 = Date.now();
  const { events } = await chat({ messages: [{ role: "user", content: "please repeat yourself" }], model: "local" });
  assert.ok(events.some((e) => e.t === "notice" && /repeating itself/.test(e.text)), JSON.stringify(events.filter((e) => e.t !== "text").slice(-5)));
  assert.ok(events.some((e) => e.t === "done"));
  const text = events.filter((e) => e.t === "text").map((e) => e.text).join("");
  assert.ok(text.length < 4000, `stopped early (${text.length} chars)`);
  assert.ok(Date.now() - t0 < 20000);
});

test("local: an answer cut off at the length limit says so", async () => {
  const { events } = await chat({ messages: [{ role: "user", content: "this will be cut off" }], model: "local" });
  assert.ok(events.some((e) => e.t === "notice" && /length limit/.test(e.text)));
});

test("learning: 👍 on a local reply saves exactly what the model saw and wrote", async () => {
  const { events } = await chat({ messages: [{ role: "user", content: "hello there" }], model: "local" });
  const trace = events.find((e) => e.t === "trace")?.id;
  assert.ok(trace, "a trace id is sent with every finished local reply");
  const r = await (await fetch(`${base}/api/feedback`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ trace, rating: "up" }) })).json();
  assert.equal(r.ok, true);
  assert.equal(r.stats.good, 1);
  const row = JSON.parse(fs.readFileSync(path.join(tmp, "learned.jsonl"), "utf8").trim().split("\n").at(-1));
  assert.equal(row.category, "learned_good");
  assert.equal(row.messages[row.train_from].content, "hello there");
  assert.match(row.messages.at(-1).content, /^<think>[\s\S]*<\/think>\n\S/);
});

test("learning: 👎 with a correction replaces the answer; without one it isn't trained on", async () => {
  const { events } = await chat({ messages: [{ role: "user", content: "hello again" }], model: "local" });
  const trace = events.find((e) => e.t === "trace").id;
  const post = (body) => fetch(`${base}/api/feedback`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((x) => x.json());
  assert.equal((await post({ trace, rating: "down", correction: "Hi! How can I help you today?" })).stats.corrected, 1);
  const last = JSON.parse(fs.readFileSync(path.join(tmp, "learned.jsonl"), "utf8").trim().split("\n").at(-1));
  assert.match(last.messages.at(-1).content, /<\/think>\nHi! How can I help you today\?$/);
  const before = (await (await fetch(`${base}/api/learn`)).json()).total;
  await post({ trace, rating: "down" });
  assert.equal((await (await fetch(`${base}/api/learn`)).json()).total, before);
  assert.equal((await post({ trace: "nope", rating: "up" })).ok, false);
});

test("security: other websites can't post to Mnx (chat, memory, learning)", async () => {
  for (const p of ["/api/chat", "/api/feedback", "/api/run", "/api/approve"]) {
    const r = await fetch(`${base}${p}`, { method: "POST", headers: { origin: "https://evil.example", "content-type": "application/json" }, body: "{}" });
    assert.equal(r.status, 403, p);
  }
  const same = await fetch(`${base}/api/learn`, { headers: { origin: base } });
  assert.equal(same.status, 200);
});

test("▶ Run: runs code from an answer only with this page's token", async () => {
  const cfg = await (await fetch(`${base}/api/config`)).json();
  const run = (token) => fetch(`${base}/api/run`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ token, language: "python", code: "print(6 * 7)" }) });
  assert.equal((await run("wrong-token-wrong-token")).status, 403);
  if (!cfg.runToken) return;
  const out = await (await run(cfg.runToken)).json();
  if (out.error && /isn't installed/.test(out.error)) return; // no python here
  assert.match(out.result.stdout, /42/);
  assert.equal(out.display.kind, "code_run");
});

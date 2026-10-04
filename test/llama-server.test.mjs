// Tests for starting / reusing llama-server (the Android/Termux backend).
import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const fake = path.join(root, "test/fixtures/bin/llama-server");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-ls-"));
const modelA = path.join(tmp, "mnx-a.gguf");
const modelB = path.join(tmp, "mnx-b.gguf");
for (const m of [modelA, modelB]) fs.writeFileSync(m, Buffer.alloc(1_100_000));

process.env.MNX_LOCAL_BACKEND = "server";
process.env.PATH = `${path.dirname(fake)}${path.delimiter}${process.env.PATH}`;
const children = [];
after(() => {
  children.forEach((c) => c.kill());
  // Mnx-started servers are stopped on process exit; trigger it explicitly.
  process.emit("exit");
});

async function startStale(port, model, log) {
  const child = spawn(fake, ["-m", model, "--port", String(port)], { env: { ...process.env, FAKE_LLAMA_LOG: log }, stdio: "ignore" });
  children.push(child);
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) return;
    } catch {
      /* starting */
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error("fake did not start");
}

async function ask(mod, file) {
  const events = [];
  await mod.handleLocalChat({
    body: {},
    messages: [{ role: "user", content: "hello" }],
    file,
    send: (e) => events.push(e),
    isClosed: () => false,
    requestApproval: async () => false,
  });
  return events;
}

test("reuses a leftover llama-server that has the same model loaded", async () => {
  const port = 34000 + Math.floor(Math.random() * 500);
  const staleLog = path.join(tmp, "stale-a.log");
  await startStale(port, modelA, staleLog);
  process.env.MNX_LLAMA_PORT = String(port);
  const mod = await import(`../local.js?reuse=${port}`);
  const events = await ask(mod, modelA);
  assert.equal(events.filter((e) => e.t === "error").length, 0, JSON.stringify(events.find((e) => e.t === "error")));
  assert.ok(fs.existsSync(staleLog), "the existing server should have answered");
});

test("starts a new server on the next port if a different model is loaded", async () => {
  const port = 34600 + Math.floor(Math.random() * 500);
  const staleLog = path.join(tmp, "stale-b.log");
  await startStale(port, modelB, staleLog);
  process.env.MNX_LLAMA_PORT = String(port);
  process.env.FAKE_LLAMA_LOG = path.join(tmp, "new.log");
  const mod = await import(`../local.js?other=${port}`);
  const events = await ask(mod, modelA);
  assert.equal(events.filter((e) => e.t === "error").length, 0, JSON.stringify(events.find((e) => e.t === "error")));
  assert.ok(!fs.existsSync(staleLog), "the other model's server must not be used");
  assert.ok(fs.existsSync(process.env.FAKE_LLAMA_LOG), "a new server should have answered");
  const props = await (await fetch(`http://127.0.0.1:${port + 1}/props`)).json();
  assert.equal(props.model_path, modelA);
});

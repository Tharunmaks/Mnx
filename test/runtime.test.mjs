// Opt-in tests against real llama.cpp with a tiny random-weight model.
//   npm run test:runtime
// Needs: pip install gguf numpy. For the llama-server part, also set
// MNX_LLAMA_SERVER_BIN to a llama-server binary (Termux: $(which llama-server)).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const enabled = process.env.MNX_RUNTIME_TESTS === "1";
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-rt-"));
const model = path.join(tmp, "tiny.gguf");
if (enabled) execFileSync("python3", [path.join(root, "test/fixtures/make-tiny-gguf.py"), model]);

async function ask(mod, messages, { abortAfter } = {}) {
  const events = [];
  const ctrl = new AbortController();
  let n = 0;
  await mod.handleLocalChat({
    body: {},
    messages,
    file: model,
    signal: ctrl.signal,
    send: (e) => {
      events.push(e);
      if ((e.t === "text" || e.t === "thinking") && ++n === abortAfter) ctrl.abort();
    },
    isClosed: () => ctrl.signal.aborted,
    requestApproval: async () => false,
  });
  return events;
}

async function conversation(mod) {
  const first = await ask(mod, [{ role: "user", content: "Hello" }]);
  assert.equal(first.find((e) => e.t === "error")?.text, undefined);
  assert.ok(first.some((e) => e.t === "text" || e.t === "thinking"), "model produced output");
  const turn = first.find((e) => e.t === "assistant_turn");
  assert.ok(turn);
  const second = await ask(mod, [{ role: "user", content: "Hello" }, { role: "assistant", content: turn.content }, { role: "user", content: "More" }]);
  assert.ok(second.some((e) => e.t === "done"));
  const aborted = await ask(mod, [{ role: "user", content: "Stop me" }], { abortAfter: 3 });
  assert.ok(!aborted.some((e) => e.t === "done"));
  const after = await ask(mod, [{ role: "user", content: "Still alive?" }]);
  assert.ok(after.some((e) => e.t === "done"), "works after an abort");
}

test("in-process llama.cpp (node-llama-cpp)", { skip: !enabled && "set MNX_RUNTIME_TESTS=1", timeout: 300000 }, async () => {
  delete process.env.MNX_LOCAL_BACKEND;
  await conversation(await import("../local.js?inproc"));
});

test("real llama-server", { skip: (!enabled || !process.env.MNX_LLAMA_SERVER_BIN) && "set MNX_RUNTIME_TESTS=1 and MNX_LLAMA_SERVER_BIN", timeout: 300000 }, async () => {
  process.env.MNX_LOCAL_BACKEND = "server";
  process.env.MNX_LLAMA_PORT = String(35000 + Math.floor(Math.random() * 1000));
  const mod = await import("../local.js?server");
  assert.equal(await mod.warmUpLocal(model), true);
  await conversation(mod);
  process.emit("exit"); // stop the llama-server Mnx started
});

// With no API key, choosing an online model must not hang: Mnx answers with
// the local model, or explains what's missing.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-nokey-"));

async function withServer(model, fn) {
  const port = 33000 + Math.floor(Math.random() * 1000);
  const env = { ...process.env, PORT: String(port), HOST: "127.0.0.1", MNX_PREWARM: "off", MNX_LOCAL_BACKEND: "server", MNX_LLAMA_PORT: String(port + 1),
    PATH: `${path.join(root, "test/fixtures/bin")}${path.delimiter}${process.env.PATH}`, MNX_LOCAL_MODEL: model || path.join(tmp, "missing.gguf") };
  delete env.ANTHROPIC_API_KEY;
  delete env.ANTHROPIC_AUTH_TOKEN;
  const server = spawn(process.execPath, ["server.js"], { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });
  await new Promise((resolve) => server.stdout.on("data", (d) => String(d).includes("Mnx is live") && resolve()));
  try {
    const res = await fetch(`http://127.0.0.1:${port}/api/chat`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "claude-opus-5-5", messages: [{ role: "user", content: "Hi" }] }), signal: AbortSignal.timeout(30000) });
    const events = (await res.text()).split("\n\n").filter((l) => l.startsWith("data: ")).map((l) => JSON.parse(l.slice(6)));
    await fn(events);
  } finally {
    server.kill();
  }
}

test("an online model without an API key falls back to the local model", async () => {
  const model = path.join(tmp, "mnx.gguf");
  fs.writeFileSync(model, Buffer.alloc(1_100_000));
  await withServer(model, (events) => {
    assert.equal(events.filter((e) => e.t === "error").length, 0, JSON.stringify(events));
    assert.ok(events.some((e) => e.t === "text"), "the local model answered");
    assert.ok(!events.some((e) => e.t === "notice" && /garbled/.test(e.text)), "no retry loop");
  });
});

test("without an API key or a local model, Mnx says what's missing", async () => {
  await withServer(null, (events) => {
    assert.match(events.find((e) => e.t === "error")?.text || "", /ANTHROPIC_API_KEY/);
  });
});

// Tharun's own Mnx 7B runner (mnx7b/run.py): numpy maths must match the
// PyTorch training code, and Mnx must chat with it end to end. Uses a tiny
// random-weight model with the same architecture (test/fixtures/make-mnx7b-tiny.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const fixture = path.join(root, "test/fixtures/mnx7b-tiny");
const py = ["python3", "python"].find((p) => spawnSync(p, ["-c", "import numpy"]).status === 0);
const skip = !py && "python3 with numpy isn't installed";

const CHECK = `
import json, sys, importlib.util, numpy as np, builtins
real = builtins.__import__
if sys.argv[2] == "nolib":  # force the pure-Python tokenizer
    builtins.__import__ = lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n)) if n.startswith("tokenizers") else real(n, *a, **k)
spec = importlib.util.spec_from_file_location("run", sys.argv[1] + "/mnx7b/run.py"); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
fx = sys.argv[1] + "/test/fixtures/mnx7b-tiny"; ref = json.load(open(fx + "/reference.json")); eng = run.Engine(fx); m = eng.model
for s in ref["tokenizer"]:
    assert eng.tk.encode(s["text"]) == s["ids"], ("encode", s["text"])
    assert eng.tk.decode(s["ids"]) == s["text"], ("decode", s["text"])
assert np.array_equal(m.expert_ids(ref["tokens"], 0), np.array(ref["ex_ids"])), "expert ids"
r = np.array(ref["logits"]); d = float(np.abs(m.full_logits(ref["tokens"]) - r).max())
assert d < 1e-3, ("logits differ", d)
m.reset(); m.forward(ref["tokens"][:5]); d2 = float(np.abs(m.forward(ref["tokens"][5:]) - r[-1]).max())
assert d2 < 1e-3, ("cached prompt differs", d2)
print("ok", d, d2)
`;

for (const mode of ["lib", "nolib"])
  test(`matches the PyTorch model exactly (tokenizer: ${mode === "lib" ? "tokenizers library" : "built-in"})`, { skip }, (t) => {
    if (mode === "lib" && spawnSync(py, ["-c", "import tokenizers"]).status !== 0) return t.skip("tokenizers not installed");
    const r = spawnSync(py, ["-I", "-c", CHECK, root, mode], { encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr || r.stdout);
  });

test("answers in the terminal (--prompt)", { skip }, () => {
  const r = spawnSync(py, [path.join(root, "mnx7b/run.py"), "-m", fixture, "--prompt", "Hello there"], { encoding: "utf8", timeout: 60000 });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /Mnx 7B loaded/);
  assert.match(r.stderr, /tokens at/);
});

test("Mnx detects it, starts it and chats with it (streaming)", { skip, timeout: 120000 }, async () => {
  const port = 37000 + Math.floor(Math.random() * 1000);
  const env = { ...process.env, PORT: String(port), HOST: "127.0.0.1", MNX_LOCAL_MODEL: fixture, MNX_LLAMA_PORT: String(port + 1), MNX_PREWARM: "off" };
  delete env.ANTHROPIC_API_KEY;
  const server = spawn(process.execPath, ["server.js"], { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });
  try {
    await new Promise((resolve) => server.stdout.on("data", (d) => String(d).includes("Mnx is live") && resolve()));
    const cfg = await (await fetch(`http://127.0.0.1:${port}/api/config`)).json();
    assert.equal(cfg.models.local, "Mnx 7B (your model)");
    const res = await fetch(`http://127.0.0.1:${port}/api/chat`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "local", messages: [{ role: "user", content: "Hello there, how are you?" }] }) });
    const events = (await res.text()).split("\n\n").filter((l) => l.startsWith("data: ")).map((l) => JSON.parse(l.slice(6)));
    assert.equal(events.filter((e) => e.t === "error").length, 0, JSON.stringify(events.filter((e) => e.t === "error")));
    assert.ok(events.some((e) => e.t === "text"), "streamed text");
    assert.ok(events.some((e) => e.t === "done"));
  } finally {
    server.kill();
  }
});

// Keeps the training pipeline in sync with the app: the exported prompt must
// match local.js, generated data must pass Mnx's runtime parsers, and a model
// that answers like the reference data must score 100% in the evaluation.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { STABLE_PROMPT, LOCAL_TOOLS } from "../local.js";
import { MORE_NAMES } from "../tools-more.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const hasPython = spawnSync("python3", ["-c", "1"]).status === 0;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-train-"));

test("training/system_prompt.json matches the app's prompt (run: node training/export_prompt.mjs)", () => {
  const exported = JSON.parse(fs.readFileSync(path.join(root, "training/system_prompt.json"), "utf8"));
  assert.equal(exported.stable_prompt, STABLE_PROMPT);
  assert.deepEqual(exported.tools, [...LOCAL_TOOLS.map((t) => t.function.name), ...MORE_NAMES]);
});

test("every code task compiles/runs (hand-written, 10-language programs, generated families)", { skip: !hasPython && "python3 not installed", timeout: 900000 }, () => {
  const r = spawnSync("python3", [path.join(root, "training/check_code.py")], { encoding: "utf8" });
  assert.equal(r.status, 0, r.stdout + r.stderr);
});

test("generated data passes Mnx's runtime parsers", { skip: !hasPython && "python3 not installed" }, () => {
  execFileSync("python3", [path.join(root, "training/generate.py"), "--train", "300", "--test", "60", "--out", tmp]);
  const r = spawnSync(process.execPath, [path.join(root, "training/validate.mjs"), path.join(tmp, "train.jsonl")], { encoding: "utf8" });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  assert.match(r.stdout, /300\/300 conversations valid/);
});

test("a perfect model scores 100% in the evaluation", { skip: !hasPython && "python3 not installed", timeout: 120000 }, () => {
  const model = path.join(tmp, "oracle.gguf");
  fs.writeFileSync(model, Buffer.alloc(1_100_000));
  const r = spawnSync(process.execPath, [path.join(root, "training/eval.mjs"), "--model", model, "--cases", path.join(tmp, "test.jsonl")], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${path.join(root, "test/fixtures/bin")}${path.delimiter}${process.env.PATH}`,
      FAKE_ORACLE: path.join(tmp, "test.jsonl"),
      FAKE_LLAMA_DELAY_MS: "0",
      MNX_LLAMA_PORT: String(36000 + Math.floor(Math.random() * 1000)),
      MNX_LOCAL_BACKEND: "server",
    },
  });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  assert.match(r.stdout, /OVERALL\s+100\.0%/);
});

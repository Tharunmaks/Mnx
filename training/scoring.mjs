// Shared scoring for training/eval.mjs and training/soak.mjs.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

// Does the code the model wrote actually compile? Uses whatever compilers
// are installed; a missing tool means "not checked" rather than a failure.
const has = (cmd) => spawnSync("sh", ["-c", `command -v ${cmd}`]).status === 0;
function syntaxError(ext, content) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-eval-"));
  try {
    let file = path.join(dir, `main.${ext}`);
    const java = ext === "java" && content.match(/public\s+class\s+(\w+)/);
    if (java) file = path.join(dir, `${java[1]}.java`);
    fs.writeFileSync(file, content);
    const run = (cmd, args) => {
      if (!has(cmd)) return null;
      const r = spawnSync(cmd, args, { cwd: dir, encoding: "utf8", timeout: 60000 });
      return r.status === 0 ? null : (r.stderr || r.stdout || "failed").trim().split("\n").slice(-3).join(" ");
    };
    switch (ext) {
      case "py": return run("python3", ["-m", "py_compile", file]);
      case "js": case "mjs": return run("node", ["--check", file]);
      case "ts": return run("tsc", ["--noEmit", "--target", "es2020", file]);
      case "c": return run("gcc", ["-fsyntax-only", file]);
      case "cpp": return run("g++", ["-std=c++17", "-fsyntax-only", file]);
      case "java": return run("javac", ["-d", dir, file]);
      case "go": return run("gofmt", ["-e", file]) ;
      case "rs": return run("rustc", ["--emit=metadata", "--crate-type=bin", "-o", path.join(dir, "out"), file]);
      case "rb": return run("ruby", ["-c", file]);
      case "php": return run("php", ["-l", file]);
      case "sh": return run("bash", ["-n", file]);
      case "json":
        try {
          JSON.parse(content);
          return null;
        } catch (e) {
          return e.message;
        }
      case "sql": return run("python3", ["-c", "import sqlite3,sys; sqlite3.connect(':memory:').executescript(open(sys.argv[1]).read())", file]);
      case "html": {
        const js = [...content.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)].map((m) => m[1]).join("\n");
        if (!js.trim()) return null;
        fs.writeFileSync(path.join(dir, "inline.js"), js);
        return run("node", ["--check", path.join(dir, "inline.js")]);
      }
      default: return null;
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// checkAnswers=false skips answer-content checks (used with real, live tool results).
export function score(ex, run, { checkAnswers = true } = {}) {
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
    if (e.syntax && first.name === "create_file" && typeof first.input.content === "string") {
      const err = syntaxError(e.syntax, first.input.content);
      if (err) fails.push(`the ${e.syntax} file doesn't compile: ${err.slice(0, 160)}`);
    }
  }
  if (run.unexpected.length) fails.push(`unexpected tools: ${run.unexpected.join(", ")}`);
  const answer = run.answer.trim();
  if (!answer || answer === "…") fails.push("no final answer");
  if (/<\/?tool_call>|<\/?think>/.test(answer)) fails.push("raw tags in the answer");
  if (checkAnswers) for (const pattern of e.answer || []) if (answer && !re(pattern).test(answer)) fails.push(`answer missing /${pattern}/`);
  return fails;
}


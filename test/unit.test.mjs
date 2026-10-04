// Unit tests: stream parsers, tool-call repair, model detection and tools.
// Run with: npm test
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { ThinkSplitter, ToolCallScanner, parseToolCall, localModelPath, localModelLabel, normalizeToolCall as n } from "../local.js";
import { runTool, parseDuckDuckGo, htmlToText, TOOLS } from "../tools.js";

const hasPython = (() => {
  try {
    execFileSync("python3", ["-c", "1"]);
    return true;
  } catch {
    return false;
  }
})();

function split(chunks) {
  const think = [];
  const text = [];
  let closes = 0;
  const s = new ThinkSplitter(
    (t) => (t === null ? closes++ : think.push(t)),
    (t) => text.push(t),
  );
  chunks.forEach((c) => s.push(c));
  s.end();
  return { think: think.join(""), text: text.join(""), closes };
}

// Feed text one character at a time: the hardest case for tag detection.
const byChar = (s) => [...s];

describe("ThinkSplitter", () => {
  test("separates thinking from the answer across chunk boundaries", () => {
    const r = split(byChar("<think>**Plan**\nDo it.</think>\n\nHello world"));
    assert.equal(r.think, "**Plan**\nDo it.");
    assert.equal(r.text, "Hello world");
    assert.equal(r.closes, 1);
  });
  test("answer without thinking passes straight through", () => {
    assert.deepEqual(split(["Hi", " there"]), { think: "", text: "Hi there", closes: 0 });
  });
  test("leading whitespace before <think> is ignored", () => {
    assert.equal(split(["  \n<thi", "nk>x</think>y"]).think, "x");
  });
  test("an unclosed think block is also shown as the answer", () => {
    const r = split(["<think>never closed"]);
    assert.equal(r.think, "never closed");
    assert.equal(r.text, "never closed");
  });
  test("text that merely starts with '<t' is not swallowed", () => {
    assert.equal(split(["<table>", "</table>"]).text, "<table></table>");
  });
  test("handles unicode and emoji split across chunks", () => {
    assert.equal(split(byChar("<think>🤔 думаю</think>Готово ✅")).text, "Готово ✅");
  });
});

describe("ToolCallScanner", () => {
  function scan(chunks) {
    const out = { text: "", calls: [] };
    let cur = null;
    const s = new ToolCallScanner({
      onText: (t) => (out.text += t),
      onCallStart: () => (cur = ""),
      onCallChunk: (t) => (cur += t),
      onCallEnd: () => out.calls.push(cur),
    });
    chunks.forEach((c) => s.push(c));
    s.end();
    return out;
  }
  test("extracts a call split one character at a time", () => {
    const r = scan(byChar('Sure.\n<tool_call>\n{"name":"a","arguments":{}}\n</tool_call>'));
    assert.equal(r.text, "Sure.\n");
    assert.equal(r.calls.length, 1);
    assert.equal(JSON.parse(r.calls[0]).name, "a");
  });
  test("extracts multiple calls", () => {
    const r = scan(['<tool_call>{"name":"a"}</tool_call><tool_call>{"name":"b"}</tool_call>']);
    assert.deepEqual(r.calls.map((c) => JSON.parse(c).name), ["a", "b"]);
  });
  test("an unterminated call is still delivered at the end", () => {
    const r = scan(['<tool_call>{"name":"a","arguments":{"x":1}}']);
    assert.equal(r.calls.length, 1);
  });
  test("ordinary text containing '<' is left alone", () => {
    assert.equal(scan(["if a < b and <tool>"]).text, "if a < b and <tool>");
  });
});

describe("parseToolCall", () => {
  test("parses the standard Qwen format", () => {
    assert.deepEqual(parseToolCall('{"name": "search_web", "arguments": {"query": "x"}}'), { name: "search_web", input: { query: "x" } });
  });
  test("repairs raw newlines and tabs inside strings", () => {
    const r = parseToolCall('{"name": "run_code", "arguments": {"code": "a = 1\n\tprint(a)"}}');
    assert.equal(r.input.code, "a = 1\n\tprint(a)");
  });
  test("keeps escaped quotes and backslashes intact while repairing", () => {
    const r = parseToolCall('{"name": "run_code", "arguments": {"code": "print(\\"hi\\\\n\\")\nx=1"}}');
    assert.equal(r.input.code, 'print("hi\\n")\nx=1');
  });
  test("accepts code fences, trailing commas and string arguments", () => {
    assert.equal(parseToolCall('```json\n{"name":"a","arguments":{"q":1,},}\n```').input.q, 1);
    assert.equal(parseToolCall('{"name":"a","arguments":"{\\"q\\":2}"}').input.q, 2);
    assert.equal(parseToolCall('{"name":"a","parameters":{"q":3}}').input.q, 3);
  });
  test("returns null for garbage or a missing name", () => {
    assert.equal(parseToolCall("oops"), null);
    assert.equal(parseToolCall('{"arguments":{}}'), null);
    assert.equal(parseToolCall('{"name":"a","arguments":{"prompt": oops}'), null);
  });
  test("non-object arguments become an empty object", () => {
    assert.deepEqual(parseToolCall('{"name":"a","arguments":[1,2]}').input, {});
  });
});

describe("local model detection", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-models-"));
  fs.mkdirSync(path.join(dir, "models"));
  test("no models folder content → null", () => {
    assert.equal(localModelPath(dir), null);
  });
  test("skips Git LFS pointer files", () => {
    fs.writeFileSync(path.join(dir, "models", "mnx-q4_k_m.gguf"), "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1929902592\n");
    assert.equal(localModelPath(dir), null);
  });
  test("prefers mnx* over other models", () => {
    fs.writeFileSync(path.join(dir, "models", "aaa-llama-3.2-1b.gguf"), Buffer.alloc(1_100_000));
    assert.match(localModelPath(dir), /aaa-llama/);
    fs.writeFileSync(path.join(dir, "models", "mnx-q4_k_m.gguf"), Buffer.alloc(1_100_000));
    assert.match(localModelPath(dir), /mnx-q4_k_m\.gguf$/);
  });
  test("labels", () => {
    assert.equal(localModelLabel("/m/mnx-q4_k_m.gguf"), "Local (your model)");
    assert.equal(localModelLabel("/m/llama-3.2-1b-instruct.Q4_K_M.gguf"), "Local · Llama 3.2 1B");
    assert.equal(localModelLabel("/m/qwen2.5-3b-instruct-q4_k_m.gguf"), "Local · Qwen 2.5 3B");
  });
});

describe("web helpers", () => {
  test("parseDuckDuckGo unwraps redirect links, skips ads, decodes entities", () => {
    const html = `<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fx&amp;rut=1">A &amp; B</a><a class="result__snippet" href="#">Snip <b>bold</b></a>
      <a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a>
      <a class="result__a" href="https://c.org/">C</a>`;
    assert.deepEqual(parseDuckDuckGo(html), [
      { title: "A & B", url: "https://a.com/x", snippet: "Snip bold" },
      { title: "C", url: "https://c.org/", snippet: "" },
    ]);
  });
  test("htmlToText drops scripts/styles and keeps paragraphs", () => {
    const t = htmlToText("<html><style>x{}</style><script>alert(1)</script><p>One &amp; two</p><p>Three</p></html>");
    assert.equal(t, "One & two\nThree");
  });
});

describe("tools", () => {
  test("unknown tools and bad input are reported, not thrown", async () => {
    assert.match((await runTool("nope", {})).error, /Unknown tool/);
    assert.match((await runTool("create_file", null)).error, /JSON object/);
    assert.match((await runTool("create_file", { filename: "a.py" })).error, /required/);
  });
  test("create_file sanitizes the filename", async () => {
    const r = await runTool("create_file", { filename: "../../etc/pass:wd", content: "x" });
    assert.equal(r.display.filename, ".._.._etc_pass_wd");
  });
  test("create_document validates format-specific content", async () => {
    assert.match((await runTool("create_document", { format: "pptx", title: "T" })).error, /slide/);
    assert.match((await runTool("create_document", { format: "pdf", title: "T" })).error, /section/);
    const ok = await runTool("create_document", { format: "pptx", title: "T", slides: [{ title: "S", bullets: ["a", 5] }] });
    assert.equal(ok.display.filename, "T.pptx");
    assert.deepEqual(ok.display.slides[0].bullets, ["a"]);
  });
  test("create_image builds a Pollinations URL with the aspect", async () => {
    const r = await runTool("create_image", { prompt: "a cat & dog", aspect: "portrait" });
    assert.match(r.display.url, /^https:\/\/image\.pollinations\.ai\/prompt\/a%20cat%20%26%20dog\?width=768&height=1280/);
  });
  test("location tool without a shared location asks the user", async () => {
    const r = await runTool("get_user_location", {}, {});
    assert.equal(r.result.available, false);
  });
  test("run_code is offered to the online models", () => {
    assert.ok(TOOLS.some((t) => t.name === "run_code"));
  });
});

describe("code runner safety", () => {
  const marker = path.join(os.tmpdir(), `mnx-marker-${process.pid}`);
  const touch = `open(${JSON.stringify(marker)}, "w").write("ran")\nprint("ok")`;

  test("never runs without an approval function", async () => {
    fs.rmSync(marker, { force: true });
    const r = await runTool("run_code", { language: "python", code: touch }, {});
    assert.equal(r.result.ran, false);
    assert.equal(fs.existsSync(marker), false);
  });
  test("never runs when the user declines", async () => {
    let asked = null;
    const r = await runTool("run_code", { language: "python", code: touch }, { requestApproval: async (p) => ((asked = p), false) });
    assert.equal(r.display.declined, true);
    assert.equal(asked.code, touch);
    assert.equal(fs.existsSync(marker), false);
  });
  test("rejects empty and oversized programs before asking", async () => {
    let asked = false;
    const ctx = { requestApproval: async () => ((asked = true), true) };
    assert.match((await runTool("run_code", { language: "python", code: "  " }, ctx)).error, /required/);
    assert.match((await runTool("run_code", { language: "python", code: "x".repeat(100_001) }, ctx)).error, /too long/);
    assert.equal(asked, false);
  });

  const yes = { requestApproval: async () => true };
  test("JavaScript runs and reports output", async () => {
    const r = await runTool("run_code", { language: "javascript", code: "console.log(6*7); console.error('warn')" }, yes);
    assert.equal(r.result.exit_code, 0);
    assert.equal(r.result.stdout.trim(), "42");
    assert.equal(r.result.stderr.trim(), "warn");
  });
  test("non-zero exit codes and errors are reported", async () => {
    const r = await runTool("run_code", { language: "javascript", code: "throw new Error('boom')" }, yes);
    assert.notEqual(r.result.exit_code, 0);
    assert.match(r.result.stderr, /boom/);
  });
  test("API keys and other secrets are not passed to the program", async () => {
    process.env.ANTHROPIC_API_KEY = "sk-should-not-leak";
    process.env.HF_TOKEN = "hf_should_not_leak";
    const r = await runTool(
      "run_code",
      { language: "javascript", code: "console.log(JSON.stringify([process.env.ANTHROPIC_API_KEY, process.env.HF_TOKEN]))" },
      yes,
    );
    assert.equal(r.result.stdout.trim(), "[null,null]");
  });
  test("the temporary folder is removed afterwards", async () => {
    const r = await runTool("run_code", { language: "javascript", code: "console.log(process.cwd())" }, yes);
    assert.equal(fs.existsSync(r.result.stdout.trim()), false);
  });
  test("long-running programs are killed at the timeout", { timeout: 15000 }, async () => {
    process.env.MNX_CODE_TIMEOUT_MS = "800";
    const t0 = Date.now();
    const r = await runTool("run_code", { language: "javascript", code: "setInterval(()=>{}, 1000)" }, yes);
    delete process.env.MNX_CODE_TIMEOUT_MS;
    assert.equal(r.result.timed_out, true);
    assert.ok(Date.now() - t0 < 5000);
  });
  test("huge output is capped", async () => {
    const r = await runTool("run_code", { language: "javascript", code: "for (let i=0;i<20000;i++) console.log('line '+i)" }, yes);
    assert.ok(r.display.stdout.length <= 20000);
    assert.ok(r.result.stdout.length <= 4000);
  });
  test("Python runs, and saved images are returned", { skip: !hasPython && "python3 not installed" }, async () => {
    // A tiny valid PNG written by the program.
    const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    const code = `import base64\nopen("dot.png","wb").write(base64.b64decode("${png}"))\nprint("π ≈", 3.14159)`;
    const r = await runTool("run_code", { language: "python", code }, yes);
    assert.equal(r.result.exit_code, 0, r.result.stderr);
    assert.equal(r.result.stdout.trim(), "π ≈ 3.14159");
    assert.deepEqual(r.result.images_shown_to_user, ["dot.png"]);
    assert.match(r.display.images[0].src, /^data:image\/png;base64,/);
  });
});

describe("tool-call normalization (smooth tool use)", () => {

  test("maps near-miss tool names", () => {
    for (const [alias, real] of [["web_search", "search_web"], ["generate_image", "create_image"], ["write_file", "create_file"], ["execute_code", "run_code"], ["weather", "get_weather"]])
      assert.equal(n(alias, {}).name, real);
    assert.equal(n("delete_everything", {}).name, "delete_everything"); // unknown stays unknown
  });
  test("maps near-miss argument names and keeps correct ones", () => {
    assert.deepEqual(n("create_file", { file_name: "a.py", code: "x=1" }).input, { filename: "a.py", content: "x=1" });
    assert.deepEqual(n("get_weather", { city: "Chennai" }).input, { location: "Chennai" });
    assert.deepEqual(n("search_web", { query: "a", q: "b" }).input, { query: "a", q: "b" });
  });
  test("strips markdown fences around file contents and code", () => {
    assert.equal(n("create_file", { filename: "a.js", content: "```js\nconsole.log(1)\n```" }).input.content, "console.log(1)\n");
    assert.equal(n("run_code", { language: "py", code: "```python\nprint(1)\n```" }).input.code, "print(1)\n");
  });
  test("normalizes enums: language, travel mode, image aspect, document format", () => {
    assert.equal(n("run_code", { lang: "node", code: "1" }).input.language, "javascript");
    assert.equal(n("get_directions", { from: "a", to: "b", mode: "bike" }).input.mode, "cycling");
    assert.equal(n("create_image", { prompt: "x", aspect: "16:9" }).input.aspect, "landscape");
    assert.equal(n("create_document", { format: "PowerPoint", title: "t", slides: ["Intro"] }).input.format, "pptx");
    assert.deepEqual(n("create_document", { format: "pptx", title: "t", slides: ["Intro"] }).input.slides, [{ title: "Intro", bullets: [] }]);
  });
});

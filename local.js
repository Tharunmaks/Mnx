// Runs a local GGUF model (e.g. the Mnx Qwen2.5-3B fine-tune) and streams it
// to the browser using the same events as the Claude path. The model writes
// its reasoning inside <think>…</think> first; that streams into the thinking
// area, and the rest becomes the answer.
//
// Two backends:
//  - in-process llama.cpp via node-llama-cpp (desktop Linux/macOS/Windows)
//  - llama.cpp's own `llama-server` over HTTP (Android/Termux, or whenever
//    MNX_LLAMA_SERVER is set or node-llama-cpp isn't installed). Mnx starts
//    llama-server itself if it's on the PATH (Termux: `pkg install llama-cpp`).
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { TOOLS, LOCAL_ONLY_TOOLS, runTool } from "./tools.js";

export const LOCAL_ID = "local";
const ON_ANDROID = process.platform === "android" || !!process.env.TERMUX_VERSION;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// MNX_LOCAL_MODEL, else the first .gguf in models/ (preferring mnx*).
export function localModelPath(baseDir) {
  if (process.env.MNX_LOCAL_MODEL) return path.resolve(baseDir, process.env.MNX_LOCAL_MODEL);
  const dir = path.join(baseDir, "models");
  let files = [];
  try {
    // Skip Git LFS placeholder files (small text pointers left when the
    // real model hasn't been downloaded with `git lfs pull`).
    files = fs
      .readdirSync(dir)
      .filter((f) => /\.gguf$/i.test(f))
      .filter((f) => {
        try {
          return fs.statSync(path.join(dir, f)).size > 1_000_000;
        } catch {
          return false;
        }
      })
      .sort();
  } catch {
    /* no models folder */
  }
  const pick = files.find((f) => /^mnx/i.test(f)) || files[0];
  return pick ? path.join(dir, pick) : null;
}

export function localModelLabel(file) {
  const name = path.basename(file).replace(/\.gguf$/i, "");
  if (/^mnx/i.test(name)) return "Local (your model)";
  const m = name.match(/(llama|qwen|gemma|phi|mistral)[-_]?(\d+(?:\.\d+)?)?[-_]?(\d+(?:\.\d+)?b)?/i);
  if (m) return `Local · ${m[1][0].toUpperCase()}${m[1].slice(1).toLowerCase()}${m[2] ? ` ${m[2]}` : ""}${m[3] ? ` ${m[3].toUpperCase()}` : ""}`;
  return `Local · ${name}`;
}

// Qwen2.5-Instruct's recommended sampling settings (override with env vars).
const SAMPLING = {
  temperature: Number(process.env.MNX_TEMPERATURE) || 0.7,
  top_p: Number(process.env.MNX_TOP_P) || 0.8,
  top_k: Number(process.env.MNX_TOP_K) || 20,
  repeat_penalty: Number(process.env.MNX_REPEAT_PENALTY) || 1.05,
};

/* ───────────── Backend 1: node-llama-cpp (in process) ───────────── */
let loading = null;
async function loadInProcess(file) {
  loading ||= (async () => {
    const { getLlama, LlamaChatSession } = await import("node-llama-cpp");
    const llama = await getLlama();
    const model = await llama.loadModel({ modelPath: file });
    const context = await model.createContext({ contextSize: { max: 8192 } });
    return { file, LlamaChatSession, context };
  })();
  try {
    const loaded = await loading;
    if (loaded.file !== file) throw new Error("The local model file changed — restart Mnx to load the new one.");
    return loaded;
  } catch (err) {
    loading = null;
    throw err;
  }
}

async function generateInProcess(file, history, prompt, onChunk, signal) {
  const { LlamaChatSession, context } = await loadInProcess(file);
  const sequence = context.getSequence();
  const session = new LlamaChatSession({ contextSequence: sequence, autoDisposeSequence: false });
  session.setChatHistory(
    history.map((h) => (h.role === "system" ? { type: "system", text: h.content } : h.role === "user" ? { type: "user", text: h.content } : { type: "model", response: [h.content] })),
  );
  try {
    await session.prompt(prompt, {
      maxTokens: 4096,
      temperature: SAMPLING.temperature,
      topP: SAMPLING.top_p,
      topK: SAMPLING.top_k,
      repeatPenalty: { penalty: SAMPLING.repeat_penalty },
      signal,
      stopOnAbortSignal: true,
      onTextChunk: onChunk,
    });
  } finally {
    session.dispose({ disposeSequence: false });
    sequence.dispose();
  }
}

/* ───────────── Backend 2: llama-server (HTTP) ───────────── */
function findOnPath(name) {
  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    const p = path.join(dir, name);
    try {
      fs.accessSync(p, fs.constants.X_OK);
      return p;
    } catch {
      /* keep looking */
    }
  }
  return null;
}

let serverStart = null;
async function ensureLlamaServer(file) {
  if (process.env.MNX_LLAMA_SERVER) return process.env.MNX_LLAMA_SERVER.replace(/\/+$/, "");
  serverStart ||= (async () => {
    const bin = process.env.MNX_LLAMA_SERVER_BIN || findOnPath("llama-server");
    if (!bin) throw new Error("llama-server wasn't found. On Termux run: pkg install llama-cpp");
    const port = Number(process.env.MNX_LLAMA_PORT) || 8089;
    const args = ["-m", file, "--host", "127.0.0.1", "--port", String(port), "-c", String(Number(process.env.MNX_CTX) || 8192)];
    if (process.env.MNX_THREADS) args.push("-t", process.env.MNX_THREADS);
    console.log(`[local] starting ${bin} ${args.join(" ")}`);
    const child = spawn(bin, args, { stdio: ["ignore", "ignore", "pipe"] });
    let log = "";
    let exited = null;
    child.stderr.on("data", (d) => (log = (log + d).slice(-2000)));
    child.on("exit", (code) => {
      exited = code ?? "signal";
      serverStart = null;
    });
    child.on("error", (e) => (exited = e.message));
    const stop = () => child.kill();
    process.once("exit", stop);
    for (const sig of ["SIGINT", "SIGTERM"]) process.once(sig, () => (stop(), process.exit(0)));

    const url = `http://127.0.0.1:${port}`;
    for (let i = 0; i < 600; i++) {
      if (exited !== null) throw new Error(`llama-server stopped (${exited}). Last output:\n${log.trim().split("\n").slice(-6).join("\n")}`);
      try {
        const r = await fetch(`${url}/health`);
        if (r.ok) return url;
      } catch {
        /* not up yet */
      }
      await sleep(500);
    }
    stop();
    throw new Error("llama-server took too long to load the model.");
  })();
  try {
    return await serverStart;
  } catch (err) {
    serverStart = null;
    throw err;
  }
}

async function generateViaServer(file, history, prompt, onChunk, signal) {
  const url = await ensureLlamaServer(file);
  const res = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal,
    body: JSON.stringify({
      messages: [...history, { role: "user", content: prompt }],
      stream: true,
      max_tokens: 4096,
      ...SAMPLING,
    }),
  });
  if (!res.ok || !res.body) throw new Error(`llama-server returned HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const dec = new TextDecoder();
  let buf = "";
  try {
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        let json;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        const delta = json.choices?.[0]?.delta || {};
        // Reasoning models served with --reasoning-format put thoughts here.
        if (delta.reasoning_content) onChunk(delta.reasoning_content, "reasoning");
        if (delta.content) onChunk(delta.content);
      }
    }
  } catch (err) {
    if (signal?.aborted) return;
    throw err;
  }
}

let useServer = ON_ANDROID || !!process.env.MNX_LLAMA_SERVER || process.env.MNX_LOCAL_BACKEND === "server";
async function generate(file, history, prompt, onChunk, signal) {
  if (!useServer) {
    try {
      return await generateInProcess(file, history, prompt, onChunk, signal);
    } catch (err) {
      // node-llama-cpp not installed (e.g. npm install --omit=optional): use llama-server.
      if (err?.code !== "ERR_MODULE_NOT_FOUND") throw err;
      useServer = true;
    }
  }
  return generateViaServer(file, history, prompt, onChunk, signal);
}

/* ───────────── Shared ───────────── */
// One generation at a time.
let queue = Promise.resolve();
function exclusive(fn) {
  const run = queue.then(fn, fn);
  queue = run.catch(() => {});
  return run;
}

const blocksText = (content) =>
  typeof content === "string"
    ? content
    : (content || [])
        .map((b) => {
          if (b.type === "text") return b.text;
          if (b.type === "image") return "[The user attached an image, which this local model can't see.]";
          if (b.type === "document") return `[The user attached a PDF${b.title ? ` "${b.title}"` : ""}, which this local model can't read.]`;
          return "";
        })
        .filter(Boolean)
        .join("\n\n");

// Tools the local model can call. Kept compact: on a phone, every token of
// the system prompt costs prompt-processing time.
const LOCAL_TOOL_NAMES = ["search_web", "read_webpage", "get_weather", "get_user_location", "find_places", "get_directions", "create_image", "create_document", "create_file", "run_code"];
const toolSpecs = () => {
  const all = [...LOCAL_ONLY_TOOLS, ...TOOLS];
  const strip = (schema) => JSON.parse(JSON.stringify(schema, (k, v) => (k === "additionalProperties" || k === "minimum" || k === "maximum" ? undefined : v)));
  return LOCAL_TOOL_NAMES.map((n) => all.find((t) => t.name === n))
    .filter(Boolean)
    .map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: strip(t.input_schema) } }));
};

function systemPrompt(userName) {
  return `You are Mnx, a friendly and helpful AI assistant running locally on the user's own device. Today's date is ${new Date().toISOString().slice(0, 10)}.${
    userName ? ` The user's name is ${userName}.` : ""
  }
Help with anything legal; politely decline only requests that would facilitate crimes or serious harm.

Before every reply, think briefly inside <think></think> tags: start with a short bold title like **Understanding the question**, then 1-3 sentences about what the user needs and whether a tool is needed. After </think>, either call a tool or write the final answer in markdown.

How to write great answers:
- Start with the direct answer in the first sentence, then add the details that matter.
- Format with markdown: short paragraphs, **bold** key terms, numbered lists for steps, bullet lists for options, tables for comparisons, and fenced code blocks with the language name for code.
- Fit the length to the question: a quick question gets 1-3 sentences; explanations, how-tos and plans get clear sections with ### headings.
- Be specific: real numbers, names and examples. If you're not sure about something, say so instead of guessing.
- After using tools, explain what you found in your own words, cite web sources as [title](url), and mention any image, file, slides or code results you made.

Use tools when they help: search_web for current events, facts you're unsure of or anything recent (then read_webpage for details); get_weather for weather; find_places / get_directions for places and routes; create_image to draw or generate a picture; create_document for slides (pptx), PDF or Word files; create_file for code or text files; run_code to calculate, analyse data or make charts (the user approves each run). When you receive a <tool_response>, use it to answer. Never invent tool results, and don't paste file contents or image links that were already shown to the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${toolSpecs().map((t) => JSON.stringify(t)).join("\n")}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>`;
}

// Splits the stream into thinking / answer as it arrives.
export class ThinkSplitter {
  constructor(onThink, onText) {
    this.onThink = onThink;
    this.onText = onText;
    this.mode = "start"; // start | think | text
    this.buf = "";
    this.thought = "";
  }
  push(chunk) {
    this.buf += chunk;
    for (;;) {
      if (this.mode === "start") {
        const t = this.buf.trimStart();
        if (t.startsWith("<think>")) {
          this.buf = t.slice(7);
          this.mode = "think";
        } else if ("<think>".startsWith(t)) return; // might still become the tag
        else {
          this.mode = "text";
        }
      } else if (this.mode === "think") {
        const end = this.buf.indexOf("</think>");
        if (end >= 0) {
          this.thought += this.buf.slice(0, end);
          this.onThink(this.buf.slice(0, end));
          this.buf = this.buf.slice(end + 8).replace(/^\s+/, "");
          this.mode = "text";
          this.onThink(null);
          continue;
        }
        const keep = partialTagLength(this.buf, "</think>");
        if (this.buf.length > keep) {
          this.thought += this.buf.slice(0, this.buf.length - keep);
          this.onThink(this.buf.slice(0, this.buf.length - keep));
        }
        this.buf = this.buf.slice(this.buf.length - keep);
        return;
      } else {
        if (this.buf) this.onText(this.buf);
        this.buf = "";
        return;
      }
    }
  }
  end() {
    if (this.mode === "think") {
      // The model never closed its <think>: show what it wrote as the answer too.
      this.thought += this.buf;
      this.onThink(this.buf);
      this.onThink(null);
      if (this.thought.trim()) this.onText(this.thought.trim());
    } else if (this.buf) this.onText(this.buf);
    this.buf = "";
  }
}
function partialTagLength(s, tag) {
  for (let n = Math.min(tag.length - 1, s.length); n > 0; n--) if (tag.startsWith(s.slice(-n))) return n;
  return 0;
}

// Pulls <tool_call>…</tool_call> sections out of the answer stream.
export class ToolCallScanner {
  constructor({ onText, onCallStart, onCallChunk, onCallEnd }) {
    Object.assign(this, { onText, onCallStart, onCallChunk, onCallEnd });
    this.buf = "";
    this.inCall = false;
  }
  push(chunk) {
    this.buf += chunk;
    for (;;) {
      const tag = this.inCall ? "</tool_call>" : "<tool_call>";
      const at = this.buf.indexOf(tag);
      if (at >= 0) {
        const before = this.buf.slice(0, at);
        this.buf = this.buf.slice(at + tag.length);
        if (this.inCall) {
          if (before) this.onCallChunk(before);
          this.onCallEnd();
        } else {
          if (before) this.onText(before);
          this.onCallStart();
        }
        this.inCall = !this.inCall;
        continue;
      }
      const keep = partialTagLength(this.buf, tag);
      const out = this.buf.slice(0, this.buf.length - keep);
      this.buf = this.buf.slice(this.buf.length - keep);
      if (out) (this.inCall ? this.onCallChunk : this.onText)(out);
      return;
    }
  }
  end() {
    if (this.inCall) {
      if (this.buf) this.onCallChunk(this.buf);
      this.onCallEnd();
    } else if (this.buf) this.onText(this.buf);
    this.buf = "";
    this.inCall = false;
  }
}

// Small models often put raw newlines/tabs inside JSON strings (e.g. code).
function escapeControlCharsInStrings(json) {
  let out = "";
  let inString = false;
  for (let i = 0; i < json.length; i++) {
    const ch = json[i];
    if (inString) {
      if (ch === "\\") {
        out += ch + (json[i + 1] ?? "");
        i++;
        continue;
      }
      if (ch === '"') inString = false;
      else if (ch === "\n") {
        out += "\\n";
        continue;
      } else if (ch === "\r") continue;
      else if (ch === "\t") {
        out += "\\t";
        continue;
      }
    } else if (ch === '"') inString = true;
    out += ch;
  }
  return out;
}

// Parse a tool call body like {"name": "x", "arguments": {...}}, tolerating
// small-model slips (code fences, trailing commas, arguments as a string).
export function parseToolCall(raw) {
  let t = raw.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  const first = t.indexOf("{");
  const last = t.lastIndexOf("}");
  if (first < 0 || last < first) return null;
  t = t.slice(first, last + 1);
  let obj;
  const fixed = escapeControlCharsInStrings(t);
  for (const candidate of [t, fixed, fixed.replace(/,\s*([}\]])/g, "$1")]) {
    try {
      obj = JSON.parse(candidate);
      break;
    } catch {
      /* try the next repair */
    }
  }
  if (!obj || typeof obj.name !== "string") return null;
  let args = obj.arguments ?? obj.parameters ?? {};
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = {};
    }
  }
  return { name: obj.name, input: args && typeof args === "object" && !Array.isArray(args) ? args : {} };
}

const MAX_STEPS = 5;

export async function handleLocalChat({ body, messages, file, send, isClosed, signal, requestApproval }) {
  if (!file || !fs.existsSync(file)) {
    send({
      t: "error",
      text: "No local model found. Put your .gguf file in the models/ folder (or run npm run get-model) and restart Mnx.",
    });
    return;
  }
  const ctx = { location: body.location && Number.isFinite(body.location.latitude) ? body.location : null };

  // Convert the shared (Claude-format) history into plain chat turns.
  const chat = [{ role: "system", content: systemPrompt(String(body.userName || "").slice(0, 60)) }];
  for (const m of messages.slice(0, -1)) {
    const text = blocksText(m.content);
    if (!text) continue;
    const prev = chat[chat.length - 1];
    const role = m.role === "user" ? "user" : "assistant";
    if (prev.role === role) prev.content += `\n\n${text}`;
    else if (role === "user" || prev.role === "user") chat.push({ role, content: text });
  }
  // Drop the oldest turns if the history is too long for the context.
  const budget = chat[0].content.length + 8000;
  while (chat.length > 3 && JSON.stringify(chat).length > budget) chat.splice(1, 2);
  chat.push({ role: "user", content: blocksText(messages[messages.length - 1].content) });

  send({ t: "phase", phrases: ["Waking up your local model", "Loading Mnx into memory", "Warming up the neurons"] });

  await exclusive(async () => {
    const shown = []; // user-visible answer text across steps

    for (let it = 0; it < MAX_STEPS && !isClosed(); it++) {
      send({ t: "iteration", it });
      let index = -1;
      let thinkingOpen = false;
      let textOpen = false;
      let raw = ""; // what the model wrote, minus thinking (kept in its history)
      let answer = "";
      const calls = [];
      let call = null;

      const closeOpen = () => {
        if (thinkingOpen || textOpen) send({ t: "block_stop", it, i: index });
        thinkingOpen = textOpen = false;
      };
      const openThinking = () => {
        if (thinkingOpen) return;
        closeOpen();
        index++;
        thinkingOpen = true;
        send({ t: "block_start", it, i: index, block: { type: "thinking" } });
      };
      const scanner = new ToolCallScanner({
        onText: (t) => {
          raw += t;
          t = t.replace(/<\/?think>/g, "");
          if (!textOpen && !t.trim()) return;
          if (!textOpen) {
            closeOpen();
            index++;
            textOpen = true;
            send({ t: "block_start", it, i: index, block: { type: "text" } });
          }
          answer += t;
          send({ t: "text", it, i: index, text: t });
        },
        onCallStart: () => {
          closeOpen();
          call = { json: "", id: `local_${it}_${calls.length}_${Date.now().toString(36)}`, index: null };
        },
        onCallChunk: (t) => {
          call.json += t;
          if (call.index === null) {
            const name = call.json.match(/"name"\s*:\s*"([^"]+)"/)?.[1];
            if (!name) return;
            call.index = ++index;
            send({ t: "block_start", it, i: call.index, block: { type: "tool_use", id: call.id, name, input: {} } });
            send({ t: "input", it, i: call.index, json: call.json });
          } else send({ t: "input", it, i: call.index, json: t });
        },
        onCallEnd: () => {
          raw += `<tool_call>\n${call.json.trim()}\n</tool_call>`;
          if (call.index !== null) send({ t: "block_stop", it, i: call.index });
          calls.push(call);
          call = null;
        },
      });
      const splitter = new ThinkSplitter(
        (t) => {
          if (t === null) {
            if (thinkingOpen) closeOpen();
            return;
          }
          openThinking();
          if (t) send({ t: "thinking", it, i: index, text: t });
        },
        (t) => scanner.push(t),
      );

      await generate(
        file,
        chat.slice(0, -1),
        chat[chat.length - 1].content,
        (chunk, kind) => {
          if (isClosed()) return;
          if (kind === "reasoning") {
            openThinking();
            send({ t: "thinking", it, i: index, text: chunk });
          } else splitter.push(chunk);
        },
        signal,
      );
      splitter.end();
      scanner.end();
      if (isClosed()) return;
      closeOpen();
      if (answer.trim()) shown.push(answer.trim());

      if (!calls.length) break;
      if (it === MAX_STEPS - 1) {
        send({ t: "notice", level: "warn", text: "Mnx used the maximum number of tool steps for one reply." });
        break;
      }

      // Run the requested tools on this device and hand the results back.
      chat.push({ role: "assistant", content: raw.trim() });
      const responses = [];
      for (const c of calls) {
        const parsed = parseToolCall(c.json);
        if (!parsed) {
          if (c.index !== null) send({ t: "tool_done", id: c.id, name: c.json.match(/"name"\s*:\s*"([^"]+)"/)?.[1] || "tool", error: true });
          responses.push({ error: "Your tool call wasn't valid JSON. Use the exact <tool_call> format with properly escaped strings." });
          continue;
        }
        const known = LOCAL_TOOL_NAMES.includes(parsed.name);
        if (c.index === null) {
          c.index = ++index;
          send({ t: "block_start", it, i: c.index, block: { type: "tool_use", id: c.id, name: parsed.name, input: {} } });
          send({ t: "block_stop", it, i: c.index });
        }
        send({ t: "tool_run", id: c.id, name: parsed.name, input: parsed.input });
        const out = known ? await runTool(parsed.name, parsed.input, {
              ...ctx,
              requestApproval: (p) => requestApproval({ ...p, tool_use_id: c.id }),
            }) : { error: `Unknown tool "${parsed.name}". Available: ${LOCAL_TOOL_NAMES.join(", ")}` };
        if (isClosed()) return;
        send({ t: "tool_done", id: c.id, name: parsed.name, error: !!out.error, display: out.display });
        let json = JSON.stringify(out.error ? { error: out.error } : out.result);
        if (json.length > 3500) json = `${json.slice(0, 3500)}…(truncated)`;
        responses.push(json);
      }
      chat.push({
        role: "user",
        content: responses.map((r) => `<tool_response>\n${typeof r === "string" ? r : JSON.stringify(r)}\n</tool_response>`).join("\n"),
      });
      send({ t: "phase", phrases: ["Reading the results", "Making sense of it all", "Writing your answer"] });
    }

    if (isClosed()) return;
    send({ t: "assistant_turn", content: [{ type: "text", text: shown.join("\n\n") || "…" }], model: LOCAL_ID });
    send({ t: "done" });
  });
}

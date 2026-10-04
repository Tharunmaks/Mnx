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
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { CODE_RUNNER, runTool } from "./tools.js";

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
    // One long-lived sequence: llama.cpp reuses the already-processed prefix
    // (system prompt + earlier turns) instead of re-reading it every time.
    return { file, LlamaChatSession, sequence: context.getSequence() };
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
  const { LlamaChatSession, sequence } = await loadInProcess(file);
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

// What the local model is doing, so the page can say so instead of looking stuck.
export const localStatus = { state: "idle", progress: 0, error: "" };
function setStatus(state, progress = 0, error = "") {
  const changed = state !== localStatus.state || Math.floor(progress / 10) !== Math.floor(localStatus.progress / 10);
  Object.assign(localStatus, { state, progress, error });
  if (changed && state === "warming") console.log(`[local] getting ready: ${progress}%`);
}
export function localStatusText() {
  if (localStatus.state === "loading") return "Loading your model into memory";
  if (localStatus.state === "warming") return `Getting Mnx ready · ${localStatus.progress}% (first start only)`;
  return "";
}

// Phones mix fast and slow CPU cores; llama.cpp runs far faster on just the
// fast ones. Count the cores that aren't in the slowest cluster.
function fastCores() {
  try {
    const dir = "/sys/devices/system/cpu";
    const freqs = fs.readdirSync(dir).filter((d) => /^cpu\d+$/.test(d)).map((d) => {
      try {
        return Number(fs.readFileSync(`${dir}/${d}/cpufreq/cpuinfo_max_freq`, "utf8"));
      } catch {
        return 0;
      }
    }).filter(Boolean);
    if (freqs.length) {
      const slowest = Math.min(...freqs);
      const fast = freqs.filter((f) => f > slowest).length;
      return fast || freqs.length;
    }
  } catch {
    /* not Linux */
  }
  return Math.max(1, Math.floor((os.cpus().length || 4) / 2));
}

// "free" if nothing answers on the port, else the model path it serves (or "busy").
async function probeLlamaServer(url) {
  try {
    const r = await fetch(`${url}/props`, { signal: AbortSignal.timeout(1500) });
    const props = await r.json().catch(() => ({}));
    return props.model_path ? path.resolve(props.model_path) : "busy";
  } catch (err) {
    return err?.cause?.code === "ECONNREFUSED" ? "free" : "busy";
  }
}

let serverStart = null;
async function ensureLlamaServer(file) {
  if (process.env.MNX_LLAMA_SERVER) return process.env.MNX_LLAMA_SERVER.replace(/\/+$/, "");
  serverStart ||= (async () => {
    const bin = process.env.MNX_LLAMA_SERVER_BIN || findOnPath("llama-server");
    if (!bin) throw new Error("llama-server wasn't found. On Termux run: pkg install llama-cpp");
    // Android can kill Mnx without warning, leaving an old llama-server
    // running. Reuse it if it has this model loaded; otherwise find a free port.
    let port = Number(process.env.MNX_LLAMA_PORT) || 8089;
    for (let tries = 0; tries < 10; tries++, port++) {
      const state = await probeLlamaServer(`http://127.0.0.1:${port}`);
      if (state === "free") break;
      if (state === file) {
        console.log(`[local] reusing llama-server already running on port ${port}`);
        return `http://127.0.0.1:${port}`;
      }
    }
    const args = ["-m", file, "--host", "127.0.0.1", "--port", String(port), "-c", String(Number(process.env.MNX_CTX) || 8192), "-np", "1"];
    if (process.env.MNX_THREADS) args.push("-t", process.env.MNX_THREADS);
    else if (ON_ANDROID) args.push("-t", String(fastCores()));
    if (process.env.MNX_GPU_LAYERS) args.push("-ngl", process.env.MNX_GPU_LAYERS); // e.g. 99 on a GPU machine
    console.log(`[local] starting ${bin} ${args.join(" ")}`);
    setStatus("loading");
    const child = spawn(bin, args, { stdio: ["ignore", "ignore", "pipe"] });
    serverChild = child;
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

// If llama-server died (Android can kill it to free memory) the next request
// fails before any output. Restart it and retry once, so the user never
// sees that error.
let serverChild = null;
async function generateViaServer(file, history, prompt, onChunk, signal) {
  let produced = false;
  const tracked = (chunk, kind) => {
    produced = true;
    onChunk(chunk, kind);
  };
  try {
    return await generateViaServerOnce(file, history, prompt, tracked, signal);
  } catch (err) {
    const lost = /fetch failed|ECONNREFUSED|ECONNRESET|socket|other side closed/i.test(`${err?.message} ${err?.cause?.code} ${err?.cause?.message}`);
    if (!lost || produced || signal?.aborted || process.env.MNX_LLAMA_SERVER) throw err;
    console.warn("[local] llama-server connection lost; restarting it");
    serverChild?.kill();
    serverStart = null;
    return generateViaServerOnce(file, history, prompt, tracked, signal);
  }
}

// Long chats outgrow the model's context window (8192 tokens by default,
// less on phones with MNX_CTX=2048). Drop the oldest turns, keeping the
// system prompt and whole exchanges, so the newest message always fits.
const REPLY_ROOM = 1536;
export function fitHistory(history, prompt, ctx, tokensPerChar = 1 / 3) {
  const cost = (m) => Math.ceil(String(m.content).length * tokensPerChar) + 8;
  const budget = ctx - REPLY_ROOM;
  const total = (h) => h.reduce((n, m) => n + cost(m), cost({ content: prompt }));
  if (total(history) <= budget) return history;
  // Trim well below the limit so it happens rarely (each trim costs the prompt cache).
  const [system, ...rest] = history;
  const isTurnStart = (m) => m.role === "user" && !String(m.content).startsWith("<tool_response>");
  const trimTo = (limit) => {
    const kept = [...rest];
    while (kept.length && (total([system, ...kept]) > limit || !isTurnStart(kept[0]))) kept.shift();
    return kept;
  };
  const roomy = trimTo(budget * 0.7);
  return [system, ...(roomy.length ? roomy : trimTo(budget))];
}

let serverCtx = Number(process.env.MNX_CTX) || 8192;
async function generateViaServerOnce(file, history, prompt, onChunk, signal) {
  const url = await ensureLlamaServer(file);
  let res;
  let perChar = 1 / 3;
  for (let attempt = 0; ; attempt++) {
    const messages = [...fitHistory(history, prompt, serverCtx, perChar), { role: "user", content: prompt }];
    res = await fetch(`${url}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal,
      body: JSON.stringify({ messages, stream: true, cache_prompt: true, return_progress: true, max_tokens: 4096, ...SAMPLING }),
    });
    if (res.ok || res.status !== 400 || attempt >= 3) break;
    const text = await res.text();
    const info = (() => {
      try {
        return JSON.parse(text).error;
      } catch {
        return null;
      }
    })();
    if (!/exceed|context/i.test(`${info?.type} ${info?.message}`)) throw new Error(`llama-server returned HTTP 400: ${text.slice(0, 300)}`);
    // Learn the real context size and token density, then trim harder.
    if (info.n_ctx) serverCtx = info.n_ctx;
    const chars = messages.reduce((n, m) => n + String(m.content).length, 0);
    if (info.n_prompt_tokens && chars) perChar = Math.max(perChar * 1.25, (info.n_prompt_tokens / chars) * 1.1);
    else perChar *= 1.5;
    if (messages.length <= 2) throw new Error("Your message is too long for the local model's memory. Try a shorter message, or start Mnx with a larger MNX_CTX.");
  }
  if (!res.ok || !res.body) throw new Error(`llama-server returned HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  if (localStatus.state !== "ready") setStatus("ready", 100);
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
        // Prompt reading progress (slow on phones), then speed stats at the end.
        if (json.prompt_progress) {
          const { total, processed, cache } = json.prompt_progress;
          if (total > (cache || 0)) onChunk(Math.round((100 * (processed - (cache || 0))) / (total - (cache || 0))), "progress");
        }
        if (json.timings?.predicted_n) {
          const t = json.timings;
          console.log(`[local] reply: read ${t.prompt_n} new tokens in ${(t.prompt_ms / 1000).toFixed(1)}s (${t.prompt_per_second?.toFixed(1)}/s), wrote ${t.predicted_n} tokens at ${t.predicted_per_second?.toFixed(1)}/s`);
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

// Tools the local model can call. Written compactly on purpose: on a phone,
// every token of the system prompt costs prompt-processing time.
const fn = (name, description, properties = {}, required = []) => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
});
const str = { type: "string" };
export const LOCAL_TOOLS = [
  fn("search_web", "Search the internet for current or unknown information.", { query: str }, ["query"]),
  fn("read_webpage", "Read the text of a web page.", { url: str }, ["url"]),
  fn("get_weather", "Current weather and forecast. Omit location to use the user's location.", { location: str, days: { type: "integer" } }),
  fn("get_user_location", "The user's current location."),
  fn("find_places", "Find restaurants, shops, places or addresses.", { query: str, near: str }, ["query"]),
  fn("get_directions", "Route between two places. origin may be 'my location'.", { origin: str, destination: str, mode: { enum: ["driving", "walking", "cycling"] } }, ["origin", "destination"]),
  fn("create_image", "Generate an image from a detailed English description.", { prompt: str, aspect: { enum: ["square", "landscape", "portrait"] } }, ["prompt"]),
  fn(
    "create_document",
    "Make a pptx slide deck (use slides) or a pdf/docx document (use sections; body is markdown).",
    {
      format: { enum: ["pptx", "pdf", "docx"] },
      title: str,
      subtitle: str,
      slides: { type: "array", items: { type: "object", properties: { title: str, bullets: { type: "array", items: str } } } },
      sections: { type: "array", items: { type: "object", properties: { heading: str, body: str } } },
    },
    ["format", "title"],
  ),
  fn("create_file", "Give the user a code or text file.", { filename: str, content: str }, ["filename", "content"]),
  ...(CODE_RUNNER
    ? [fn("run_code", "Run Python or JavaScript on the user's device (they approve first). Print results; save charts as .png.", { language: { enum: ["python", "javascript"] }, code: str }, ["language", "code"])]
    : []),
];
const LOCAL_TOOL_NAMES = LOCAL_TOOLS.map((t) => t.function.name);

// Everything except the last line is identical between requests, so
// llama.cpp can reuse its cached processing of it.
export const STABLE_PROMPT = `You are Mnx, a friendly and helpful AI assistant running on the user's own device. Help with anything legal; politely decline only requests that would facilitate crimes or serious harm.

Before every reply, think briefly inside <think></think>: a short bold title like **Understanding the question**, then 1-3 sentences on what the user needs and whether a tool helps. After </think>, call a tool or write the answer.

Answer well:
- Give the direct answer first, then the details that matter.
- Use markdown: short paragraphs, **bold** key terms, numbered steps, bullet lists, tables for comparisons, fenced code blocks with the language.
- Quick questions get 1-3 sentences; explanations and how-tos get ### sections.
- Be specific. If unsure, say so instead of guessing.
- After tools, explain the results in your own words and cite web sources as [title](url). Never invent tool results; don't repeat files or images already shown.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${LOCAL_TOOLS.map((t) => JSON.stringify(t)).join("\n")}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>`;

export function systemPrompt(userName, date = new Date()) {
  return `${STABLE_PROMPT}\n\nToday's date is ${date.toISOString().slice(0, 10)}.${userName ? ` The user's name is ${userName}.` : ""}`;
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
          this.buf = this.buf.slice(end + 8);
          this.mode = "text";
          this.trimLead = true; // drop blank lines after </think>, even across chunks
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
        if (this.trimLead) {
          this.buf = this.buf.replace(/^\s+/, "");
          if (!this.buf) return;
          this.trimLead = false;
        }
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

// Small models sometimes use a near-miss tool or argument name ("code" for
// "content", "city" for "location", "web_search" for "search_web") or wrap
// code in markdown fences. Map those to what the tools expect, so a call that
// is clearly meant for a tool still works.
const TOOL_ALIASES = {
  web_search: "search_web", search: "search_web", google: "search_web", browse: "read_webpage", fetch_url: "read_webpage", open_url: "read_webpage",
  weather: "get_weather", get_location: "get_user_location", location: "get_user_location", places: "find_places", search_places: "find_places",
  directions: "get_directions", route: "get_directions", generate_image: "create_image", image: "create_image", draw: "create_image",
  create_presentation: "create_document", make_slides: "create_document", create_pdf: "create_document", write_file: "create_file",
  save_file: "create_file", python: "run_code", execute_code: "run_code", run_python: "run_code", code_interpreter: "run_code",
};
const ARG_ALIASES = {
  search_web: { query: ["q", "search", "keywords", "text", "search_query"] },
  read_webpage: { url: ["link", "href", "page", "website"] },
  get_weather: { location: ["city", "place", "loc", "where"] },
  find_places: { query: ["type", "what", "category", "keyword"], near: ["location", "city", "place", "area", "around"] },
  get_directions: { origin: ["from", "start", "source"], destination: ["to", "end", "dest", "target"], mode: ["travel_mode", "by", "transport"] },
  create_image: { prompt: ["description", "text", "image", "query", "subject"], aspect: ["orientation", "size", "aspect_ratio", "ratio"] },
  create_document: { format: ["type", "file_type", "kind"], slides: ["pages"], sections: ["content", "body", "chapters"] },
  create_file: { filename: ["file_name", "name", "path", "file", "filepath"], content: ["code", "text", "contents", "body", "source", "data"] },
  run_code: { language: ["lang", "runtime"], code: ["content", "source", "script", "program"] },
};
const stripFences = (text) => {
  const m = typeof text === "string" && text.trim().match(/^```[\w+#.-]*\n([\s\S]*?)\n?```$/);
  return m ? `${m[1]}\n` : text;
};

export function normalizeToolCall(name, input) {
  const tool = TOOL_ALIASES[name] || name;
  const args = { ...(input || {}) };
  for (const [key, aliases] of Object.entries(ARG_ALIASES[tool] || {})) {
    if (args[key] !== undefined) continue;
    const alias = aliases.find((a) => args[a] !== undefined);
    if (alias) {
      args[key] = args[alias];
      delete args[alias];
    }
  }
  if (tool === "create_file") args.content = stripFences(args.content);
  if (tool === "run_code") {
    args.code = stripFences(args.code);
    const lang = String(args.language || "python").toLowerCase();
    args.language = /^(js|javascript|node|nodejs)$/.test(lang) ? "javascript" : "python";
  }
  if (tool === "get_directions" && typeof args.mode === "string") {
    const m = args.mode.toLowerCase();
    args.mode = /walk|foot/.test(m) ? "walking" : /bike|bicycl|cycl/.test(m) ? "cycling" : "driving";
  }
  if (tool === "create_image" && typeof args.aspect === "string") {
    const a = args.aspect.toLowerCase();
    args.aspect = /land|wide|horiz|16:9|desktop|banner/.test(a) ? "landscape" : /port|tall|vert|9:16|phone|mobile/.test(a) ? "portrait" : "square";
  }
  if (tool === "create_document") {
    const f = String(args.format || (name === "create_pdf" ? "pdf" : "pptx")).toLowerCase().replace(/^\./, "");
    args.format = /ppt|slide|power|deck|present/.test(f) ? "pptx" : /doc|word/.test(f) ? "docx" : "pdf";
    if (Array.isArray(args.slides))
      args.slides = args.slides.map((sl) =>
        typeof sl === "string" ? { title: sl, bullets: [] } : { ...sl, bullets: sl.bullets ?? sl.points ?? sl.items ?? (typeof sl.content === "string" ? sl.content.split(/\n+/).map((x) => x.replace(/^[-*•]\s*/, "")).filter(Boolean) : []) },
      );
    if (Array.isArray(args.sections)) args.sections = args.sections.map((sec) => (typeof sec === "string" ? { body: sec } : { ...sec, body: sec.body ?? sec.content ?? sec.text ?? "" }));
  }
  return { name: tool, input: args };
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

// Start the model and process the system prompt in the background, so the
// first message doesn't wait for it. Used on the llama-server backend.
export async function warmUpLocal(file) {
  if (!useServer || !file || !fs.existsSync(file)) return false;
  return exclusive(async () => {
    const t0 = Date.now();
    try {
      setStatus("loading");
      const url = await ensureLlamaServer(file);
      setStatus("warming", 0);
      const res = await fetch(`${url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "system", content: systemPrompt("") }, { role: "user", content: "hi" }], max_tokens: 1, cache_prompt: true, stream: true, return_progress: true }),
      });
      if (!res.ok || !res.body) throw new Error(`llama-server returned HTTP ${res.status}`);
      const dec = new TextDecoder();
      for await (const chunk of res.body) {
        for (const m of dec.decode(chunk, { stream: true }).matchAll(/"prompt_progress":\{"total":(\d+),"cache":(\d+),"processed":(\d+)/g)) {
          const [total, cache, done] = m.slice(1).map(Number);
          if (total > cache) setStatus("warming", Math.min(99, Math.round((100 * (done - cache)) / (total - cache))));
        }
      }
      setStatus("ready", 100);
      console.log(`[local] model ready (system prompt cached in ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
      return true;
    } catch (err) {
      setStatus("error", 0, err.message);
      throw err;
    }
  });
}

const MAX_STEPS = 5;

// `runToolImpl` lets the evaluation harness supply fixed tool results.
export async function handleLocalChat({ body, messages, file, send, isClosed, signal, requestApproval, runToolImpl = runTool }) {
  if (!file || !fs.existsSync(file)) {
    send({
      t: "error",
      text: "No local model found. Put your .gguf file in the models/ folder (or run npm run get-model) and restart Mnx.",
    });
    return;
  }
  const ctx = { location: body.location && Number.isFinite(body.location.latitude) ? body.location : null, messages };

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

  // While the model is still loading or warming up, say so (with progress)
  // instead of showing a loader that looks stuck.
  const waiting = setInterval(() => {
    const text = localStatusText();
    if (text) send({ t: "phase", phrases: [text] });
  }, 1000);
  await exclusive(async () => {
    clearInterval(waiting);
    const shown = []; // user-visible answer text across steps

    for (let it = 0; it < MAX_STEPS && !isClosed(); it++) {
      send({ t: "iteration", it });
      let index = -1;
      let thinkingOpen = false;
      let textOpen = false;
      let raw = ""; // what the model wrote after thinking
      let thought = ""; // its <think> text; kept in history so it matches training
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
          if (t) {
            thought += t;
            send({ t: "thinking", it, i: index, text: t });
          }
        },
        (t) => scanner.push(t),
      );

      await generate(
        file,
        chat.slice(0, -1),
        chat[chat.length - 1].content,
        (chunk, kind) => {
          if (isClosed()) return;
          if (kind === "progress") {
            if (chunk < 100) send({ t: "phase", phrases: [`Reading your message · ${chunk}%`] });
            return;
          }
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
      chat.push({ role: "assistant", content: `${thought.trim() ? `<think>${thought.trim()}</think>\n` : ""}${raw.trim()}` });
      const responses = [];
      for (const c of calls) {
        const raw = parseToolCall(c.json);
        const parsed = raw && normalizeToolCall(raw.name, raw.input);
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
        const out = known ? await runToolImpl(parsed.name, parsed.input, {
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
  clearInterval(waiting);
}

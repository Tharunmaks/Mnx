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

export const LOCAL_ID = "local";
const ON_ANDROID = process.platform === "android" || !!process.env.TERMUX_VERSION;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// MNX_LOCAL_MODEL, else the first .gguf in models/ (preferring mnx*).
export function localModelPath(baseDir) {
  if (process.env.MNX_LOCAL_MODEL) return path.resolve(baseDir, process.env.MNX_LOCAL_MODEL);
  const dir = path.join(baseDir, "models");
  let files = [];
  try {
    files = fs.readdirSync(dir).filter((f) => /\.gguf$/i.test(f)).sort();
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
    await session.prompt(prompt, { maxTokens: 2048, temperature: 0.7, signal, stopOnAbortSignal: true, onTextChunk: onChunk });
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
    const args = ["-m", file, "--host", "127.0.0.1", "--port", String(port), "-c", String(Number(process.env.MNX_CTX) || 4096)];
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
      max_tokens: 2048,
      temperature: 0.7,
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

function systemPrompt(userName) {
  return `You are Mnx, a friendly and helpful AI assistant running locally on the user's own device. Today's date is ${new Date().toISOString().slice(0, 10)}.${
    userName ? ` The user's name is ${userName}.` : ""
  }
Help with anything legal; politely decline only requests that would facilitate crimes or serious harm.

Before every answer, think briefly inside <think></think> tags: start with a short bold title like **Understanding the question**, then 1-3 sentences about what the user needs and how you'll answer. After </think>, write the final answer in markdown.
You can't browse the internet or check live data such as weather or news; if asked, say so and suggest switching to an online Mnx model.`;
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

export async function handleLocalChat({ body, messages, file, send, isClosed, signal }) {
  if (!file || !fs.existsSync(file)) {
    send({
      t: "error",
      text: "No local model found. Put your .gguf file in the models/ folder (or run npm run get-model) and restart Mnx.",
    });
    return;
  }

  // Convert the shared (Claude-format) history into plain chat turns.
  const history = [{ role: "system", content: systemPrompt(String(body.userName || "").slice(0, 60)) }];
  for (const m of messages.slice(0, -1)) {
    const text = blocksText(m.content);
    if (!text) continue;
    const prev = history[history.length - 1];
    const role = m.role === "user" ? "user" : "assistant";
    if (prev.role === role) prev.content += `\n\n${text}`;
    else if (role === "user" || prev.role === "user") history.push({ role, content: text });
  }
  // Drop the oldest turns if the history is too long for the context.
  while (history.length > 3 && JSON.stringify(history).length > 12000) history.splice(1, 2);
  const prompt = blocksText(messages[messages.length - 1].content);

  send({ t: "iteration", it: 0 });
  send({ t: "phase", phrases: ["Waking up your local model", "Loading Mnx into memory", "Warming up the neurons"] });

  await exclusive(async () => {
    if (isClosed()) return;
    let index = -1;
    let thinkingOpen = false;
    let textOpen = false;
    let answer = "";
    const openThinking = () => {
      if (!thinkingOpen) {
        index++;
        thinkingOpen = true;
        send({ t: "block_start", it: 0, i: index, block: { type: "thinking" } });
      }
    };
    const splitter = new ThinkSplitter(
      (t) => {
        if (t === null) {
          if (thinkingOpen) send({ t: "block_stop", it: 0, i: index });
          thinkingOpen = false;
          return;
        }
        openThinking();
        if (t) send({ t: "thinking", it: 0, i: index, text: t });
      },
      (t) => {
        t = t.replace(/<\/?think>/g, "");
        if (!textOpen && !t.trim()) return;
        if (!textOpen) {
          if (thinkingOpen) send({ t: "block_stop", it: 0, i: index });
          thinkingOpen = false;
          index++;
          textOpen = true;
          send({ t: "block_start", it: 0, i: index, block: { type: "text" } });
        }
        answer += t;
        send({ t: "text", it: 0, i: index, text: t });
      },
    );

    await generate(
      file,
      history,
      prompt,
      (chunk, kind) => {
        if (isClosed()) return;
        if (kind === "reasoning") {
          openThinking();
          send({ t: "thinking", it: 0, i: index, text: chunk });
        } else splitter.push(chunk);
      },
      signal,
    );
    splitter.end();
    if (isClosed()) return;
    if (thinkingOpen) send({ t: "block_stop", it: 0, i: index });
    if (textOpen) send({ t: "block_stop", it: 0, i: index });
    send({ t: "assistant_turn", content: [{ type: "text", text: answer.trim() || "…" }], model: LOCAL_ID });
    send({ t: "done" });
  });
}

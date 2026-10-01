// Runs a local GGUF model (e.g. Llama 3.2 1B Instruct) with llama.cpp via
// node-llama-cpp, and streams it to the browser using the same events as the
// Claude path. The model writes its reasoning inside <think>…</think> first;
// that streams into the thinking area, and the rest becomes the answer.
import fs from "node:fs";
import path from "node:path";

export const LOCAL_ID = "local";

export function localModelPath(baseDir) {
  return path.resolve(baseDir, process.env.MNX_LOCAL_MODEL || "models/llama-3.2-1b-instruct.Q4_K_M.gguf");
}

export function localModelLabel(file) {
  const name = path.basename(file).replace(/\.gguf$/i, "");
  const m = name.match(/llama-?(\d+(?:\.\d+)?)-?(\d+b)/i);
  return m ? `Local · Llama ${m[1]} ${m[2].toUpperCase()}` : `Local · ${name}`;
}

let loading = null;
async function load(file) {
  loading ||= (async () => {
    const { getLlama, LlamaChatSession } = await import("node-llama-cpp");
    const llama = await getLlama();
    const model = await llama.loadModel({ modelPath: file });
    const context = await model.createContext({ contextSize: { max: 8192 } });
    return { LlamaChatSession, context };
  })();
  try {
    return await loading;
  } catch (err) {
    loading = null;
    throw err;
  }
}

// One generation at a time: the context has a single sequence.
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
  return `You are Mnx, a friendly and helpful AI assistant running locally on the user's own computer. Today's date is ${new Date().toISOString().slice(0, 10)}.${
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
  if (!fs.existsSync(file)) {
    send({
      t: "error",
      text: `Local model not found at ${file}. Download your .gguf file into the models/ folder (see README) or set MNX_LOCAL_MODEL.`,
    });
    return;
  }

  // Convert the shared (Claude-format) history into plain chat turns.
  const history = [{ type: "system", text: systemPrompt(String(body.userName || "").slice(0, 60)) }];
  for (const m of messages.slice(0, -1)) {
    const text = blocksText(m.content);
    if (!text) continue;
    const prev = history[history.length - 1];
    if (m.role === "user") {
      if (prev.type === "user") prev.text += `\n\n${text}`;
      else history.push({ type: "user", text });
    } else if (prev.type === "model") prev.response[0] += `\n\n${text}`;
    else if (prev.type === "user") history.push({ type: "model", response: [text] });
  }
  const prompt = blocksText(messages[messages.length - 1].content);

  send({ t: "iteration", it: 0 });
  send({ t: "phase", phrases: ["Waking up your local model", "Loading Llama into memory", "Warming up the neurons"] });
  const { LlamaChatSession, context } = await load(file);

  await exclusive(async () => {
    if (isClosed()) return;
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence, autoDisposeSequence: false });
    // Drop the oldest turns if the history is too long for the context.
    while (history.length > 3 && JSON.stringify(history).length > 20000) history.splice(1, 2);
    session.setChatHistory(history);

    let index = -1;
    let thinkingOpen = false;
    let textOpen = false;
    let answer = "";
    const splitter = new ThinkSplitter(
      (t) => {
        if (t === null) {
          if (thinkingOpen) send({ t: "block_stop", it: 0, i: index });
          thinkingOpen = false;
          return;
        }
        if (!thinkingOpen) {
          index++;
          thinkingOpen = true;
          send({ t: "block_start", it: 0, i: index, block: { type: "thinking" } });
        }
        if (t) send({ t: "thinking", it: 0, i: index, text: t });
      },
      (t) => {
        if (!textOpen && !t.replace(/<\/?think>/g, "").trim()) return;
        if (!textOpen) {
          index++;
          textOpen = true;
          send({ t: "block_start", it: 0, i: index, block: { type: "text" } });
        }
        t = t.replace(/<\/?think>/g, "");
        if (!t) return;
        answer += t;
        send({ t: "text", it: 0, i: index, text: t });
      },
    );

    try {
      await session.prompt(prompt, {
        maxTokens: 2048,
        temperature: 0.7,
        signal,
        stopOnAbortSignal: true,
        onTextChunk: (chunk) => !isClosed() && splitter.push(chunk),
      });
      splitter.end();
    } finally {
      session.dispose({ disposeSequence: false });
      sequence.dispose();
    }
    if (isClosed()) return;
    if (textOpen) send({ t: "block_stop", it: 0, i: index });
    const content = [{ type: "text", text: answer.trim() || "…" }];
    send({ t: "assistant_turn", content, model: LOCAL_ID });
    send({ t: "done" });
  });
}

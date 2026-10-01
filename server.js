// Mnx server: serves the web UI and streams live Claude responses (thinking,
// tool activity and answer text) to the browser as Server-Sent Events.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { TOOLS, runTool } from "./tools.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;
const MAX_ITERATIONS = 16;

const MODELS = {
  "claude-opus-5-5": "Opus 5.5",
  "claude-sonnet-5-5": "Sonnet 5.5",
  "claude-fable-5-1": "Fable 5.1",
};
const EFFORTS = new Set(["low", "medium", "high", "xhigh", "max"]);

const client = new Anthropic();

// Browser bundles served from node_modules so the app works without a CDN.
const VENDOR = {
  "marked.js": "marked/lib/marked.umd.js",
  "purify.js": "dompurify/dist/purify.min.js",
  "highlight.js": "@highlightjs/cdn-assets/highlight.min.js",
  "hljs-dark.css": "@highlightjs/cdn-assets/styles/github-dark.min.css",
  "hljs-light.css": "@highlightjs/cdn-assets/styles/github.min.css",
  "jspdf.js": "jspdf/dist/jspdf.umd.min.js",
  "pptxgen.js": "pptxgenjs/dist/pptxgen.bundle.js",
  "docx.js": "docx/dist/index.iife.js",
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".json": "application/json",
};

function systemPrompt(userName) {
  const today = new Date().toISOString().slice(0, 10);
  return `You are Mnx, a powerful, friendly AI assistant. Today's date is ${today}.${
    userName ? ` The user's name is ${userName}.` : ""
  }

Be genuinely helpful: do what the user asks, completely and directly. Help with anything legal; decline only requests that would facilitate crimes or serious harm, and say so briefly.

Use your tools whenever they make the answer better:
- web_search / web_fetch for anything current, factual or that you are unsure about. Cite sources inline as markdown links.
- get_user_location when the user says "here", "near me", "today's weather" etc. without naming a place.
- get_weather for weather and forecasts. The UI renders a weather card from the result, so keep your prose short and add useful advice.
- find_places for restaurants, shops, attractions and other places. The UI shows place cards with map links.
- get_directions for routes between places. The UI shows a route card.
- create_file to deliver any code or text file (Python, HTML, JSON, Ruby, C, C++, Lua, Markdown, CSV, etc). Always put full, working file contents in the tool call instead of pasting large files into chat. HTML files can be previewed live.
- create_document for PDF reports, Word documents (docx) and slide decks (pptx). Write rich, well structured content.
- Tools from connected MCP servers (connectors) when relevant.

Format answers in GitHub-flavoured markdown. Keep answers focused; use headings, lists and tables when they help.`;
}

function sse(res, payload) {
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

function readJson(req, limit = 40 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error("Request too large"));
        req.destroy();
      } else chunks.push(c);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        reject(new Error("Invalid JSON"));
      }
    });
    req.on("error", reject);
  });
}

function sanitizeMcpServers(list) {
  if (!Array.isArray(list)) return [];
  const seen = new Set();
  const out = [];
  for (const s of list.slice(0, 20)) {
    if (!s || typeof s.url !== "string" || !/^https:\/\//i.test(s.url)) continue;
    let name = String(s.name || "connector")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "connector";
    while (seen.has(name)) name += "-x";
    seen.add(name);
    const server = { type: "url", url: s.url, name };
    if (typeof s.token === "string" && s.token.trim()) server.authorization_token = s.token.trim();
    out.push(server);
  }
  return out;
}

// Short, display-only summary of a block so the UI can label each step.
function describeBlock(block) {
  const d = { type: block.type };
  if (block.name) d.name = block.name;
  if (block.id) d.id = block.id;
  if (block.tool_use_id) d.tool_use_id = block.tool_use_id;
  if (block.server_name) d.server = block.server_name;
  if (block.input && Object.keys(block.input).length) d.input = block.input;
  if (block.type === "web_search_tool_result") {
    d.results = Array.isArray(block.content)
      ? block.content.slice(0, 8).map((r) => ({ title: r.title, url: r.url }))
      : [];
    if (!Array.isArray(block.content)) d.error = block.content?.error_code || "search failed";
  }
  if (block.type === "web_fetch_tool_result") {
    d.url = block.content?.url;
    d.title = block.content?.content?.title;
    if (block.content?.type?.includes("error")) d.error = block.content?.error_code || "fetch failed";
  }
  if (block.type === "mcp_tool_result") {
    d.error = !!block.is_error;
    const text = Array.isArray(block.content)
      ? block.content.filter((c) => c.type === "text").map((c) => c.text).join("\n")
      : "";
    d.preview = text.slice(0, 600);
  }
  return d;
}

async function handleChat(req, res) {
  let body;
  try {
    body = await readJson(req);
  } catch (e) {
    res.writeHead(400, { "content-type": "application/json" });
    return res.end(JSON.stringify({ error: e.message }));
  }

  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (!messages.length || messages[messages.length - 1].role !== "user") {
    res.writeHead(400, { "content-type": "application/json" });
    return res.end(JSON.stringify({ error: "messages must end with a user turn" }));
  }
  const model = MODELS[body.model] ? body.model : "claude-opus-5-5";
  const effort = EFFORTS.has(body.effort) ? body.effort : "medium";
  const ctx = {
    location: body.location && Number.isFinite(body.location.latitude) ? body.location : null,
  };
  const mcpServers = sanitizeMcpServers(body.mcpServers);

  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-transform",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });

  let closed = false;
  let current = null;
  res.on("close", () => {
    closed = true;
    current?.abort();
  });

  const tools = [
    { type: "web_search_20260209", name: "web_search", max_uses: 6 },
    { type: "web_fetch_20260209", name: "web_fetch", max_uses: 6 },
    ...TOOLS,
    ...mcpServers.map((s) => ({ type: "mcp_toolset", mcp_server_name: s.name })),
  ];
  const betas = ["server-side-fallback-2026-07-01"];
  if (mcpServers.length) betas.push("mcp-client-2025-11-20");

  const keepAlive = setInterval(() => !closed && res.write(": ping\n\n"), 15000);

  try {
    for (let it = 0; it < MAX_ITERATIONS && !closed; it++) {
      const params = {
        model,
        max_tokens: 64000,
        system: systemPrompt(String(body.userName || "").slice(0, 60)),
        thinking: { type: "adaptive", display: "summarized" },
        output_config: { effort },
        tools,
        messages,
        betas,
        fallbacks: "default",
      };
      if (mcpServers.length) params.mcp_servers = mcpServers;

      sse(res, { t: "iteration", it });
      const stream = client.beta.messages.stream(params);
      current = stream;

      let message;
      try {
        for await (const ev of stream) {
          if (closed) break;
          if (ev.type === "content_block_start") {
            sse(res, { t: "block_start", it, i: ev.index, block: describeBlock(ev.content_block) });
          } else if (ev.type === "content_block_delta") {
            const d = ev.delta;
            if (d.type === "thinking_delta") sse(res, { t: "thinking", it, i: ev.index, text: d.thinking });
            else if (d.type === "text_delta") sse(res, { t: "text", it, i: ev.index, text: d.text });
            else if (d.type === "input_json_delta") sse(res, { t: "input", it, i: ev.index, json: d.partial_json });
          } else if (ev.type === "content_block_stop") {
            sse(res, { t: "block_stop", it, i: ev.index });
          }
        }
        if (closed) break;
        message = await stream.finalMessage();
      } catch (err) {
        // With eager input streaming, an unparseable tool input rejects the
        // stream. Re-issue the turn a couple of times; rethrow API errors.
        if (err instanceof Anthropic.APIError || closed || it >= MAX_ITERATIONS - 1) throw err;
        sse(res, { t: "notice", text: "Retrying a garbled tool call…" });
        continue;
      }

      messages.push({ role: "assistant", content: message.content });
      sse(res, { t: "assistant_turn", content: message.content, model: message.model });

      if (message.stop_reason === "pause_turn") continue;
      if (message.stop_reason === "refusal") {
        sse(res, { t: "notice", level: "warn", text: "Mnx can't help with that request." });
        break;
      }
      const toolUses = message.content.filter((b) => b.type === "tool_use");
      if (message.stop_reason === "max_tokens") {
        sse(res, { t: "notice", level: "warn", text: "The response hit the length limit." });
        if (toolUses.length) {
          // A truncated tool call must not run; answer it with errors so the
          // history stays valid for the next turn.
          const results = toolUses.map((tu) => ({
            type: "tool_result",
            tool_use_id: tu.id,
            is_error: true,
            content: "Tool input was truncated by the output limit; not executed.",
          }));
          messages.push({ role: "user", content: results });
          sse(res, { t: "user_turn", content: results });
        }
        break;
      }
      if (message.stop_reason !== "tool_use" || !toolUses.length) break;

      const outcomes = await Promise.all(
        toolUses.map(async (tu) => {
          sse(res, { t: "tool_run", id: tu.id, name: tu.name, input: tu.input });
          const out = await runTool(tu.name, tu.input, ctx);
          sse(res, { t: "tool_done", id: tu.id, name: tu.name, error: !!out.error, display: out.display });
          return {
            type: "tool_result",
            tool_use_id: tu.id,
            content: JSON.stringify(out.error ? { error: out.error } : out.result),
            ...(out.error ? { is_error: true } : {}),
          };
        }),
      );
      messages.push({ role: "user", content: outcomes });
      sse(res, { t: "user_turn", content: outcomes });
    }
    if (!closed) sse(res, { t: "done" });
  } catch (err) {
    console.error("[chat]", err?.status || "", err?.message || err);
    if (!closed) {
      let text = err?.message || "Something went wrong.";
      if (err instanceof Anthropic.AuthenticationError)
        text = "The server has no valid Anthropic API key. Set ANTHROPIC_API_KEY and restart Mnx.";
      else if (err instanceof Anthropic.RateLimitError) text = "Rate limited — please wait a moment and try again.";
      else if (err instanceof Anthropic.APIConnectionError) text = "Couldn't reach the AI service. Check the server's network.";
      sse(res, { t: "error", text });
    }
  } finally {
    clearInterval(keepAlive);
    if (!closed) res.end();
  }
}

function serveFile(res, file) {
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404, { "content-type": "text/plain" });
      return res.end("Not found");
    }
    res.writeHead(200, {
      "content-type": MIME[path.extname(file)] || "application/octet-stream",
      "cache-control": "no-cache",
    });
    res.end(data);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (req.method === "POST" && url.pathname === "/api/chat") return handleChat(req, res);
  if (req.method === "GET" && url.pathname === "/api/config") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ models: MODELS, hasKey: !!(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN) }));
  }
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405);
    return res.end();
  }
  if (url.pathname.startsWith("/vendor/")) {
    const rel = VENDOR[url.pathname.slice("/vendor/".length)];
    if (!rel) {
      res.writeHead(404);
      return res.end();
    }
    return serveFile(res, path.join(here, "node_modules", rel));
  }
  const publicDir = path.join(here, "public");
  const file = path.normalize(path.join(publicDir, url.pathname === "/" ? "index.html" : url.pathname));
  if (!file.startsWith(publicDir + path.sep)) {
    res.writeHead(403);
    return res.end();
  }
  serveFile(res, file);
});

server.listen(PORT, () => console.log(`Mnx is live at http://localhost:${PORT}`));

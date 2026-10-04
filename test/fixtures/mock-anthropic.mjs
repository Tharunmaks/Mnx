// Minimal mock of the Anthropic Messages streaming API for tests.
// Picks a scenario from the first user message, records every request, and
// rejects histories the real API would reject (unanswered tool_use blocks).
import http from "node:http";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chunks = (s, n) => s.match(new RegExp(`[\\s\\S]{1,${n}}`, "g"));
const textOf = (content) => (typeof content === "string" ? content : content.map((b) => b.text || "").join(" "));

function validate(messages) {
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
    const ids = m.content.filter((b) => b.type === "tool_use").map((b) => b.id);
    if (!ids.length) continue;
    const next = messages[i + 1];
    const answered = new Set((Array.isArray(next?.content) ? next.content : []).filter((b) => b.type === "tool_result").map((b) => b.tool_use_id));
    for (const id of ids) if (!answered.has(id)) return `tool_use ${id} has no matching tool_result`;
  }
  return null;
}

export function startMockAnthropic() {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    let raw = "";
    for await (const c of req) raw += c;
    const p = JSON.parse(raw);
    requests.push({ url: req.url, headers: req.headers, body: p });
    const problem = validate(p.messages);
    if (problem) {
      res.writeHead(400, { "content-type": "application/json" });
      return res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: problem } }));
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    const send = async (type, data) => {
      res.write(`event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`);
      await sleep(2);
    };
    const first = textOf(p.messages.find((m) => m.role === "user").content).toLowerCase();
    const last = p.messages[p.messages.length - 1];
    const afterTool = Array.isArray(last.content) && last.content.some((b) => b.type === "tool_result");
    await send("message_start", {
      message: { id: `msg_${requests.length}`, type: "message", role: "assistant", model: p.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 1, output_tokens: 0 } },
    });
    let idx = 0;
    const thinking = async (t) => {
      await send("content_block_start", { index: idx, content_block: { type: "thinking", thinking: "", signature: "" } });
      for (const c of chunks(t, 8)) await send("content_block_delta", { index: idx, delta: { type: "thinking_delta", thinking: c } });
      await send("content_block_delta", { index: idx, delta: { type: "signature_delta", signature: "sig" } });
      await send("content_block_stop", { index: idx++ });
    };
    const text = async (t) => {
      await send("content_block_start", { index: idx, content_block: { type: "text", text: "" } });
      for (const c of chunks(t, 8)) await send("content_block_delta", { index: idx, delta: { type: "text_delta", text: c } });
      await send("content_block_stop", { index: idx++ });
    };
    const tool = async (name, input, server = false) => {
      const id = `${server ? "srvtoolu" : "toolu"}_${requests.length}_${idx}`;
      await send("content_block_start", { index: idx, content_block: { type: server ? "server_tool_use" : "tool_use", id, name, input: {} } });
      for (const c of chunks(JSON.stringify(input), 12)) await send("content_block_delta", { index: idx, delta: { type: "input_json_delta", partial_json: c } });
      await send("content_block_stop", { index: idx++ });
      return id;
    };
    const finish = async (stop) => {
      await send("message_delta", { delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: 10 } });
      await send("message_stop", {});
      res.end();
    };

    if (afterTool) {
      const results = last.content.filter((b) => b.type === "tool_result").map((b) => b.content).join(" | ");
      await thinking("**Reviewing the tool output**\nSummarize it.");
      await text(`Tool said: ${results.slice(0, 200)}`);
      return finish("end_turn");
    }
    if (first.includes("deck")) {
      await thinking("**Planning the slides**\nTwo slides.");
      await tool("create_document", { format: "pptx", title: "Mock Deck", slides: [{ title: "A", bullets: ["x"] }] });
      return finish("tool_use");
    }
    if (first.includes("run some code")) {
      await tool("run_code", { language: "python", code: "print(sum(range(10)))" });
      return finish("tool_use");
    }
    if (first.includes("search")) {
      await thinking("**Searching**\nLook it up.");
      const sid = await tool("web_search", { query: "mnx test" }, true);
      await send("content_block_start", {
        index: idx,
        content_block: { type: "web_search_tool_result", tool_use_id: sid, content: [{ type: "web_search_result", title: "Result", url: "https://example.com/r", encrypted_content: "x" }] },
      });
      await send("content_block_stop", { index: idx++ });
      await text("Found it ([source](https://example.com/r)).");
      return finish("end_turn");
    }
    if (first.includes("refuse")) {
      return finish("refusal");
    }
    await thinking("**Understanding the request**\nSay hello.");
    await text("Hello from the **mock** model!");
    return finish("end_turn");
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port, requests })));
}

// Turning online-model ("teacher") chats into local-model training data.
import { test } from "node:test";
import assert from "node:assert/strict";
import { teacherToLocal, pyJson } from "../learn.js";
import { systemPrompt, LOCAL_TOOL_NAMES } from "../local.js";

const sys = systemPrompt("Tharun");
test("a tool-using chat becomes local format (think, tool_call, tool_response, answer)", () => {
  const messages = [
    { role: "user", content: "hi" },
    { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
    { role: "user", content: [{ type: "text", text: "What's 15% of 240?" }] },
    { role: "assistant", content: [{ type: "thinking", thinking: "**Calculating** The user wants a percentage. I'll use the calculator." }, { type: "tool_use", id: "t1", name: "calculate", input: { expression: "15% of 240" } }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: '{"expression":"15% of 240","result":36}' }] },
    { role: "assistant", content: [{ type: "thinking", thinking: "Got it." }, { type: "text", text: "**36**" }] },
  ];
  const t = teacherToLocal(messages, sys, LOCAL_TOOL_NAMES);
  assert.ok(t);
  assert.deepEqual(t.messages.map((m) => m.role), ["system", "user", "assistant", "user", "assistant", "user", "assistant"]);
  assert.equal(t.messages[t.train_from].content, "What's 15% of 240?");
  assert.equal(t.messages[4].content, `<think>**Calculating**\nThe user wants a percentage.</think>\n<tool_call>\n${pyJson({ name: "calculate", arguments: { expression: "15% of 240" } })}\n</tool_call>`);
  assert.equal(t.messages[5].content, '<tool_response>\n{"expression": "15% of 240", "result": 36}\n</tool_response>');
  assert.equal(t.messages[6].content, "<think>**Answering**\nGot it.</think>\n**36**");
});

test("chats using tools the local model doesn't have are skipped", () => {
  const web = [
    { role: "user", content: "news?" },
    { role: "assistant", content: [{ type: "server_tool_use", id: "s1", name: "web_search", input: { query: "news" } }, { type: "web_search_tool_result", tool_use_id: "s1", content: [] }, { type: "text", text: "Here's the news" }] },
  ];
  assert.equal(teacherToLocal(web, sys, LOCAL_TOOL_NAMES), null);
  const mcp = [{ role: "user", content: "x" }, { role: "assistant", content: [{ type: "tool_use", id: "a", name: "some_connector_tool", input: {} }] }, { role: "user", content: [{ type: "tool_result", tool_use_id: "a", content: "{}" }] }, { role: "assistant", content: [{ type: "text", text: "ok" }] }];
  assert.equal(teacherToLocal(mcp, sys, LOCAL_TOOL_NAMES), null);
});

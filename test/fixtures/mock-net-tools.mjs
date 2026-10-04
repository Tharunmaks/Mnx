// Test-only: replaces Mnx's internet tools (weather, search, places,
// directions, page reading, location) with the stored results from
// training/data/test.jsonl, so screenshots and demos work without internet
// and match the stand-in model's answers. Offline tools (images, documents,
// files, the code runner) run for real. Loaded via use-mock-net-tools.mjs.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as real from "../../tools.js?real";
import { extraDisplay } from "../../tools-extra.js";

export const { TOOLS, CODE_RUNNER, htmlToText, parseDuckDuckGo, runSnippet } = real;
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "../..");
const NET = new Set(["get_weather", "search_web", "read_webpage", "find_places", "get_directions", "get_user_location", "convert_currency", "get_time", "wikipedia", "define_word", "translate",
  "public_holidays", "country_info", "crypto_price", "book_search", "find_words", "air_quality", "sunrise_sunset", "recent_earthquakes", "random_joke", "quote", "github_repo", "package_info"]);
const key = (name, args) => `${name}:${JSON.stringify(args, Object.keys(args || {}).sort())}`;
// The same call (e.g. weather in Lisbon) appears in several cases with
// different results, so results are stored per conversation as well.
const text = (c) => (typeof c === "string" ? c : (c || []).map((b) => b.text || "").join(""));
const convo = (messages) => messages.filter((m) => m.role === "user").map((m) => text(m.content)).filter(Boolean).join("\n");
const stored = new Map();
for (const line of fs.readFileSync(process.env.MOCK_TOOLS_FROM || path.join(root, "training/data/test.jsonl"), "utf8").trim().split("\n")) {
  const { reference: ref, messages } = JSON.parse(line);
  const c = convo(messages);
  for (let i = 0; i < ref.length - 1; i++) {
    const m = ref[i].content.match(/<tool_call>\n([\s\S]*?)\n<\/tool_call>/);
    if (!m) continue;
    const call = JSON.parse(m[1]);
    const res = JSON.parse(ref[i + 1].content.slice("<tool_response>\n".length, -"\n</tool_response>".length));
    stored.set(`${c}\u0000${key(call.name, call.arguments)}`, res);
    stored.set(key(call.name, call.arguments), res);
  }
}

function display(name, r) {
  if (name === "get_weather") {
    const n = (s) => parseInt(String(s), 10);
    return { kind: "weather", location: r.location, unit: "°C", condition: r.current.condition, icon: r.daily[0].icon, temperature: n(r.current.temperature),
             feels_like: n(r.current.feels_like), humidity: n(r.current.humidity), wind: r.current.wind, sunrise: r.sunrise, sunset: r.sunset, daily: r.daily };
  }
  if (name === "find_places") return { kind: "places", query: r.query, near: r.near, places: r.places };
  if (name === "get_directions") return { kind: "directions", ...r };
  if (name === "search_web") return { kind: "sources", query: r.query, results: r.results };
  if (name === "read_webpage") return { kind: "sources", query: null, results: [{ title: r.title, url: r.url }] };
  if (extraDisplay(name, r)) return extraDisplay(name, r);
  // Online quick tools: a simple card from the stored result.
  if (!["get_weather", "find_places", "get_directions", "search_web", "read_webpage", "get_user_location"].includes(name)) {
    const entries = Object.entries(r).filter(([, v]) => v !== null && typeof v !== "object");
    const lists = Object.entries(r).filter(([, v]) => Array.isArray(v));
    const rows = lists.length ? lists[0][1].slice(0, 8).map((x) => (typeof x === "object" ? [Object.values(x)[0], Object.values(x).slice(1).join(" · ")] : ["•", String(x)])) : entries.slice(1).map(([k, v]) => [k.replace(/_/g, " "), String(v)]);
    return { kind: "info", icon: "globe", title: String(entries[0]?.[1] ?? name), subtitle: name.replace(/_/g, " "), rows };
  }
  if (name === "get_user_location") return r.available === false ? { kind: "location", available: false } : { kind: "location", available: true, ...r };
  return undefined;
}

export async function runTool(name, input, ctx) {
  if (NET.has(name)) {
    const r = stored.get(`${convo(ctx?.messages || [])}\u0000${key(name, input)}`) ?? stored.get(key(name, input));
    if (!r) return { error: "No stored result for this request (offline test mode)." };
    if (r.error) return { error: r.error };
    return { result: r, display: display(name, r) };
  }
  return real.runTool(name, input, ctx);
}

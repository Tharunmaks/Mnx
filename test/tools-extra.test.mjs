// The everyday tools: offline math/units, and the online ones against
// recorded API responses (no network needed).
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { calculate, convertUnits, extraHandlers as h } from "../tools-extra.js";
import { normalizeToolCall } from "../local.js";

describe("calculate", () => {
  const cases = [["2+3*4", 14], ["(2+3)*4", 20], ["2^3^2", 512], ["-3^2", -9], ["2^-1", 0.5], ["15% of 240", 36], ["200 + 10%", 220], ["200 - 25%", 150],
    ["sqrt(144)", 12], ["5!", 120], ["2pi", 2 * Math.PI], ["3(4+1)", 15], ["1,234,567 + 1", 1234568], ["10 mod 3", 1], ["max(3, 7, 2)", 7], ["3 x 4", 12],
    ["12 × 3 ÷ 4", 9], ["log(1000)", 3], ["ln(e)", 1], ["round(3.14159, 2)", 3.14], ["1e3/4", 250]];
  for (const [e, want] of cases) test(e, () => assert.ok(Math.abs(calculate(e) - want) < 1e-9, `${e} = ${calculate(e)}`));
  test("sin uses degrees", () => assert.ok(Math.abs(calculate("sin(30)") - 0.5) < 1e-12));
  for (const bad of ["7/0", "2 +", "abc", "process.exit()", "constructor"]) test(`rejects ${bad}`, () => assert.throws(() => calculate(bad)));
});

describe("convert_units", () => {
  const cases = [[5, "km", "miles", 3.10685596119], [100, "F", "C", 37.7777777778], [0, "celsius", "kelvin", 273.15], [1, "kg", "lbs", 2.20462262185],
    [60, "mph", "km/h", 96.56064], [3, "feet", "inches", 36], [2, "cups", "ml", 473.176473], [1, "acre", "square meters", 4046.8564224], [100, "degrees fahrenheit", "degrees celsius", 37.7777777778]];
  for (const [v, a, b, w] of cases) test(`${v} ${a} → ${b}`, () => assert.ok(Math.abs(convertUnits(v, a, b) - w) < 1e-6));
  test("refuses mismatched units", () => assert.throws(() => convertUnits(1, "kg", "km"), /Can't convert/));
  test("refuses unknown units", () => assert.throws(() => convertUnits(1, "parsec-ish", "km"), /Unknown unit/));
});

describe("online tools (recorded responses)", () => {
  const realFetch = globalThis.fetch;
  const seen = [];
  const fixtures = [
    [/open\.er-api\.com\/v6\/latest\/USD/, { result: "success", time_last_update_utc: "Sun, 05 Oct 2025 00:02:31 +0000", rates: { USD: 1, INR: 88.7 } }],
    [/geocoding-api\.open-meteo\.com/, { results: [{ name: "Tokyo", admin1: "Tokyo", country: "Japan", timezone: "Asia/Tokyo" }] }],
    [/w\/api\.php/, { query: { search: [{ title: "Eiffel Tower" }] } }],
    [/rest_v1\/page\/summary\/Eiffel_Tower/, { title: "Eiffel Tower", description: "Tower in Paris", extract: "The Eiffel Tower is a wrought-iron lattice tower in Paris.", content_urls: { desktop: { page: "https://en.wikipedia.org/wiki/Eiffel_Tower" } }, thumbnail: { source: "https://upload.wikimedia.org/e.jpg" } }],
    [/dictionaryapi\.dev\/api\/v2\/entries\/en\/serendipity/, [{ word: "serendipity", phonetic: "/x/", meanings: [{ partOfSpeech: "noun", definitions: [{ definition: "Luck in finding good things.", example: "Pure serendipity." }], synonyms: ["luck"] }] }]],
    [/dictionaryapi\.dev/, null, 404],
    [/mymemory\.translated\.net/, { responseStatus: 200, responseData: { translatedText: "Bonjour" } }],
  ];
  before(() => {
    globalThis.fetch = async (url) => {
      seen.push(String(url));
      const hit = fixtures.find(([re]) => re.test(String(url)));
      if (!hit) throw new Error(`unexpected request ${url}`);
      const [, body, status = 200] = hit;
      return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
    };
  });
  after(() => (globalThis.fetch = realFetch));

  test("convert_currency", async () => {
    const r = await h.convert_currency({ amount: 100, from: "usd", to: "inr" });
    assert.equal(r.result.converted, 8870);
    assert.equal(r.display.title, "8,870 INR");
    await assert.rejects(h.convert_currency({ amount: 1, from: "dollars", to: "INR" }), /3-letter/);
  });
  test("get_time", async () => {
    const r = await h.get_time({ location: "Tokyo" });
    assert.equal(r.result.timezone, "Asia/Tokyo");
    assert.equal(r.result.utc_offset, "UTC+09:00");
    assert.match(r.result.time, /^\d\d:\d\d$/);
  });
  test("wikipedia", async () => {
    const r = await h.wikipedia({ topic: "eiffel tower" });
    assert.equal(r.result.title, "Eiffel Tower");
    assert.equal(r.display.image, "https://upload.wikimedia.org/e.jpg");
  });
  test("define_word, and a word that isn't in the dictionary", async () => {
    const r = await h.define_word({ word: "Serendipity" });
    assert.equal(r.result.meanings[0].part_of_speech, "noun");
    await assert.rejects(h.define_word({ word: "qwzx" }), /No dictionary entry/);
  });
  test("translate takes language names", async () => {
    const r = await h.translate({ text: "Good morning", to: "French" });
    assert.equal(r.result.translation, "Bonjour");
    assert.ok(seen.some((u) => u.includes("langpair=en%7Cfr")));
  });
  test("create_qr_code", async () => {
    const r = await h.create_qr_code({ text: "https://example.com" });
    assert.equal(r.display.kind, "image");
    assert.match(r.display.url, /api\.qrserver\.com.*data=https%3A%2F%2Fexample\.com/);
  });
});

test("small-model near-misses map to the new tools", () => {
  assert.deepEqual(normalizeToolCall("calculator", { expr: "2+2" }), { name: "calculate", input: { expression: "2+2" } });
  assert.deepEqual(normalizeToolCall("translate_text", { text: "hi", target_language: "German" }), { name: "translate", input: { text: "hi", to: "German" } });
  assert.deepEqual(normalizeToolCall("currency", { amount: 5, from_currency: "USD", to_currency: "EUR" }), { name: "convert_currency", input: { amount: 5, from: "USD", to: "EUR" } });
  assert.deepEqual(normalizeToolCall("wiki", { query: "Mars" }), { name: "wikipedia", input: { topic: "Mars" } });
});

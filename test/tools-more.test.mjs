// The 65 quick tools: exact answers for the offline ones, recorded responses
// for the online ones, memory on disk, and the Claude/local registrations.
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-more-"));
process.env.MNX_MEMORY_FILE = path.join(tmp, "memory.json");
const { MORE_TOOLS, moreHandlers: h, numberToWords, toRoman, fromRoman, memoryPrompt } = await import("../tools-more.js");
const { TOOLS } = await import("../tools.js");
const { LOCAL_TOOL_NAMES, STABLE_PROMPT, systemPrompt, normalizeToolCall } = await import("../local.js");
const r = async (name, input) => (await h[name](input, {})).result;

test("65 quick tools, all registered for online and local models", () => {
  assert.equal(MORE_TOOLS.length, 65);
  for (const t of MORE_TOOLS) {
    assert.ok(TOOLS.some((x) => x.name === t.name), `${t.name} missing for online models`);
    assert.ok(LOCAL_TOOL_NAMES.includes(t.name), `${t.name} missing for the local model`);
    assert.ok(STABLE_PROMPT.includes(t.sig), `${t.name} missing from the local prompt`);
  }
  assert.equal(new Set(TOOLS.map((t) => t.name)).size, TOOLS.length, "duplicate tool names");
});

describe("offline tools give exact answers", () => {
  test("money", async () => {
    assert.deepEqual(await r("tip_split", { bill: 675, tip_percent: 20, people: 7 }), { bill: 675, tip_percent: 20, tip: 135, total: 810, people: 7, each: 115.71 });
    assert.equal((await r("loan_emi", { principal: 500000, annual_rate: 9, months: 60 })).emi, 10379.18);
    assert.equal((await r("discount", { price: 2499, percent: 20 })).final_price, 1999.2);
    assert.deepEqual(await r("tax", { amount: 1180, rate: 18, inclusive: true }), { net: 1000, tax: 180, gross: 1180, rate: 18, inclusive: true });
    assert.equal((await r("percentage_change", { from: 80, to: 100 })).percent, 25);
  });
  test("numbers", async () => {
    assert.equal(numberToWords(1234567), "one million two hundred and thirty-four thousand five hundred and sixty-seven");
    assert.equal(numberToWords(1234567, "indian"), "twelve lakh thirty-four thousand five hundred and sixty-seven");
    assert.equal(toRoman(1994), "MCMXCIV");
    assert.equal(fromRoman("MMXXIV"), 2024);
    assert.throws(() => fromRoman("IIII"));
    assert.deepEqual((await r("prime_factors", { n: 360 })).factors, [2, 2, 2, 3, 3, 5]);
    assert.equal((await r("prime_factors", { n: 7919 })).is_prime, true);
    assert.deepEqual(await r("gcd_lcm", { numbers: [12, 18, 30] }), { numbers: [12, 18, 30], gcd: 6, lcm: 180 });
    assert.equal((await r("number_base", { value: "0xff" })).decimal, "255");
    const st = await r("statistics", { numbers: "2, 4, 4, 4, 5, 5, 7, 9" });
    assert.deepEqual([st.mean, st.median, st.mode[0]], [5, 4.5, 4]);
  });
  test("dates", async () => {
    assert.equal((await r("date_diff", { from: "2025-01-01", to: "2025-12-25" })).total_days, 358);
    assert.equal((await r("add_to_date", { date: "2025-01-31", months: 1 })).date, "2025-02-28");
    assert.equal((await r("day_of_week", { date: "1947-08-15" })).weekday, "Friday");
    assert.equal((await r("unix_time", { value: "1700000000" })).iso_utc, "2023-11-14T22:13:20.000Z");
    assert.match((await r("timezone_convert", { time: "9:30 pm", from_zone: "IST", to_zone: "UTC" })).time, /^16:00/);
  });
  test("health, text and developer helpers", async () => {
    assert.deepEqual(await r("bmi", { weight_kg: 70, height_cm: 175 }), { bmi: 22.9, category: "healthy weight", healthy_weight_range_kg: [56.7, 76.3] });
    assert.equal((await r("bmr", { weight_kg: 70, height_cm: 175, age: 30, sex: "male" })).bmr, 1649);
    assert.equal((await r("change_case", { text: "hello big world", case: "snake" })).text, "hello_big_world");
    assert.equal((await r("slugify", { text: "Héllo Wörld: 2025!" })).slug, "hello-world-2025");
    assert.equal((await r("hash_text", { text: "abc" })).hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert.equal((await r("base64", { text: "aGkgdGhlcmU=", mode: "decode" })).output, "hi there");
    assert.equal((await r("json_format", { json: '{"a":1,}' })).valid, false);
    assert.equal((await r("regex_test", { pattern: "\\d+", text: "a1 b22 c333" })).count, 3);
    assert.deepEqual((await r("text_diff", { a: "a\nb\nc", b: "a\nc\nd" })).changes, ["- b", "+ d"]);
    assert.equal((await r("color_convert", { color: "coral" })).hex, "#ff7f50");
    assert.equal((await r("cron_explain", { expression: "0 9 * * 1-5" })).meaning, "at 9:00, on Monday to Friday");
    assert.equal((await r("ip_subnet", { cidr: "192.168.1.77/26" })).usable_hosts, 62);
    assert.equal((await r("password_strength", { password: "password123" })).rating, "very weak");
    const pw = await r("generate_password", { length: 20 });
    assert.equal(pw.password.length, 20);
    assert.match(pw.password, /[A-Z]/);
  });
  test("bad input gives a clear error, not a crash", async () => {
    await assert.rejects(h.bmi({ weight_kg: 70, height_cm: 0 }), /Check the weight/);
    await assert.rejects(h.roll_dice({ dice: "lots" }), /2d6/);
    await assert.rejects(h.timezone_convert({ time: "10:00", from_zone: "Mars/Base", to_zone: "UTC" }), /Unknown time zone/);
  });
});

test("memory: remember, appear in the instructions, recall, forget", async () => {
  await h.remember({ fact: "My favourite color is teal" });
  assert.match(memoryPrompt(), /- My favourite color is teal/);
  assert.match(systemPrompt("Tharun"), /Things the user asked you to remember:\n- My favourite color is teal$/);
  assert.ok(systemPrompt("").startsWith(STABLE_PROMPT), "memories go after the cached part of the prompt");
  assert.deepEqual((await r("recall", {})).memories, ["My favourite color is teal"]);
  assert.equal((await r("forget", { fact: "teal" })).removed, 1);
  assert.equal(memoryPrompt(), "");
});

describe("online quick tools (recorded responses)", () => {
  const real = globalThis.fetch;
  const fixtures = [
    [/date\.nager\.at\/api\/v3\/PublicHolidays\/2030\/US/, [{ date: "2030-01-01", localName: "New Year's Day", name: "New Year's Day" }, { date: "2030-07-04", localName: "Independence Day", name: "Independence Day" }]],
    [/restcountries\.com\/v3\.1\/name\/japan/i, [{ name: { common: "Japan", official: "Japan" }, capital: ["Tokyo"], population: 123210000, languages: { jpn: "Japanese" }, currencies: { JPY: { name: "Japanese yen", symbol: "¥" } }, region: "Asia", subregion: "Eastern Asia", idd: { root: "+8", suffixes: ["1"] }, flag: "🇯🇵", area: 377930 }]],
    [/coingecko\.com.*ids=bitcoin/, { bitcoin: { usd: 65000, usd_24h_change: -1.234 } }],
    [/api\.datamuse\.com\/words\?rel_rhy=cat/, [{ word: "hat" }, { word: "bat" }]],
    [/icanhazdadjoke\.com/, { joke: "I'm reading a book about anti-gravity. It's impossible to put down." }],
  ];
  before(() => {
    globalThis.fetch = async (url) => {
      const hit = fixtures.find(([re]) => re.test(String(url)));
      if (!hit) throw new Error(`unexpected request ${url}`);
      return new Response(JSON.stringify(hit[1]), { status: 200, headers: { "content-type": "application/json" } });
    };
  });
  after(() => (globalThis.fetch = real));
  test("public_holidays", async () => assert.equal((await r("public_holidays", { country_code: "us", year: 2030 })).holidays.length, 2));
  test("country_info", async () => {
    const c = await r("country_info", { country: "Japan" });
    assert.deepEqual([c.capital, c.calling_code, c.currencies[0]], ["Tokyo", "+81", "Japanese yen (JPY, ¥)"]);
  });
  test("crypto_price takes symbols", async () => assert.deepEqual(await r("crypto_price", { coin: "BTC" }), { coin: "bitcoin", currency: "USD", price: 65000, change_24h_percent: -1.23 }));
  test("find_words", async () => assert.deepEqual((await r("find_words", { word: "cat", kind: "rhymes" })).words, ["hat", "bat"]));
  test("random_joke", async () => assert.match((await r("random_joke", {})).joke, /anti-gravity/));
});

test("small-model near-misses map to quick tools", () => {
  assert.equal(normalizeToolCall("bmi_calculator", {}).name, "bmi");
  assert.equal(normalizeToolCall("timer", {}).name, "set_timer");
  assert.equal(normalizeToolCall("joke", {}).name, "random_joke");
});

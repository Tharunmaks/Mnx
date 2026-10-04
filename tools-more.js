// "Quick tools": 56 small, dependable helpers. Most run offline on the
// device; the online ones use free services that need no API key.
//
// Each tool: { name, sig, description, params, required, run(input, ctx) }
//   sig  — compact signature shown to the local model ("bmi(weight_kg, height_cm)")
//   run  — returns { result, display }; display is a chat card ({ kind: "info", … })
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const UA = "Mnx/1.0 (personal assistant)";
async function getJson(url, headers = {}) {
  const res = await fetch(url, { headers: { "user-agent": UA, accept: "application/json", ...headers }, signal: AbortSignal.timeout(12000) });
  if (!res.ok) throw new Error(`${new URL(url).host} returned HTTP ${res.status}`);
  return res.json();
}

/* ───────────── helpers ───────────── */
const S = { type: "string" };
const N = { type: "number" };
const I = { type: "integer" };
const B = { type: "boolean" };
const A = (items = S) => ({ type: "array", items });
const str = (v, max = 5000) => (v === undefined || v === null ? "" : String(v)).slice(0, max);
const need = (v, name) => {
  if (v === undefined || v === null || v === "") throw new Error(`'${name}' is required`);
  return v;
};
const num = (v, name) => {
  const n = typeof v === "number" ? v : Number(String(v ?? "").replace(/[,\s]/g, ""));
  if (!Number.isFinite(n)) throw new Error(`'${name}' must be a number`);
  return n;
};
const r2 = (x, d = 2) => Math.round(x * 10 ** d) / 10 ** d;
const fmt = (x, d = 2) => Number(r2(x, d)).toLocaleString("en-US", { maximumFractionDigits: d });
const info = (icon, title, subtitle, rows = [], extra = {}) => ({ kind: "info", icon, title: String(title), subtitle, rows: rows.filter((r) => r && r[1] !== undefined && r[1] !== ""), ...extra });
const listNums = (v, name) => {
  const arr = Array.isArray(v) ? v : String(v ?? "").split(/[\s,;]+/).filter(Boolean);
  const out = arr.map((x) => num(x, name));
  if (!out.length) throw new Error(`'${name}' needs at least one number`);
  return out;
};

function parseDate(v, name = "date") {
  const s = str(v).trim().toLowerCase();
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  if (!s || s === "today" || s === "now") return today;
  if (s === "tomorrow") return new Date(today.getTime() + 864e5);
  if (s === "yesterday") return new Date(today.getTime() - 864e5);
  const iso = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (iso) return new Date(Number(iso[1]), Number(iso[2]) - 1, Number(iso[3]));
  const dmy = s.match(/^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$/); // 25/12/2025 (day first)
  if (dmy) return new Date(Number(dmy[3]), Number(dmy[2]) - 1, Number(dmy[1]));
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) throw new Error(`Couldn't read '${name}' as a date (use YYYY-MM-DD)`);
  d.setHours(0, 0, 0, 0);
  return d;
}
const isoDate = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const longDate = (d) => d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const dayMs = 864e5;
const daysBetween = (a, b) => Math.round((Date.UTC(b.getFullYear(), b.getMonth(), b.getDate()) - Date.UTC(a.getFullYear(), a.getMonth(), a.getDate())) / dayMs);

function ymdDiff(a, b) {
  let [from, to] = a <= b ? [a, b] : [b, a];
  let y = to.getFullYear() - from.getFullYear();
  let m = to.getMonth() - from.getMonth();
  let d = to.getDate() - from.getDate();
  if (d < 0) {
    m -= 1;
    d += new Date(to.getFullYear(), to.getMonth(), 0).getDate();
  }
  if (m < 0) {
    y -= 1;
    m += 12;
  }
  return { years: y, months: m, days: d };
}

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
function words999(n) {
  const out = [];
  if (n >= 100) {
    out.push(`${ONES[Math.floor(n / 100)]} hundred`);
    n %= 100;
    if (n) out.push("and");
  }
  if (n >= 20) out.push(TENS[Math.floor(n / 10)] + (n % 10 ? `-${ONES[n % 10]}` : ""));
  else if (n || !out.length) out.push(ONES[n]);
  return out.join(" ");
}
export function numberToWords(value, system = "international") {
  let n = Math.trunc(Math.abs(value));
  if (n > 999_999_999_999_999) throw new Error("Number too large");
  const neg = value < 0 ? "minus " : "";
  if (n === 0) return "zero";
  const parts = [];
  if (system === "indian") {
    const units = [[1e7, "crore"], [1e5, "lakh"], [1e3, "thousand"]];
    for (const [size, name] of units) {
      if (n >= size) {
        const q = Math.floor(n / size);
        parts.push(`${size === 1e7 && q >= 1000 ? numberToWords(q, "indian") : words999(q)} ${name}`);
        n %= size;
      }
    }
  } else {
    for (const [size, name] of [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"], [1e3, "thousand"]]) {
      if (n >= size) {
        parts.push(`${words999(Math.floor(n / size))} ${name}`);
        n %= size;
      }
    }
  }
  if (n) parts.push((parts.length && n < 100 ? "and " : "") + words999(n));
  return neg + parts.join(" ");
}

const ROMAN = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
export function toRoman(n) {
  if (!Number.isInteger(n) || n < 1 || n > 3999) throw new Error("Roman numerals cover 1–3999");
  let out = "";
  for (const [v, s] of ROMAN) while (n >= v) (out += s), (n -= v);
  return out;
}
export function fromRoman(s) {
  const map = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
  const t = String(s).toUpperCase().trim();
  if (!/^[IVXLCDM]+$/.test(t)) throw new Error(`"${s}" isn't a Roman numeral`);
  let total = 0;
  for (let i = 0; i < t.length; i++) total += map[t[i]] < (map[t[i + 1]] || 0) ? -map[t[i]] : map[t[i]];
  if (toRoman(total) !== t) throw new Error(`"${s}" isn't a valid Roman numeral`);
  return total;
}

function primeFactors(n) {
  const out = [];
  for (let p = 2; p * p <= n; p++) while (n % p === 0) out.push(p), (n /= p);
  if (n > 1) out.push(n);
  return out;
}
const gcd = (a, b) => (b ? gcd(b, a % b) : Math.abs(a));

function hexToRgb(h) {
  let s = h.replace("#", "");
  if (s.length === 3) s = [...s].map((c) => c + c).join("");
  if (!/^[0-9a-f]{6}$/i.test(s)) return null;
  return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16));
}
function rgbToHsl([r, g, b]) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    h *= 60;
  }
  return [Math.round(h), Math.round(s * 100), Math.round(l * 100)];
}
function hslToRgb([h, s, l]) {
  s /= 100;
  l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)].map((x) => Math.round(x * 255));
}
const NAMED = { red: "#ff0000", green: "#008000", blue: "#0000ff", black: "#000000", white: "#ffffff", yellow: "#ffff00", orange: "#ffa500", purple: "#800080", pink: "#ffc0cb", gray: "#808080", grey: "#808080", teal: "#008080", navy: "#000080", maroon: "#800000", olive: "#808000", cyan: "#00ffff", magenta: "#ff00ff", gold: "#ffd700", silver: "#c0c0c0", brown: "#a52a2a", indigo: "#4b0082", violet: "#ee82ee", coral: "#ff7f50", salmon: "#fa8072", turquoise: "#40e0d0", lime: "#00ff00", beige: "#f5f5dc", crimson: "#dc143c" };

function passwordStrength(p) {
  const pools = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((re) => re.test(p)).length;
  const pool = [26, 26, 10, 33].filter((_, i) => [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/][i].test(p)).reduce((a, b) => a + b, 0) || 1;
  let bits = p.length * Math.log2(pool);
  const common = /^(password|123456|qwerty|letmein|admin|welcome|iloveyou|abc123|111111|monkey|dragon)/i.test(p) || /(.)\1{3,}/.test(p) || /(0123|1234|2345|3456|4567|5678|6789|abcd|qwer)/i.test(p);
  if (common) bits = Math.min(bits, 20);
  const rating = bits < 28 ? "very weak" : bits < 40 ? "weak" : bits < 60 ? "fair" : bits < 80 ? "strong" : "very strong";
  const tips = [];
  if (p.length < 12) tips.push("use at least 12 characters");
  if (pools < 3) tips.push("mix upper and lower case, numbers and symbols");
  if (common) tips.push("avoid common words, sequences and repeats");
  return { rating, entropy_bits: Math.round(bits), length: p.length, tips };
}

function diffLines(a, b) {
  const x = a.split("\n");
  const y = b.split("\n");
  const m = x.length;
  const n = y.length;
  if (m * n > 4e6) throw new Error("Texts are too long to compare");
  const L = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = m - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0;
  let j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && x[i] === y[j]) out.push(`  ${x[i++]}`), j++;
    else if (j < n && (i >= m || L[i][j + 1] >= L[i + 1][j])) out.push(`+ ${y[j++]}`);
    else out.push(`- ${x[i++]}`);
  }
  return out;
}

/* ───────────── memory (stored on this device) ───────────── */
const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "data");
const MEMORY_FILE = () => process.env.MNX_MEMORY_FILE || path.join(DATA_DIR, "memory.json");
export function loadMemories() {
  try {
    return JSON.parse(fs.readFileSync(MEMORY_FILE(), "utf8"));
  } catch {
    return [];
  }
}
function saveMemories(list) {
  fs.mkdirSync(path.dirname(MEMORY_FILE()), { recursive: true });
  fs.writeFileSync(MEMORY_FILE(), JSON.stringify(list.slice(-200), null, 1));
}
// What Mnx knows about the user, added to the end of its instructions.
export function memoryPrompt() {
  const m = loadMemories();
  return m.length ? `\n\nThings the user asked you to remember:\n${m.slice(-40).map((x) => `- ${x.fact}`).join("\n")}` : "";
}

/* ───────────── the tools ───────────── */
const LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

const T = [];
const tool = (name, sig, description, params, required, run) => T.push({ name, sig, description, params, required, run });

// Text & data
tool("generate_password", "generate_password(length?, symbols?)", "Generate a strong random password.", { length: I, symbols: B }, [], (i) => {
  const len = Math.min(128, Math.max(6, Math.round(num(i.length ?? 16, "length"))));
  const sets = ["abcdefghijkmnopqrstuvwxyz", "ABCDEFGHJKLMNPQRSTUVWXYZ", "23456789", ...(i.symbols === false ? [] : ["!@#$%^&*-_=+?"])];
  const all = sets.join("");
  const pick = (s) => s[crypto.randomInt(s.length)];
  const chars = sets.map(pick);
  while (chars.length < len) chars.push(pick(all));
  for (let k = chars.length - 1; k > 0; k--) {
    const j = crypto.randomInt(k + 1);
    [chars[k], chars[j]] = [chars[j], chars[k]];
  }
  const password = chars.join("");
  const s = passwordStrength(password);
  return { result: { password, length: len, strength: s.rating }, display: info("key", password, `${len} characters · ${s.rating}`, [], { copy: password }) };
});
tool("password_strength", "password_strength(password)", "Check how strong a password is, with tips.", { password: S }, ["password"], (i) => {
  const s = passwordStrength(str(need(i.password, "password"), 200));
  return { result: s, display: info("key", s.rating[0].toUpperCase() + s.rating.slice(1), `${s.length} characters · about ${s.entropy_bits} bits`, s.tips.map((t) => ["Tip", t])) };
});
tool("uuid", "uuid(count?)", "Generate random UUIDs (v4).", { count: I }, [], (i) => {
  const ids = Array.from({ length: Math.min(20, Math.max(1, Math.round(num(i.count ?? 1, "count")))) }, () => crypto.randomUUID());
  return { result: { uuids: ids }, display: info("key", ids[0], ids.length > 1 ? `${ids.length} UUIDs` : "UUID v4", ids.slice(1).map((u, k) => [`#${k + 2}`, u]), { copy: ids.join("\n") }) };
});
tool("word_count", "word_count(text)", "Count words, characters, sentences and reading time of a text.", { text: S }, ["text"], (i) => {
  const t = str(need(i.text, "text"), 200000);
  const words = (t.match(/[\p{L}\p{N}'’-]+/gu) || []).length;
  const r = { words, characters: [...t].length, characters_no_spaces: [...t.replace(/\s/g, "")].length, sentences: (t.match(/[^.!?]+[.!?]+/g) || []).length || (t.trim() ? 1 : 0),
    paragraphs: t.split(/\n\s*\n/).filter((p) => p.trim()).length, reading_time_min: Math.max(1, Math.round(words / 230)) };
  return { result: r, display: info("text", `${words.toLocaleString()} words`, `${r.characters.toLocaleString()} characters`, [["No spaces", r.characters_no_spaces.toLocaleString()], ["Sentences", r.sentences], ["Paragraphs", r.paragraphs], ["Reading time", `~${r.reading_time_min} min`]]) };
});
tool("change_case", "change_case(text, case: upper|lower|title|sentence|snake|camel|kebab)", "Change text to UPPER, lower, Title, Sentence, snake_case, camelCase or kebab-case.", { text: S, case: { enum: ["upper", "lower", "title", "sentence", "snake", "camel", "kebab"] } }, ["text", "case"], (i) => {
  const t = str(need(i.text, "text"), 20000);
  const w = t.trim().split(/[\s_-]+|(?<=[a-z])(?=[A-Z])/).filter(Boolean).map((x) => x.toLowerCase());
  const small = new Set(["a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "of", "in", "with"]);
  const out = { upper: t.toUpperCase(), lower: t.toLowerCase(), title: t.toLowerCase().replace(/\b[\p{L}']+/gu, (x, k) => (k && small.has(x) ? x : x[0].toUpperCase() + x.slice(1))),
    sentence: t.toLowerCase().replace(/(^\s*\p{L}|[.!?]\s+\p{L})/gu, (x) => x.toUpperCase()), snake: w.join("_"), kebab: w.join("-"),
    camel: w.map((x, k) => (k ? x[0].toUpperCase() + x.slice(1) : x)).join("") }[String(i.case).toLowerCase()];
  if (out === undefined) throw new Error("'case' must be upper, lower, title, sentence, snake, camel or kebab");
  return { result: { text: out }, display: info("text", out.slice(0, 300), `${i.case} case`, [], { copy: out }) };
});
tool("slugify", "slugify(text)", "Turn text into a URL slug (my-blog-post).", { text: S }, ["text"], (i) => {
  const slug = str(need(i.text, "text"), 500).normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return { result: { slug }, display: info("link", slug || "(empty)", "URL slug", [], { copy: slug }) };
});
tool("base64", "base64(text, mode: encode|decode)", "Encode or decode Base64.", { text: S, mode: { enum: ["encode", "decode"] } }, ["text"], (i) => {
  const t = str(need(i.text, "text"), 100000);
  const decode = String(i.mode).toLowerCase() === "decode";
  let out;
  if (decode) {
    if (!/^[A-Za-z0-9+/=\s_-]+$/.test(t)) throw new Error("That isn't valid Base64");
    out = Buffer.from(t.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
  } else out = Buffer.from(t, "utf8").toString("base64");
  return { result: { mode: decode ? "decode" : "encode", output: out }, display: info("code", out.slice(0, 300), decode ? "Decoded Base64" : "Base64", [], { copy: out }) };
});
tool("url_encode", "url_encode(text, mode: encode|decode)", "Percent-encode or decode text for URLs.", { text: S, mode: { enum: ["encode", "decode"] } }, ["text"], (i) => {
  const t = str(need(i.text, "text"), 20000);
  const decode = String(i.mode).toLowerCase() === "decode";
  let out;
  try {
    out = decode ? decodeURIComponent(t.replace(/\+/g, " ")) : encodeURIComponent(t);
  } catch {
    throw new Error("That text isn't valid URL encoding");
  }
  return { result: { mode: decode ? "decode" : "encode", output: out }, display: info("link", out.slice(0, 300), decode ? "URL-decoded" : "URL-encoded", [], { copy: out }) };
});
tool("hash_text", "hash_text(text, algorithm?: md5|sha1|sha256|sha512)", "Hash text with MD5, SHA-1, SHA-256 or SHA-512.", { text: S, algorithm: { enum: ["md5", "sha1", "sha256", "sha512"] } }, ["text"], (i) => {
  const algo = String(i.algorithm || "sha256").toLowerCase().replace("-", "");
  if (!["md5", "sha1", "sha256", "sha512"].includes(algo)) throw new Error("algorithm must be md5, sha1, sha256 or sha512");
  const hash = crypto.createHash(algo).update(str(need(i.text, "text"), 1e6)).digest("hex");
  return { result: { algorithm: algo, hash }, display: info("key", hash, algo.toUpperCase(), [], { copy: hash, mono: true }) };
});
tool("json_format", "json_format(json)", "Check and pretty-print JSON, or explain where it's broken.", { json: S }, ["json"], (i) => {
  const t = typeof i.json === "string" ? i.json : JSON.stringify(i.json);
  try {
    const v = JSON.parse(t);
    const pretty = JSON.stringify(v, null, 2);
    return { result: { valid: true, formatted: pretty.slice(0, 3000), type: Array.isArray(v) ? `array of ${v.length}` : typeof v }, display: info("code", "Valid JSON", Array.isArray(v) ? `Array of ${v.length} items` : typeof v, [], { pre: pretty.slice(0, 4000), copy: pretty }) };
  } catch (e) {
    const pos = Number(String(e.message).match(/position (\d+)/)?.[1]);
    const where = Number.isFinite(pos) ? `line ${t.slice(0, pos).split("\n").length}` : "";
    return { result: { valid: false, error: e.message, where }, display: info("code", "Invalid JSON", where || "Couldn't parse", [["Error", e.message]]) };
  }
});
tool("regex_test", "regex_test(pattern, text, flags?)", "Test a regular expression and list its matches.", { pattern: S, text: S, flags: S }, ["pattern", "text"], (i) => {
  const flags = String(i.flags || "g").replace(/[^gimsuy]/g, "");
  let re;
  try {
    re = new RegExp(str(need(i.pattern, "pattern"), 500), flags.includes("g") ? flags : `${flags}g`);
  } catch (e) {
    throw new Error(`Invalid pattern: ${e.message}`);
  }
  const t = str(need(i.text, "text"), 20000);
  const matches = [...t.matchAll(re)].slice(0, 50).map((m) => ({ match: m[0], index: m.index, groups: m.slice(1) }));
  return { result: { count: matches.length, matches }, display: info("code", `${matches.length} match${matches.length === 1 ? "" : "es"}`, `/${i.pattern}/${flags}`, matches.slice(0, 8).map((m) => [`@${m.index}`, m.match])) };
});
tool("text_diff", "text_diff(a, b)", "Compare two texts line by line.", { a: S, b: S }, ["a", "b"], (i) => {
  const lines = diffLines(str(i.a, 20000), str(i.b, 20000));
  const added = lines.filter((l) => l[0] === "+").length;
  const removed = lines.filter((l) => l[0] === "-").length;
  const shown = lines.filter((l) => l[0] !== " ").slice(0, 60);
  return { result: { added, removed, changes: shown }, display: info("code", added || removed ? `${added} added, ${removed} removed` : "No differences", "Line-by-line comparison", [], { pre: shown.join("\n") || "(identical)" }) };
});
tool("lorem_ipsum", "lorem_ipsum(paragraphs?)", "Placeholder text.", { paragraphs: I }, [], (i) => {
  const n = Math.min(10, Math.max(1, Math.round(num(i.paragraphs ?? 2, "paragraphs"))));
  const text = Array.from({ length: n }, () => LOREM).join("\n\n");
  return { result: { text }, display: info("text", "Lorem ipsum", `${n} paragraph${n > 1 ? "s" : ""}`, [], { body: text.slice(0, 900), copy: text }) };
});
tool("sort_lines", "sort_lines(text, order?: asc|desc|shuffle, unique?)", "Sort, shuffle or de-duplicate lines.", { text: S, order: { enum: ["asc", "desc", "shuffle"] }, unique: B }, ["text"], (i) => {
  let lines = str(need(i.text, "text"), 100000).split("\n").map((l) => l.trimEnd()).filter((l) => l.trim());
  if (i.unique) lines = [...new Set(lines)];
  const order = String(i.order || "asc").toLowerCase();
  if (order === "shuffle") for (let k = lines.length - 1; k > 0; k--) {
    const j = crypto.randomInt(k + 1);
    [lines[k], lines[j]] = [lines[j], lines[k]];
  }
  else lines.sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }) * (order === "desc" ? -1 : 1));
  const text = lines.join("\n");
  return { result: { lines: lines.slice(0, 200), count: lines.length }, display: info("text", `${lines.length} lines`, order === "shuffle" ? "Shuffled" : `Sorted ${order === "desc" ? "Z→A" : "A→Z"}${i.unique ? ", duplicates removed" : ""}`, [], { pre: text.slice(0, 3000), copy: text }) };
});
tool("extract_contacts", "extract_contacts(text)", "Pull out emails, phone numbers, links and hashtags from text.", { text: S }, ["text"], (i) => {
  const t = str(need(i.text, "text"), 100000);
  const uniq = (a) => [...new Set(a)];
  const r = { emails: uniq(t.match(/[\w.+-]+@[\w-]+\.[\w.-]+/g) || []), phones: uniq((t.match(/\+?\d[\d\s().-]{7,}\d/g) || []).map((p) => p.trim())),
    links: uniq(t.match(/https?:\/\/[^\s)>"']+/g) || []), hashtags: uniq(t.match(/#[\p{L}\p{N}_]+/gu) || []) };
  const rows = Object.entries(r).filter(([, v]) => v.length).map(([k, v]) => [k, v.slice(0, 10).join(", ")]);
  return { result: r, display: info("text", `${Object.values(r).reduce((n, v) => n + v.length, 0)} found`, "Emails, phones, links, hashtags", rows) };
});

// Random
tool("random_number", "random_number(min?, max?, count?)", "Random whole numbers in a range.", { min: I, max: I, count: I }, [], (i) => {
  const lo = Math.round(num(i.min ?? 1, "min"));
  const hi = Math.round(num(i.max ?? 100, "max"));
  if (hi < lo) throw new Error("'max' must be ≥ 'min'");
  const nums = Array.from({ length: Math.min(100, Math.max(1, Math.round(num(i.count ?? 1, "count")))) }, () => lo + crypto.randomInt(hi - lo + 1));
  return { result: { numbers: nums, min: lo, max: hi }, display: info("dice", nums.join(", "), `Between ${lo} and ${hi}`) };
});
tool("roll_dice", "roll_dice(dice?: e.g. '2d6')", "Roll dice like 1d20 or 3d6+2.", { dice: S }, [], (i) => {
  const m = String(i.dice || "1d6").toLowerCase().replace(/\s/g, "").match(/^(\d*)d(\d+)([+-]\d+)?$/);
  if (!m) throw new Error("Write dice like 2d6, 1d20 or 3d6+2");
  const count = Math.min(100, Number(m[1] || 1));
  const sides = Math.min(1000, Number(m[2]));
  const rolls = Array.from({ length: count }, () => 1 + crypto.randomInt(sides));
  const mod = Number(m[3] || 0);
  const total = rolls.reduce((a, b) => a + b, 0) + mod;
  return { result: { dice: `${count}d${sides}${m[3] || ""}`, rolls, total }, display: info("dice", total, `${count}d${sides}${m[3] || ""}`, [["Rolls", rolls.join(", ")]]) };
});
tool("flip_coin", "flip_coin(count?)", "Flip one or more coins.", { count: I }, [], (i) => {
  const flips = Array.from({ length: Math.min(100, Math.max(1, Math.round(num(i.count ?? 1, "count")))) }, () => (crypto.randomInt(2) ? "Heads" : "Tails"));
  const heads = flips.filter((f) => f === "Heads").length;
  return { result: { flips, heads, tails: flips.length - heads }, display: info("dice", flips.length === 1 ? flips[0] : `${heads} heads, ${flips.length - heads} tails`, flips.length === 1 ? "Coin flip" : flips.join(", ").slice(0, 200)) };
});
tool("pick_random", "pick_random(options, count?)", "Pick randomly from a list (who goes first, what to eat…).", { options: A(), count: I }, ["options"], (i) => {
  const opts = (Array.isArray(i.options) ? i.options : String(i.options || "").split(/,|\n| or /)).map((x) => String(x).trim()).filter(Boolean);
  if (opts.length < 2) throw new Error("Give at least two options");
  const k = Math.min(opts.length, Math.max(1, Math.round(num(i.count ?? 1, "count"))));
  const pool = [...opts];
  const picked = Array.from({ length: k }, () => pool.splice(crypto.randomInt(pool.length), 1)[0]);
  return { result: { picked, from: opts }, display: info("dice", picked.join(", "), `Picked from ${opts.length} options`) };
});

// Math & money
tool("statistics", "statistics(numbers)", "Mean, median, mode, range, standard deviation of numbers.", { numbers: A(N) }, ["numbers"], (i) => {
  const xs = listNums(i.numbers, "numbers");
  const s = [...xs].sort((a, b) => a - b);
  const n = xs.length;
  const sum = xs.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const counts = new Map();
  xs.forEach((x) => counts.set(x, (counts.get(x) || 0) + 1));
  const top = Math.max(...counts.values());
  const mode = top > 1 ? [...counts].filter(([, c]) => c === top).map(([x]) => x) : [];
  const variance = n > 1 ? xs.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1) : 0;
  const r = { count: n, sum: r2(sum, 6), mean: r2(mean, 6), median, mode, min: s[0], max: s[n - 1], range: s[n - 1] - s[0], std_dev: r2(Math.sqrt(variance), 6) };
  return { result: r, display: info("chart", `Mean ${fmt(mean, 4)}`, `${n} numbers`, [["Median", fmt(median, 4)], ["Mode", mode.length ? mode.join(", ") : "none"], ["Min – max", `${s[0]} – ${s[n - 1]}`], ["Std dev", fmt(r.std_dev, 4)], ["Sum", fmt(sum, 4)]]) };
});
tool("gcd_lcm", "gcd_lcm(numbers)", "Greatest common divisor and least common multiple.", { numbers: A(I) }, ["numbers"], (i) => {
  const xs = listNums(i.numbers, "numbers").map((x) => Math.abs(Math.round(x)));
  if (xs.length < 2) throw new Error("Give at least two numbers");
  const g = xs.reduce(gcd);
  const l = xs.reduce((a, b) => (a / gcd(a, b)) * b);
  return { result: { numbers: xs, gcd: g, lcm: l }, display: info("calc", `GCD ${g} · LCM ${l.toLocaleString()}`, xs.join(", ")) };
});
tool("prime_factors", "prime_factors(n)", "Is a number prime? Its prime factors.", { n: I }, ["n"], (i) => {
  const n = Math.round(num(i.n, "n"));
  if (n < 2 || n > 1e15) throw new Error("'n' must be between 2 and 10^15");
  const f = primeFactors(n);
  const grouped = Object.entries(f.reduce((m, p) => ((m[p] = (m[p] || 0) + 1), m), {})).map(([p, e]) => (e > 1 ? `${p}^${e}` : p)).join(" × ");
  return { result: { n, is_prime: f.length === 1, factors: f }, display: info("calc", f.length === 1 ? `${n.toLocaleString()} is prime` : grouped, f.length === 1 ? "Prime number" : `Prime factors of ${n.toLocaleString()}`) };
});
tool("number_to_words", "number_to_words(number, system?: international|indian)", "Write a number in words (e.g. for cheques).", { number: N, system: { enum: ["international", "indian"] } }, ["number"], (i) => {
  const n = num(i.number, "number");
  const sys = String(i.system || "international").toLowerCase() === "indian" ? "indian" : "international";
  const whole = numberToWords(n, sys);
  const frac = Math.round((Math.abs(n) % 1) * 100);
  const words = frac ? `${whole} point ${frac < 10 ? "zero " : ""}${words999(frac)}` : whole;
  return { result: { number: n, words, system: sys }, display: info("text", words, n.toLocaleString(sys === "indian" ? "en-IN" : "en-US")) };
});
tool("roman_numerals", "roman_numerals(value)", "Convert to or from Roman numerals.", { value: S }, ["value"], (i) => {
  const v = String(need(i.value, "value")).trim();
  const r = /^\d+$/.test(v) ? { number: Number(v), roman: toRoman(Number(v)) } : { roman: v.toUpperCase(), number: fromRoman(v) };
  return { result: r, display: info("calc", /^\d+$/.test(v) ? r.roman : r.number, /^\d+$/.test(v) ? `${v} in Roman numerals` : `${r.roman} as a number`) };
});
tool("number_base", "number_base(value, from_base?, to_base?)", "Convert between binary, octal, decimal and hex (bases 2–36).", { value: S, from_base: I, to_base: I }, ["value"], (i) => {
  let v = String(need(i.value, "value")).trim().toLowerCase();
  let from = Number(i.from_base || 10);
  if (/^0x/.test(v)) (from = 16), (v = v.slice(2));
  else if (/^0b/.test(v)) (from = 2), (v = v.slice(2));
  else if (/^0o/.test(v)) (from = 8), (v = v.slice(2));
  const to = Number(i.to_base || (from === 10 ? 2 : 10));
  if (![from, to].every((b) => Number.isInteger(b) && b >= 2 && b <= 36)) throw new Error("Bases must be 2–36");
  const digits = "0123456789abcdefghijklmnopqrstuvwxyz".slice(0, from);
  if (!v || [...v].some((c) => !digits.includes(c))) throw new Error(`"${i.value}" isn't a valid base-${from} number`);
  const big = [...v].reduce((acc, c) => acc * BigInt(from) + BigInt(digits.indexOf(c)), 0n);
  const out = big.toString(to);
  const all = { binary: big.toString(2), octal: big.toString(8), decimal: big.toString(10), hex: big.toString(16).toUpperCase() };
  return { result: { value: i.value, from_base: from, to_base: to, output: to === 16 ? out.toUpperCase() : out, ...all }, display: info("calc", to === 16 ? out.toUpperCase() : out, `${i.value} (base ${from}) in base ${to}`, [["Binary", all.binary], ["Octal", all.octal], ["Decimal", all.decimal], ["Hex", all.hex]], { mono: true }) };
});
tool("percentage_change", "percentage_change(from, to)", "Percentage increase or decrease between two numbers.", { from: N, to: N }, ["from", "to"], (i) => {
  const a = num(i.from, "from");
  const b = num(i.to, "to");
  if (a === 0) throw new Error("Can't compute a percentage change from 0");
  const pct = ((b - a) / Math.abs(a)) * 100;
  return { result: { from: a, to: b, change: r2(b - a, 6), percent: r2(pct, 4) }, display: info("chart", `${pct >= 0 ? "+" : ""}${fmt(pct)}%`, `${fmt(a, 4)} → ${fmt(b, 4)}`, [["Difference", `${b - a >= 0 ? "+" : ""}${fmt(b - a, 4)}`]]) };
});
tool("discount", "discount(price, percent)", "Sale price after a percentage discount.", { price: N, percent: N }, ["price", "percent"], (i) => {
  const p = num(i.price, "price");
  const pc = num(i.percent, "percent");
  const save = (p * pc) / 100;
  return { result: { price: p, percent: pc, you_save: r2(save), final_price: r2(p - save) }, display: info("money", fmt(p - save), `${fmt(p)} − ${pc}%`, [["You save", fmt(save)]]) };
});
tool("tax", "tax(amount, rate, inclusive?)", "Add tax/GST/VAT to an amount, or work out the tax inside a price that includes it.", { amount: N, rate: N, inclusive: B }, ["amount", "rate"], (i) => {
  const a = num(i.amount, "amount");
  const r = num(i.rate, "rate");
  const net = i.inclusive ? a / (1 + r / 100) : a;
  const t = (net * r) / 100;
  return { result: { net: r2(net), tax: r2(t), gross: r2(net + t), rate: r, inclusive: !!i.inclusive }, display: info("money", fmt(net + t), i.inclusive ? `Price incl. ${r}% tax` : `${fmt(a)} + ${r}% tax`, [["Before tax", fmt(net)], ["Tax", fmt(t)]]) };
});
tool("tip_split", "tip_split(bill, tip_percent?, people?)", "Tip and how much each person pays.", { bill: N, tip_percent: N, people: I }, ["bill"], (i) => {
  const bill = num(i.bill, "bill");
  const tp = num(i.tip_percent ?? 10, "tip_percent");
  const people = Math.max(1, Math.round(num(i.people ?? 1, "people")));
  const tip = (bill * tp) / 100;
  const each = (bill + tip) / people;
  return { result: { bill, tip_percent: tp, tip: r2(tip), total: r2(bill + tip), people, each: r2(each) }, display: info("money", `${fmt(each)} each`, `${people} ${people === 1 ? "person" : "people"} · ${tp}% tip`, [["Tip", fmt(tip)], ["Total", fmt(bill + tip)]]) };
});
tool("loan_emi", "loan_emi(principal, annual_rate, months)", "Monthly loan payment (EMI), total interest and total paid.", { principal: N, annual_rate: N, months: I }, ["principal", "annual_rate", "months"], (i) => {
  const P = num(i.principal, "principal");
  const R = num(i.annual_rate, "annual_rate") / 1200;
  const n = Math.round(num(i.months, "months"));
  if (n < 1) throw new Error("'months' must be at least 1");
  const emi = R ? (P * R * (1 + R) ** n) / ((1 + R) ** n - 1) : P / n;
  return { result: { principal: P, annual_rate: i.annual_rate, months: n, emi: r2(emi), total_paid: r2(emi * n), total_interest: r2(emi * n - P) }, display: info("money", `${fmt(emi)} / month`, `${fmt(P)} at ${i.annual_rate}% for ${n} months`, [["Total interest", fmt(emi * n - P)], ["Total paid", fmt(emi * n)]]) };
});
tool("compound_interest", "compound_interest(principal, rate, years, per_year?, monthly_add?)", "Savings growth with compound interest and optional monthly deposits.", { principal: N, rate: N, years: N, per_year: I, monthly_add: N }, ["principal", "rate", "years"], (i) => {
  const P = num(i.principal, "principal");
  const r = num(i.rate, "rate") / 100;
  const y = num(i.years, "years");
  const k = Math.max(1, Math.round(num(i.per_year ?? 12, "per_year")));
  const add = num(i.monthly_add ?? 0, "monthly_add");
  let bal = P;
  let deposited = P;
  const months = Math.round(y * 12);
  for (let m = 1; m <= months; m++) {
    bal += add;
    deposited += add;
    if ((m * k) % 12 === 0 || k > 12) bal *= (1 + r / k) ** Math.max(1, k / 12);
  }
  return { result: { final_amount: r2(bal), deposited: r2(deposited), interest_earned: r2(bal - deposited), years: y, rate: i.rate }, display: info("money", fmt(bal), `After ${y} years at ${i.rate}%`, [["Deposited", fmt(deposited)], ["Interest earned", fmt(bal - deposited)]]) };
});
tool("fuel_cost", "fuel_cost(distance_km, km_per_liter, price_per_liter)", "Fuel needed and cost for a trip.", { distance_km: N, km_per_liter: N, price_per_liter: N }, ["distance_km", "km_per_liter", "price_per_liter"], (i) => {
  const d = num(i.distance_km, "distance_km");
  const eff = num(i.km_per_liter, "km_per_liter");
  const price = num(i.price_per_liter, "price_per_liter");
  if (eff <= 0) throw new Error("'km_per_liter' must be more than 0");
  const liters = d / eff;
  return { result: { liters: r2(liters), cost: r2(liters * price) }, display: info("money", fmt(liters * price), `${fmt(d)} km at ${eff} km/L`, [["Fuel", `${fmt(liters)} L`]]) };
});
tool("pace", "pace(distance_km, time: 'h:mm:ss' or minutes)", "Running/cycling pace and speed.", { distance_km: N, time: S }, ["distance_km", "time"], (i) => {
  const d = num(i.distance_km, "distance_km");
  const t = String(need(i.time, "time")).trim();
  const parts = t.split(":").map(Number);
  if (parts.some((x) => !Number.isFinite(x))) throw new Error("time must be like 45:30 or 1:05:00, or minutes");
  const secs = parts.length === 1 ? parts[0] * 60 : parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2];
  const per = secs / d;
  const mmss = `${Math.floor(per / 60)}:${String(Math.round(per % 60)).padStart(2, "0")}`;
  return { result: { pace_per_km: mmss, speed_kmh: r2((d / secs) * 3600) }, display: info("run", `${mmss} /km`, `${d} km in ${t}`, [["Speed", `${fmt((d / secs) * 3600)} km/h`]]) };
});

// Health
tool("bmi", "bmi(weight_kg, height_cm)", "Body mass index and its category.", { weight_kg: N, height_cm: N }, ["weight_kg", "height_cm"], (i) => {
  const w = num(i.weight_kg, "weight_kg");
  const h = num(i.height_cm, "height_cm") / 100;
  if (h <= 0.5 || h > 2.6 || w <= 0) throw new Error("Check the weight (kg) and height (cm)");
  const b = w / h ** 2;
  const cat = b < 18.5 ? "underweight" : b < 25 ? "healthy weight" : b < 30 ? "overweight" : "obese";
  return { result: { bmi: r2(b, 1), category: cat, healthy_weight_range_kg: [r2(18.5 * h * h, 1), r2(24.9 * h * h, 1)] }, display: info("heart", r2(b, 1), cat, [["Healthy range", `${r2(18.5 * h * h, 1)}–${r2(24.9 * h * h, 1)} kg`]]) };
});
tool("bmr", "bmr(weight_kg, height_cm, age, sex: male|female, activity?)", "Calories burned at rest (BMR) and per day (TDEE).", { weight_kg: N, height_cm: N, age: I, sex: { enum: ["male", "female"] }, activity: { enum: ["sedentary", "light", "moderate", "active", "very active"] } }, ["weight_kg", "height_cm", "age", "sex"], (i) => {
  const base = 10 * num(i.weight_kg, "weight_kg") + 6.25 * num(i.height_cm, "height_cm") - 5 * num(i.age, "age");
  const bmr = base + (String(i.sex).toLowerCase().startsWith("f") ? -161 : 5);
  const f = { sedentary: 1.2, light: 1.375, moderate: 1.55, active: 1.725, "very active": 1.9 }[String(i.activity || "sedentary").toLowerCase()] || 1.2;
  return { result: { bmr: Math.round(bmr), daily_calories: Math.round(bmr * f), activity: i.activity || "sedentary" }, display: info("heart", `${Math.round(bmr * f)} kcal/day`, `BMR ${Math.round(bmr)} kcal (Mifflin–St Jeor)`, [["Activity", i.activity || "sedentary"]]) };
});

// Dates & time
tool("age", "age(birthdate)", "Exact age from a birth date, and days to the next birthday.", { birthdate: S }, ["birthdate"], (i) => {
  const b = parseDate(need(i.birthdate, "birthdate"), "birthdate");
  const now = parseDate("today");
  if (b > now) throw new Error("The birth date is in the future");
  const d = ymdDiff(b, now);
  let next = new Date(now.getFullYear(), b.getMonth(), b.getDate());
  if (next < now) next = new Date(now.getFullYear() + 1, b.getMonth(), b.getDate());
  return { result: { ...d, days_lived: daysBetween(b, now), next_birthday_in_days: daysBetween(now, next) }, display: info("cake", `${d.years} years`, `${d.months} months, ${d.days} days`, [["Days lived", daysBetween(b, now).toLocaleString()], ["Next birthday", `in ${daysBetween(now, next)} days`]]) };
});
tool("date_diff", "date_diff(from, to)", "Time between two dates in days, weeks, months and years.", { from: S, to: S }, ["from", "to"], (i) => {
  const a = parseDate(need(i.from, "from"), "from");
  const b = parseDate(i.to || "today", "to");
  const days = daysBetween(a, b);
  const d = ymdDiff(a, b);
  return { result: { total_days: days, weeks: r2(days / 7, 2), years_months_days: `${d.years} years, ${d.months} months, ${d.days} days`, from: isoDate(a), to: isoDate(b) }, display: info("calendar", `${Math.abs(days).toLocaleString()} days`, `${isoDate(a)} → ${isoDate(b)}`, [["Weeks", fmt(Math.abs(days) / 7, 1)], ["Y / M / D", `${d.years} y, ${d.months} m, ${d.days} d`]]) };
});
tool("add_to_date", "add_to_date(date?, days?, weeks?, months?, years?)", "Add or subtract time from a date (negative to subtract).", { date: S, days: I, weeks: I, months: I, years: I }, [], (i) => {
  const d = parseDate(i.date || "today");
  // Months/years first, keeping the day inside the month (Jan 31 + 1 month = Feb 28), then days.
  const y = d.getFullYear() + Number(i.years || 0);
  const mo = d.getMonth() + Number(i.months || 0);
  const lastDay = new Date(y, mo + 1, 0).getDate();
  const out = new Date(y, mo, Math.min(d.getDate(), lastDay) + Number(i.days || 0) + 7 * Number(i.weeks || 0));
  return { result: { date: isoDate(out), weekday: out.toLocaleDateString("en-GB", { weekday: "long" }) }, display: info("calendar", longDate(out), `From ${isoDate(d)}`) };
});
tool("day_of_week", "day_of_week(date)", "Which day of the week a date falls on.", { date: S }, ["date"], (i) => {
  const d = parseDate(need(i.date, "date"));
  return { result: { date: isoDate(d), weekday: d.toLocaleDateString("en-GB", { weekday: "long" }), day_of_year: daysBetween(new Date(d.getFullYear(), 0, 1), d) + 1 }, display: info("calendar", d.toLocaleDateString("en-GB", { weekday: "long" }), longDate(d)) };
});
tool("countdown", "countdown(date, event?)", "Days until a date or event.", { date: S, event: S }, ["date"], (i) => {
  const d = parseDate(need(i.date, "date"));
  const n = daysBetween(parseDate("today"), d);
  return { result: { event: i.event || null, date: isoDate(d), days: n, weeks: r2(n / 7, 1) }, display: info("calendar", n >= 0 ? `${n} days` : `${-n} days ago`, `${i.event ? `${i.event} · ` : ""}${longDate(d)}`) };
});
tool("unix_time", "unix_time(value?)", "Convert a Unix timestamp to a date, or a date to a timestamp (omit for now).", { value: S }, [], (i) => {
  const v = String(i.value ?? "").trim();
  let d;
  if (!v) d = new Date();
  else if (/^\d{9,13}$/.test(v)) d = new Date(v.length > 11 ? Number(v) : Number(v) * 1000);
  else d = new Date(v);
  if (Number.isNaN(d.getTime())) throw new Error("Give a Unix timestamp or a date");
  return { result: { unix: Math.floor(d.getTime() / 1000), iso_utc: d.toISOString() }, display: info("clock", Math.floor(d.getTime() / 1000), d.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC"), [], { mono: true }) };
});
tool("timezone_convert", "timezone_convert(time, from_zone, to_zone)", "Convert a time between time zones (IANA names like Asia/Kolkata, or UTC).", { time: S, from_zone: S, to_zone: S }, ["time", "from_zone", "to_zone"], (i) => {
  const ALIAS = { ist: "Asia/Kolkata", india: "Asia/Kolkata", utc: "UTC", gmt: "UTC", est: "America/New_York", edt: "America/New_York", pst: "America/Los_Angeles", pdt: "America/Los_Angeles", cst: "America/Chicago", bst: "Europe/London", cet: "Europe/Paris", jst: "Asia/Tokyo", aest: "Australia/Sydney", sgt: "Asia/Singapore", gst: "Asia/Dubai" };
  const zone = (z) => {
    const k = String(z || "").trim();
    const resolved = ALIAS[k.toLowerCase()] || k;
    try {
      new Intl.DateTimeFormat("en-US", { timeZone: resolved });
      return resolved;
    } catch {
      throw new Error(`Unknown time zone "${z}" (use names like Asia/Kolkata or Europe/London)`);
    }
  };
  const from = zone(i.from_zone);
  const to = zone(i.to_zone);
  const m = String(need(i.time, "time")).trim().toLowerCase().match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
  if (!m) throw new Error("time must be like 14:30 or 2:30 pm");
  let h = Number(m[1]) % (m[3] ? 12 : 24);
  if (m[3] === "pm") h += 12;
  const min = Number(m[2] || 0);
  // Find the UTC instant whose wall-clock time in `from` is today at h:min.
  const now = new Date();
  const parts = (d, z) => Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: z, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).formatToParts(d).map((p) => [p.type, p.value]));
  const p = parts(now, from);
  let guess = Date.UTC(Number(p.year), Number(p.month) - 1, Number(p.day), h, min);
  for (let k = 0; k < 3; k++) {
    const q = parts(new Date(guess), from);
    const wall = Date.UTC(Number(q.year), Number(q.month) - 1, Number(q.day), Number(q.hour), Number(q.minute));
    guess += Date.UTC(Number(p.year), Number(p.month) - 1, Number(p.day), h, min) - wall;
  }
  const out = parts(new Date(guess), to);
  const dayShift = Date.UTC(Number(out.year), Number(out.month) - 1, Number(out.day)) - Date.UTC(Number(p.year), Number(p.month) - 1, Number(p.day));
  const shift = dayShift > 0 ? " (next day)" : dayShift < 0 ? " (previous day)" : "";
  const time = `${out.hour}:${out.minute}`;
  return { result: { from_zone: from, to_zone: to, input: `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`, time: time + shift }, display: info("clock", time + shift, `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")} ${from} → ${to}`) };
});

// Colors
tool("color_convert", "color_convert(color)", "Convert a color between HEX, RGB and HSL, with a preview.", { color: S }, ["color"], (i) => {
  const c = String(need(i.color, "color")).trim().toLowerCase();
  let rgb = hexToRgb(NAMED[c] || c);
  const rm = c.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  const hm = c.match(/^hsla?\(\s*(\d+)\s*,\s*(\d+)%?\s*,\s*(\d+)%?/);
  if (rm) rgb = rm.slice(1, 4).map(Number);
  if (hm) rgb = hslToRgb(hm.slice(1, 4).map(Number));
  if (!rgb || rgb.some((x) => x > 255)) throw new Error("Give a color like #1e90ff, rgb(30,144,255), hsl(210,100%,56%) or a name");
  const hex = `#${rgb.map((x) => x.toString(16).padStart(2, "0")).join("")}`;
  const hsl = rgbToHsl(rgb);
  return { result: { hex, rgb: `rgb(${rgb.join(", ")})`, hsl: `hsl(${hsl[0]}, ${hsl[1]}%, ${hsl[2]}%)` }, display: info("palette", hex, `rgb(${rgb.join(", ")})`, [["HSL", `hsl(${hsl[0]}, ${hsl[1]}%, ${hsl[2]}%)`]], { swatch: hex, mono: true }) };
});

// Online (free, no key)
tool("public_holidays", "public_holidays(country_code, year?)", "Public holidays in a country (2-letter code like IN, US, GB).", { country_code: S, year: I }, ["country_code"], async (i) => {
  const cc = String(need(i.country_code, "country_code")).trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(cc)) throw new Error("country_code must be 2 letters, like IN, US, GB");
  const year = Math.round(num(i.year ?? new Date().getFullYear(), "year"));
  const data = await getJson(`https://date.nager.at/api/v3/PublicHolidays/${year}/${cc}`);
  const list = data.map((h) => ({ date: h.date, name: h.localName === h.name ? h.name : `${h.name} (${h.localName})` }));
  const today = isoDate(parseDate("today"));
  const next = list.find((h) => h.date >= today);
  return { result: { country: cc, year, holidays: list.slice(0, 40), next: next || null }, display: info("calendar", `${list.length} holidays`, `${cc} · ${year}${next ? ` · next: ${next.name} (${next.date})` : ""}`, list.slice(0, 14).map((h) => [h.date, h.name])) };
});
tool("country_info", "country_info(country)", "Capital, population, languages, currency, region, calling code and flag of a country.", { country: S }, ["country"], async (i) => {
  const q = String(need(i.country, "country")).trim();
  const fields = "name,capital,population,languages,currencies,region,subregion,idd,flag,timezones,area,cca2";
  let data;
  try {
    data = await getJson(`https://restcountries.com/v3.1/${/^[a-z]{2,3}$/i.test(q) ? "alpha" : "name"}/${encodeURIComponent(q)}?fields=${fields}`);
  } catch (e) {
    if (/404/.test(e.message)) throw new Error(`No country called "${q}"`);
    throw e;
  }
  const c = Array.isArray(data) ? data.find((x) => x.name?.common?.toLowerCase() === q.toLowerCase()) || data[0] : data;
  const r = { name: c.name?.common, official_name: c.name?.official, capital: c.capital?.[0], population: c.population, area_km2: c.area, region: [c.subregion, c.region].filter(Boolean).join(", "),
    languages: Object.values(c.languages || {}), currencies: Object.entries(c.currencies || {}).map(([code, v]) => `${v.name} (${code}${v.symbol ? `, ${v.symbol}` : ""})`),
    calling_code: c.idd?.root ? `${c.idd.root}${c.idd.suffixes?.length === 1 ? c.idd.suffixes[0] : ""}` : undefined, timezones: c.timezones?.slice(0, 3), flag: c.flag };
  return { result: r, display: info("globe", `${r.flag || ""} ${r.name}`.trim(), r.official_name, [["Capital", r.capital], ["Population", r.population?.toLocaleString()], ["Region", r.region], ["Languages", r.languages.join(", ")], ["Currency", r.currencies.join(", ")], ["Calling code", r.calling_code]]) };
});
tool("crypto_price", "crypto_price(coin, currency?)", "Current price and 24-hour change of a cryptocurrency (bitcoin, ethereum…).", { coin: S, currency: S }, ["coin"], async (i) => {
  const ALIAS = { btc: "bitcoin", eth: "ethereum", sol: "solana", doge: "dogecoin", ada: "cardano", xrp: "ripple", bnb: "binancecoin", ltc: "litecoin", dot: "polkadot", matic: "matic-network", usdt: "tether" };
  const coin = ALIAS[String(need(i.coin, "coin")).toLowerCase().trim()] || String(i.coin).toLowerCase().trim().replace(/\s+/g, "-");
  const cur = String(i.currency || "usd").toLowerCase();
  const data = await getJson(`https://api.coingecko.com/api/v3/simple/price?ids=${encodeURIComponent(coin)}&vs_currencies=${encodeURIComponent(cur)}&include_24hr_change=true`);
  const d = data[coin];
  if (!d || d[cur] === undefined) throw new Error(`No price for "${i.coin}" in ${cur.toUpperCase()}`);
  const ch = d[`${cur}_24h_change`];
  return { result: { coin, currency: cur.toUpperCase(), price: d[cur], change_24h_percent: ch === undefined ? null : r2(ch) }, display: info("chart", `${Number(d[cur]).toLocaleString("en-US", { maximumFractionDigits: 6 })} ${cur.toUpperCase()}`, coin, [["24 h", ch === undefined ? "" : `${ch >= 0 ? "+" : ""}${fmt(ch)}%`]]) };
});
tool("book_search", "book_search(query)", "Find books by title, author or subject (Open Library).", { query: S }, ["query"], async (i) => {
  const data = await getJson(`https://openlibrary.org/search.json?${new URLSearchParams({ q: String(need(i.query, "query")), limit: "6", fields: "title,author_name,first_publish_year,key,number_of_pages_median" })}`);
  const books = (data.docs || []).map((b) => ({ title: b.title, author: b.author_name?.slice(0, 2).join(", "), year: b.first_publish_year, pages: b.number_of_pages_median, url: `https://openlibrary.org${b.key}` }));
  if (!books.length) throw new Error(`No books found for "${i.query}"`);
  return { result: { books }, display: info("book", `${books.length} books`, `“${i.query}”`, books.map((b) => [b.year || "", `${b.title}${b.author ? ` — ${b.author}` : ""}`])) };
});
tool("find_words", "find_words(word, kind: synonyms|antonyms|rhymes|similar_sound)", "Synonyms, antonyms or rhymes for a word (Datamuse).", { word: S, kind: { enum: ["synonyms", "antonyms", "rhymes", "similar_sound"] } }, ["word"], async (i) => {
  const w = String(need(i.word, "word")).trim();
  const kind = String(i.kind || "synonyms").toLowerCase();
  const param = { synonyms: "rel_syn", antonyms: "rel_ant", rhymes: "rel_rhy", similar_sound: "sl" }[kind] || "rel_syn";
  let data = await getJson(`https://api.datamuse.com/words?${param}=${encodeURIComponent(w)}&max=20`);
  if (!data.length && kind === "synonyms") data = await getJson(`https://api.datamuse.com/words?ml=${encodeURIComponent(w)}&max=20`);
  const words = data.map((x) => x.word).filter((x) => x !== w).slice(0, 20);
  return { result: { word: w, kind, words }, display: info("book", words.slice(0, 8).join(", ") || "Nothing found", `${kind.replace("_", " ")} for “${w}”`) };
});
tool("air_quality", "air_quality(location)", "Current air quality (AQI, PM2.5, PM10) in a city.", { location: S }, ["location"], async (i) => {
  const g = await getJson(`https://geocoding-api.open-meteo.com/v1/search?${new URLSearchParams({ name: String(need(i.location, "location")).split(",")[0], count: "1" })}`);
  const p = g.results?.[0];
  if (!p) throw new Error(`Couldn't find "${i.location}"`);
  const a = await getJson(`https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${p.latitude}&longitude=${p.longitude}&current=us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide`);
  const c = a.current || {};
  const aqi = c.us_aqi;
  const level = aqi <= 50 ? "Good" : aqi <= 100 ? "Moderate" : aqi <= 150 ? "Unhealthy for sensitive groups" : aqi <= 200 ? "Unhealthy" : aqi <= 300 ? "Very unhealthy" : "Hazardous";
  return { result: { location: `${p.name}, ${p.country}`, us_aqi: aqi, level, pm2_5: c.pm2_5, pm10: c.pm10, ozone: c.ozone, no2: c.nitrogen_dioxide }, display: info("leaf", `AQI ${aqi} · ${level}`, `${p.name}, ${p.country}`, [["PM2.5", `${c.pm2_5} µg/m³`], ["PM10", `${c.pm10} µg/m³`], ["Ozone", `${c.ozone} µg/m³`]]) };
});
tool("sunrise_sunset", "sunrise_sunset(location, date?)", "Sunrise, sunset and day length for a place and date.", { location: S, date: S }, ["location"], async (i) => {
  const g = await getJson(`https://geocoding-api.open-meteo.com/v1/search?${new URLSearchParams({ name: String(need(i.location, "location")).split(",")[0], count: "1" })}`);
  const p = g.results?.[0];
  if (!p) throw new Error(`Couldn't find "${i.location}"`);
  const d = isoDate(parseDate(i.date || "today"));
  const w = await getJson(`https://api.open-meteo.com/v1/forecast?latitude=${p.latitude}&longitude=${p.longitude}&daily=sunrise,sunset,daylight_duration&timezone=auto&start_date=${d}&end_date=${d}`);
  const len = w.daily?.daylight_duration?.[0];
  const r = { location: `${p.name}, ${p.country}`, date: d, sunrise: w.daily?.sunrise?.[0]?.slice(11), sunset: w.daily?.sunset?.[0]?.slice(11), day_length: len ? `${Math.floor(len / 3600)} h ${Math.round((len % 3600) / 60)} min` : undefined };
  return { result: r, display: info("sun", `${r.sunrise} – ${r.sunset}`, `${r.location} · ${d}`, [["Daylight", r.day_length]]) };
});
tool("recent_earthquakes", "recent_earthquakes(min_magnitude?)", "Significant earthquakes worldwide in the last 7 days (USGS).", { min_magnitude: N }, [], async (i) => {
  const min = num(i.min_magnitude ?? 4.5, "min_magnitude");
  const feed = min >= 4.5 ? "4.5_week" : min >= 2.5 ? "2.5_week" : "1.0_week";
  const data = await getJson(`https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/${feed}.geojson`);
  const qs = data.features.filter((f) => f.properties.mag >= min).slice(0, 10).map((f) => ({ magnitude: f.properties.mag, place: f.properties.place, time: new Date(f.properties.time).toISOString().slice(0, 16).replace("T", " ") + " UTC" }));
  return { result: { count: qs.length, earthquakes: qs }, display: info("globe", `${qs.length} earthquakes ≥ M${min}`, "Last 7 days (USGS)", qs.map((q) => [`M${q.magnitude}`, `${q.place} · ${q.time}`])) };
});
tool("random_joke", "random_joke()", "A clean, family-friendly joke.", {}, [], async () => {
  const j = await getJson("https://icanhazdadjoke.com/");
  return { result: { joke: j.joke }, display: info("smile", j.joke, "Dad joke") };
});
tool("quote", "quote()", "An inspiring quote.", {}, [], async () => {
  const [q] = await getJson("https://zenquotes.io/api/random");
  return { result: { quote: q.q, author: q.a }, display: info("quote", `“${q.q}”`, `— ${q.a}`) };
});
tool("github_repo", "github_repo(repo: 'owner/name')", "Stars, forks, language and description of a GitHub repository.", { repo: S }, ["repo"], async (i) => {
  const repo = String(need(i.repo, "repo")).replace(/^https?:\/\/github\.com\//, "").replace(/\.git$|\/$/g, "");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error("repo must be like owner/name");
  const r = await getJson(`https://api.github.com/repos/${repo}`, { accept: "application/vnd.github+json" });
  const out = { name: r.full_name, description: r.description, stars: r.stargazers_count, forks: r.forks_count, open_issues: r.open_issues_count, language: r.language, license: r.license?.spdx_id, updated: r.pushed_at?.slice(0, 10), url: r.html_url };
  return { result: out, display: info("code", out.name, out.description || "GitHub repository", [["Stars", out.stars?.toLocaleString()], ["Forks", out.forks?.toLocaleString()], ["Language", out.language], ["License", out.license], ["Last push", out.updated]], { link: out.url, linkText: "Open on GitHub" }) };
});
tool("package_info", "package_info(name, registry?: npm|pypi)", "Latest version and description of an npm or PyPI package.", { name: S, registry: { enum: ["npm", "pypi"] } }, ["name"], async (i) => {
  const name = String(need(i.name, "name")).trim();
  if (String(i.registry || "npm").toLowerCase() === "pypi") {
    const d = (await getJson(`https://pypi.org/pypi/${encodeURIComponent(name)}/json`)).info;
    const out = { name: d.name, version: d.version, summary: d.summary, install: `pip install ${d.name}`, url: d.package_url };
    return { result: out, display: info("code", `${out.name} ${out.version}`, out.summary || "PyPI package", [["Install", out.install]], { link: out.url, linkText: "Open on PyPI" }) };
  }
  const d = await getJson(`https://registry.npmjs.org/${encodeURIComponent(name).replace("%40", "@")}/latest`);
  const out = { name: d.name, version: d.version, description: d.description, install: `npm install ${d.name}`, url: `https://www.npmjs.com/package/${d.name}` };
  return { result: out, display: info("code", `${out.name} ${out.version}`, out.description || "npm package", [["Install", out.install]], { link: out.url, linkText: "Open on npm" }) };
});

// Memory, notes & timers
tool("remember", "remember(fact)", "Save a fact the user wants you to remember (their preferences, details, plans). Use when they say 'remember…'.", { fact: S }, ["fact"], (i) => {
  const fact = String(need(i.fact, "fact")).trim().slice(0, 300);
  const list = loadMemories().filter((m) => m.fact.toLowerCase() !== fact.toLowerCase());
  list.push({ fact, saved: new Date().toISOString() });
  saveMemories(list);
  return { result: { saved: true, fact, total_memories: list.length }, display: info("brain", "Saved to memory", fact) };
});
tool("recall", "recall(query?)", "List what you've been asked to remember (optionally matching a word).", { query: S }, [], (i) => {
  const q = String(i.query || "").toLowerCase().trim();
  const list = loadMemories().filter((m) => !q || m.fact.toLowerCase().includes(q));
  return { result: { memories: list.map((m) => m.fact) }, display: info("brain", `${list.length} memor${list.length === 1 ? "y" : "ies"}`, q ? `matching “${q}”` : "Everything Mnx remembers", list.slice(-12).map((m) => ["•", m.fact])) };
});
tool("forget", "forget(fact)", "Delete remembered facts containing some words (or 'everything').", { fact: S }, ["fact"], (i) => {
  const q = String(need(i.fact, "fact")).toLowerCase().trim();
  const list = loadMemories();
  const keep = q === "everything" || q === "all" ? [] : list.filter((m) => !m.fact.toLowerCase().includes(q));
  saveMemories(keep);
  return { result: { removed: list.length - keep.length, remaining: keep.length }, display: info("brain", `Forgot ${list.length - keep.length}`, `${keep.length} left`) };
});
tool("set_timer", "set_timer(minutes, label?)", "Start a countdown timer that rings in the app (e.g. tea, cooking, study breaks).", { minutes: N, label: S }, ["minutes"], (i) => {
  const m = num(i.minutes, "minutes");
  if (m <= 0 || m > 24 * 60) throw new Error("minutes must be between 0 and 1440");
  const label = String(i.label || "Timer").slice(0, 80);
  const ends = new Date(Date.now() + m * 60000).toISOString();
  return { result: { set: true, minutes: m, label, ends_at: ends }, display: { kind: "timer", label, seconds: Math.round(m * 60), ends_at: ends } };
});
tool("save_note", "save_note(title, text)", "Save a note on this device (shopping list, ideas…).", { title: S, text: S }, ["text"], (i) => {
  const file = path.join(path.dirname(MEMORY_FILE()), "notes.json");
  let notes = [];
  try {
    notes = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    /* none yet */
  }
  const note = { title: String(i.title || String(i.text).split("\n")[0]).slice(0, 80), text: String(need(i.text, "text")).slice(0, 5000), saved: new Date().toISOString() };
  notes = notes.filter((n) => n.title !== note.title).concat(note).slice(-100);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(notes, null, 1));
  return { result: { saved: true, title: note.title, total_notes: notes.length }, display: info("note", note.title, "Note saved", [], { body: note.text.slice(0, 600) }) };
});
tool("list_notes", "list_notes(query?)", "Show saved notes (optionally matching a word).", { query: S }, [], (i) => {
  let notes = [];
  try {
    notes = JSON.parse(fs.readFileSync(path.join(path.dirname(MEMORY_FILE()), "notes.json"), "utf8"));
  } catch {
    /* none yet */
  }
  const q = String(i.query || "").toLowerCase();
  const hits = notes.filter((n) => !q || `${n.title} ${n.text}`.toLowerCase().includes(q));
  return { result: { notes: hits.slice(-20).map((n) => ({ title: n.title, text: n.text.slice(0, 500) })) }, display: info("note", `${hits.length} note${hits.length === 1 ? "" : "s"}`, q ? `matching “${q}”` : "Saved on this device", hits.slice(-10).map((n) => [n.title, n.text.slice(0, 120)])) };
});

// Developer helpers
tool("cron_explain", "cron_explain(expression)", "Explain a cron schedule like '0 9 * * 1-5' in plain English.", { expression: S }, ["expression"], (i) => {
  const f = String(need(i.expression, "expression")).trim().split(/\s+/);
  if (f.length !== 5) throw new Error("A cron expression has 5 fields: minute hour day month weekday");
  const [mi, h, dom, mon, dow] = f;
  const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const MONTHS = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const list = (v, names) => v.split(",").map((p) => (p.includes("-") ? p.split("-").map((x) => names?.[Number(x)] ?? x).join(" to ") : names?.[Number(p)] ?? p)).join(", ");
  const time = mi.startsWith("*/") ? `every ${mi.slice(2)} minutes` : h === "*" ? (mi === "*" ? "every minute" : `at minute ${mi} of every hour`) : h.startsWith("*/") ? `every ${h.slice(2)} hours at minute ${mi}` : `at ${list(h)}:${mi.padStart(2, "0")}`;
  const parts = [time];
  if (dom !== "*") parts.push(`on day ${list(dom)} of the month`);
  if (mon !== "*") parts.push(`in ${list(mon, MONTHS)}`);
  if (dow !== "*") parts.push(`on ${list(dow, DAYS)}`);
  const text = parts.join(", ");
  return { result: { expression: f.join(" "), meaning: text }, display: info("clock", text[0].toUpperCase() + text.slice(1), f.join(" "), [], { mono: true }) };
});
tool("http_status", "http_status(code)", "What an HTTP status code means and what to do.", { code: I }, ["code"], (i) => {
  const C = { 200: ["OK", "The request worked."], 201: ["Created", "A new resource was created."], 204: ["No Content", "Worked; nothing to return."], 301: ["Moved Permanently", "Use the new URL in the Location header."], 302: ["Found", "Temporary redirect."], 304: ["Not Modified", "Use your cached copy."],
    400: ["Bad Request", "The request is malformed. Check the body, parameters and headers."], 401: ["Unauthorized", "Missing or invalid credentials. Log in or send a valid token."], 403: ["Forbidden", "You're authenticated but not allowed. Check permissions."], 404: ["Not Found", "The URL or resource doesn't exist. Check the path."],
    405: ["Method Not Allowed", "Use a different HTTP method (GET/POST…)."], 408: ["Request Timeout", "The server waited too long. Retry."], 409: ["Conflict", "The request conflicts with the current state (e.g. duplicate)."], 413: ["Payload Too Large", "Send a smaller body."], 415: ["Unsupported Media Type", "Fix the Content-Type header."],
    422: ["Unprocessable Content", "The data failed validation."], 429: ["Too Many Requests", "You're rate limited. Wait and retry (see Retry-After)."], 500: ["Internal Server Error", "The server crashed handling it. Check server logs."], 502: ["Bad Gateway", "An upstream server sent a bad response."], 503: ["Service Unavailable", "The server is down or overloaded. Retry later."], 504: ["Gateway Timeout", "An upstream server didn't answer in time."] };
  const c = Math.round(num(i.code, "code"));
  const [name, what] = C[c] || [c < 200 ? "Informational" : c < 300 ? "Success" : c < 400 ? "Redirect" : c < 500 ? "Client error" : "Server error", "A less common status code in that class."];
  return { result: { code: c, name, meaning: what }, display: info("code", `${c} ${name}`, what) };
});
tool("jwt_decode", "jwt_decode(token)", "Decode a JWT's header and payload (does not verify the signature).", { token: S }, ["token"], (i) => {
  const parts = String(need(i.token, "token")).trim().split(".");
  if (parts.length < 2) throw new Error("That isn't a JWT (expected header.payload.signature)");
  const dec = (p) => JSON.parse(Buffer.from(p.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8"));
  let header;
  let payload;
  try {
    header = dec(parts[0]);
    payload = dec(parts[1]);
  } catch {
    throw new Error("Couldn't decode that token");
  }
  const exp = payload.exp ? new Date(payload.exp * 1000).toISOString() : null;
  return { result: { header, payload, expires: exp, expired: exp ? Date.now() > payload.exp * 1000 : null, note: "Signature not verified" }, display: info("key", header.alg || "JWT", exp ? `${Date.now() > payload.exp * 1000 ? "Expired" : "Expires"} ${exp.slice(0, 16).replace("T", " ")} UTC` : "No expiry", [], { pre: JSON.stringify(payload, null, 2).slice(0, 2000), mono: true }) };
});
tool("ip_subnet", "ip_subnet(cidr)", "IPv4 subnet details for a CIDR like 192.168.1.0/24.", { cidr: S }, ["cidr"], (i) => {
  const m = String(need(i.cidr, "cidr")).trim().match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)\/(\d+)$/);
  if (!m || m.slice(1, 5).some((x) => Number(x) > 255) || Number(m[5]) > 32) throw new Error("Give a CIDR like 192.168.1.0/24");
  const ip = m.slice(1, 5).reduce((a, x) => a * 256 + Number(x), 0);
  const bits = Number(m[5]);
  const size = 2 ** (32 - bits);
  const net = Math.floor(ip / size) * size;
  const toIp = (n) => [24, 16, 8, 0].map((s) => Math.floor(n / 2 ** s) % 256).join(".");
  const mask = toIp(2 ** 32 - size);
  const hosts = bits >= 31 ? size : size - 2;
  const r = { network: toIp(net), broadcast: toIp(net + size - 1), netmask: mask, first_host: toIp(bits >= 31 ? net : net + 1), last_host: toIp(bits >= 31 ? net + size - 1 : net + size - 2), usable_hosts: hosts };
  return { result: r, display: info("globe", `${r.network}/${bits}`, `${hosts.toLocaleString()} usable hosts`, [["Netmask", mask], ["First host", r.first_host], ["Last host", r.last_host], ["Broadcast", r.broadcast]], { mono: true }) };
});

export const MORE_TOOLS = T;
export const MORE_NAMES = T.map((t) => t.name);
export const moreHandlers = Object.fromEntries(T.map((t) => [t.name, async (input, ctx) => t.run(input || {}, ctx)]));
// Claude tool definitions.
export const MORE_TOOL_DEFS = T.map((t) => ({ name: t.name, description: t.description, input_schema: { type: "object", properties: t.params, required: t.required } }));
// Short labels for the chat's step list.
const SHORT = { generate_password: "Password", password_strength: "Password check", uuid: "UUID", word_count: "Word count", change_case: "Changed case", slugify: "URL slug",
  base64: "Base64", url_encode: "URL encoding", hash_text: "Hash", json_format: "JSON check", regex_test: "Regex test", text_diff: "Text comparison", lorem_ipsum: "Placeholder text",
  sort_lines: "Sorted lines", extract_contacts: "Contacts found", random_number: "Random number", roll_dice: "Dice roll", flip_coin: "Coin flip", pick_random: "Random pick",
  statistics: "Statistics", gcd_lcm: "GCD & LCM", prime_factors: "Prime factors", number_to_words: "Number in words", roman_numerals: "Roman numerals", number_base: "Base conversion",
  percentage_change: "Percentage change", discount: "Discount", tax: "Tax", tip_split: "Bill split", loan_emi: "Loan EMI", compound_interest: "Savings growth", fuel_cost: "Fuel cost",
  pace: "Pace", bmi: "BMI", bmr: "Daily calories", age: "Age", date_diff: "Days between", add_to_date: "Date maths", day_of_week: "Day of the week", countdown: "Countdown",
  unix_time: "Unix time", timezone_convert: "Time zones", color_convert: "Color codes", public_holidays: "Public holidays", country_info: "Country facts", crypto_price: "Crypto price",
  book_search: "Books", find_words: "Words", air_quality: "Air quality", sunrise_sunset: "Sunrise & sunset", recent_earthquakes: "Earthquakes", random_joke: "Joke", quote: "Quote",
  github_repo: "GitHub repo", package_info: "Package", remember: "Saved to memory", recall: "Memory", forget: "Forgot", set_timer: "Timer", save_note: "Note saved", list_notes: "Notes",
  cron_explain: "Cron schedule", http_status: "HTTP status", jwt_decode: "JWT", ip_subnet: "Subnet" };
export const MORE_LABELS = Object.fromEntries(T.map((t) => [t.name, SHORT[t.name] || t.name.replace(/_/g, " ")]));

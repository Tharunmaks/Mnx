// More everyday tools for Mnx. None of them need an API key:
//   calculate        exact math (offline)
//   convert_units    length, weight, temperature, volume, speed, area, data, time (offline)
//   convert_currency live exchange rates (open.er-api.com)
//   get_time         the time anywhere (Open-Meteo geocoding + the device clock)
//   wikipedia        a topic's summary (Wikipedia)
//   define_word      meanings, examples and synonyms (dictionaryapi.dev)
//   translate        text into another language (MyMemory)
//   create_qr_code   a QR code image for a link or text (goqr.me)

const UA = "Mnx/1.0 (personal assistant)";

async function getJson(url) {
  const res = await fetch(url, { headers: { "user-agent": UA, accept: "application/json" }, signal: AbortSignal.timeout(12000) });
  if (!res.ok) throw new Error(`${new URL(url).host} returned HTTP ${res.status}`);
  return res.json();
}

const text = (v, max = 500) => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : undefined);
const round = (x, digits = 10) => Number.parseFloat(Number(x).toPrecision(digits));
const pretty = (x) => (Math.abs(x) >= 1e15 || (x !== 0 && Math.abs(x) < 1e-9) ? x.toExponential(8) : round(x, 12).toLocaleString("en-US", { maximumFractionDigits: 10 }));

/* ───────────── calculate ───────────── */

const FUNCS = {
  sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, floor: Math.floor, ceil: Math.ceil, exp: Math.exp,
  ln: Math.log, log: Math.log10, log10: Math.log10, log2: Math.log2,
  sin: (d) => Math.sin((d * Math.PI) / 180), cos: (d) => Math.cos((d * Math.PI) / 180), tan: (d) => Math.tan((d * Math.PI) / 180),
  asin: (x) => (Math.asin(x) * 180) / Math.PI, acos: (x) => (Math.acos(x) * 180) / Math.PI, atan: (x) => (Math.atan(x) * 180) / Math.PI,
  round: (x, d = 0) => Math.round(x * 10 ** d) / 10 ** d, min: Math.min, max: Math.max, pow: Math.pow,
};
const CONSTS = { pi: Math.PI, e: Math.E, tau: 2 * Math.PI };

function factorial(n) {
  if (!Number.isInteger(n) || n < 0) throw new Error("Factorial needs a whole number ≥ 0");
  if (n > 170) return Infinity;
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

// A small, safe expression parser (no eval). Supports + - * / ^ mod, n!,
// percentages ("15% of 240", "200 + 10%"), parentheses, implicit
// multiplication ("2pi", "3(4+1)") and the functions above (trig in degrees).
export function calculate(expression) {
  let src = String(expression)
    .toLowerCase()
    .replace(/×/g, "*")
    .replace(/(?<=[\d)]\s*)x(?=\s*[\d(.])/g, "*") // "3 x 4", but not "max("
    .replace(/÷/g, "/")
    .replace(/−/g, "-")
    .replace(/\*\*/g, "^")
    .replace(/(\d),(?=\d{3}\b)/g, "$1")
    .replace(/\bof\b/g, "*")
    .replace(/\bmod\b/g, "@");
  const tokens = src.match(/\d*\.?\d+(?:e[+-]?\d+)?|[a-z_]+\d*|[-+*/^@%!(),]|\S/g) || [];
  let i = 0;
  const peek = () => tokens[i];
  const next = () => tokens[i++];
  const expect = (t) => {
    if (next() !== t) throw new Error(`Expected "${t}"`);
  };

  function expr() {
    let v = term();
    while (peek() === "+" || peek() === "-") {
      const op = next();
      // "200 + 10%" means 200 + 10% of 200.
      const start = i;
      let r = term();
      if (tokens[i - 1] === "%" && tokens.slice(start, i).filter((t) => t === "%").length === 1 && /^[\d.]+$/.test(tokens[start])) r *= v;
      v = op === "+" ? v + r : v - r;
    }
    return v;
  }
  function term() {
    let v = unary();
    for (;;) {
      const t = peek();
      if (t === "*" || t === "/" || t === "@") {
        next();
        const r = unary();
        v = t === "*" ? v * r : t === "/" ? v / r : ((v % r) + r) % r;
      } else if (t === "(" || (t && /^[a-z]/.test(t))) v *= power(); // implicit: 2pi, 3(4)
      else return v;
    }
  }
  // Signs bind looser than powers: -3^2 = -9, 2^-1 = 0.5.
  function unary() {
    if (peek() === "-") return next(), -unary();
    if (peek() === "+") return next(), unary();
    return power();
  }
  function power() {
    const base = postfix();
    if (peek() === "^") {
      next();
      return base ** unary();
    }
    return base;
  }
  function postfix() {
    let v = primary();
    for (;;) {
      if (peek() === "!") next(), (v = factorial(v));
      else if (peek() === "%") next(), (v /= 100);
      else return v;
    }
  }
  function primary() {
    const t = next();
    if (t === undefined) throw new Error("The expression ended early");
    if (/^[\d.]/.test(t)) return Number(t);
    if (t === "(") {
      const v = expr();
      expect(")");
      return v;
    }
    if (t in CONSTS) return CONSTS[t];
    if (t in FUNCS) {
      expect("(");
      const args = [expr()];
      while (peek() === ",") next(), args.push(expr());
      expect(")");
      return FUNCS[t](...args);
    }
    throw new Error(`Unknown "${t}"`);
  }

  const value = expr();
  if (i < tokens.length) throw new Error(`Unexpected "${tokens[i]}"`);
  if (!Number.isFinite(value)) throw new Error(Number.isNaN(value) ? "The result isn't a number" : "The result is infinite");
  return value;
}

/* ───────────── convert_units ───────────── */

const UNITS = {
  length: { m: 1, meter: 1, metre: 1, km: 1000, kilometer: 1000, cm: 0.01, centimeter: 0.01, mm: 0.001, millimeter: 0.001, um: 1e-6, micrometer: 1e-6, nm: 1e-9,
    mi: 1609.344, mile: 1609.344, yd: 0.9144, yard: 0.9144, ft: 0.3048, foot: 0.3048, feet: 0.3048, in: 0.0254, inch: 0.0254, nmi: 1852, "nautical mile": 1852, ly: 9.4607e15, "light year": 9.4607e15 },
  mass: { kg: 1, kilogram: 1, g: 0.001, gram: 0.001, mg: 1e-6, milligram: 1e-6, t: 1000, tonne: 1000, ton: 1000, lb: 0.45359237, lbs: 0.45359237, pound: 0.45359237,
    oz: 0.028349523125, ounce: 0.028349523125, st: 6.35029318, stone: 6.35029318, ct: 0.0002, carat: 0.0002 },
  volume: { l: 1, liter: 1, litre: 1, ml: 0.001, milliliter: 0.001, millilitre: 0.001, cl: 0.01, dl: 0.1, m3: 1000, "cubic meter": 1000, gal: 3.785411784, gallon: 3.785411784,
    "uk gallon": 4.54609, qt: 0.946352946, quart: 0.946352946, pt: 0.473176473, pint: 0.473176473, cup: 0.2365882365, "fl oz": 0.0295735295625, "fluid ounce": 0.0295735295625,
    tbsp: 0.01478676478125, tablespoon: 0.01478676478125, tsp: 0.00492892159375, teaspoon: 0.00492892159375 },
  speed: { "m/s": 1, "km/h": 1 / 3.6, kph: 1 / 3.6, kmh: 1 / 3.6, mph: 0.44704, knot: 0.514444, kn: 0.514444, "ft/s": 0.3048, mach: 343 },
  area: { m2: 1, "square meter": 1, km2: 1e6, "square kilometer": 1e6, cm2: 1e-4, ft2: 0.09290304, "square foot": 0.09290304, "square feet": 0.09290304,
    "sq ft": 0.09290304, in2: 0.00064516, acre: 4046.8564224, ha: 10000, hectare: 10000, mi2: 2589988.110336, "square mile": 2589988.110336 },
  data: { b: 1, byte: 1, kb: 1e3, kilobyte: 1e3, mb: 1e6, megabyte: 1e6, gb: 1e9, gigabyte: 1e9, tb: 1e12, terabyte: 1e12, pb: 1e15,
    kib: 1024, mib: 1024 ** 2, gib: 1024 ** 3, tib: 1024 ** 4, bit: 0.125, kbit: 125, mbit: 125000, gbit: 1.25e8 },
  time: { s: 1, sec: 1, second: 1, ms: 0.001, millisecond: 0.001, min: 60, minute: 60, h: 3600, hr: 3600, hour: 3600, day: 86400, week: 604800,
    month: 2629746, year: 31556952, decade: 315569520, century: 3155695200 },
  energy: { j: 1, joule: 1, kj: 1000, cal: 4.184, calorie: 4.184, kcal: 4184, kilocalorie: 4184, wh: 3600, kwh: 3.6e6, btu: 1055.06 },
  pressure: { pa: 1, kpa: 1000, bar: 100000, psi: 6894.757, atm: 101325, mmhg: 133.322 },
};
const TEMPS = { c: "C", celsius: "C", "°c": "C", f: "F", fahrenheit: "F", "°f": "F", k: "K", kelvin: "K" };

function unitKey(u) {
  let k = String(u || "").toLowerCase().trim().replace(/\s+/g, " ").replace(/²/g, "2").replace(/³/g, "3").replace(/^square /, "square ").replace(/^degrees? /, "");
  if (TEMPS[k]) return { temp: TEMPS[k] };
  for (const [dim, table] of Object.entries(UNITS)) {
    for (const cand of [k, k.replace(/(es|s)$/, ""), k.replace(/s$/, "")]) if (cand in table) return { dim, factor: table[cand], name: cand };
  }
  return null;
}

export function convertUnits(value, from, to) {
  const v = Number(value);
  if (!Number.isFinite(v)) throw new Error("'value' must be a number");
  const a = unitKey(from);
  const b = unitKey(to);
  if (!a) throw new Error(`Unknown unit "${from}"`);
  if (!b) throw new Error(`Unknown unit "${to}"`);
  if (a.temp || b.temp) {
    if (!(a.temp && b.temp)) throw new Error(`Can't convert ${from} to ${to}`);
    const c = { C: v, F: ((v - 32) * 5) / 9, K: v - 273.15 }[a.temp];
    return round({ C: c, F: (c * 9) / 5 + 32, K: c + 273.15 }[b.temp]);
  }
  if (a.dim !== b.dim) throw new Error(`Can't convert ${a.dim} (${from}) to ${b.dim} (${to})`);
  return round((v * a.factor) / b.factor);
}

/* ───────────── languages for translate ───────────── */

const LANGS = {
  english: "en", french: "fr", spanish: "es", german: "de", italian: "it", portuguese: "pt", dutch: "nl", russian: "ru", ukrainian: "uk", polish: "pl",
  turkish: "tr", arabic: "ar", hebrew: "he", persian: "fa", hindi: "hi", tamil: "ta", telugu: "te", kannada: "kn", malayalam: "ml", marathi: "mr",
  bengali: "bn", gujarati: "gu", punjabi: "pa", urdu: "ur", chinese: "zh-CN", mandarin: "zh-CN", japanese: "ja", korean: "ko", thai: "th",
  vietnamese: "vi", indonesian: "id", malay: "ms", filipino: "tl", tagalog: "tl", swahili: "sw", greek: "el", swedish: "sv", norwegian: "no",
  danish: "da", finnish: "fi", czech: "cs", hungarian: "hu", romanian: "ro",
};
const langCode = (l) => {
  const k = String(l || "").toLowerCase().trim();
  return LANGS[k] || (/^[a-z]{2}(-[a-z]{2})?$/i.test(k) ? k : null);
};

/* ───────────── cards ───────────── */

// The card shown in the chat for a tool's result (also used by the test mocks).
export function extraDisplay(name, r, extra = {}) {
  switch (name) {
    case "calculate":
      return { kind: "info", icon: "calc", title: pretty(r.result), subtitle: r.expression };
    case "convert_units":
      return { kind: "info", icon: "scale", title: `${pretty(r.result)} ${r.to}`, subtitle: `${pretty(r.value)} ${r.from}` };
    case "convert_currency":
      return { kind: "info", icon: "money", title: `${Number(r.converted).toLocaleString("en-US")} ${r.to}`, subtitle: `${Number(r.amount).toLocaleString("en-US")} ${r.from}`,
        rows: [["Rate", `1 ${r.from} = ${r.rate} ${r.to}`], ...(r.updated ? [["Updated", String(r.updated).replace(/ \+0000$/, " UTC")]] : [])] };
    case "get_time":
      return { kind: "info", icon: "clock", title: r.time, subtitle: r.location, rows: [["Date", r.date], ["Time zone", `${r.timezone} (${r.utc_offset})`]] };
    case "wikipedia":
      return { kind: "info", icon: "book", title: r.title, subtitle: r.description || "Wikipedia", body: String(r.summary || "").slice(0, 600), image: extra.image, link: r.url, linkText: "Read on Wikipedia" };
    case "define_word":
      return { kind: "info", icon: "book", title: r.word, subtitle: r.phonetic || "Dictionary",
        rows: (r.meanings || []).flatMap((m) => (m.definitions || []).map((d) => [m.part_of_speech, d.definition])).slice(0, 5) };
    case "translate":
      return { kind: "info", icon: "lang", title: r.translation, subtitle: `${r.from} → ${r.to}`, rows: [["Original", r.text]] };
    default:
      return undefined;
  }
}

/* ───────────── handlers ───────────── */

export const extraHandlers = {
  async calculate(input) {
    const expression = text(input.expression, 500);
    if (!expression) throw new Error("'expression' is required");
    const value = calculate(expression);
    const result = { expression, result: round(value, 12) };
    return { result, display: extraDisplay("calculate", result) };
  },

  async convert_units(input) {
    const value = Number(input.value);
    const out = convertUnits(value, input.from, input.to);
    const result = { value, from: input.from, to: input.to, result: out };
    return { result, display: extraDisplay("convert_units", result) };
  },

  async convert_currency(input) {
    const amount = Number(input.amount ?? 1);
    const from = String(input.from || "").toUpperCase().trim();
    const to = String(input.to || "").toUpperCase().trim();
    if (!/^[A-Z]{3}$/.test(from) || !/^[A-Z]{3}$/.test(to)) throw new Error("'from' and 'to' must be 3-letter currency codes like USD, EUR, INR");
    if (!Number.isFinite(amount)) throw new Error("'amount' must be a number");
    const data = await getJson(`https://open.er-api.com/v6/latest/${from}`);
    const rate = data.rates?.[to];
    if (data.result !== "success" || !rate) throw new Error(`No exchange rate for ${from} → ${to}`);
    const converted = Math.round(amount * rate * 100) / 100;
    const result = { amount, from, to, rate: round(rate, 6), converted, updated: data.time_last_update_utc };
    return { result, display: extraDisplay("convert_currency", result) };
  },

  async get_time(input, ctx) {
    const name = text(input.location, 200);
    let place = null;
    let zone;
    if (name) {
      const g = await getJson(`https://geocoding-api.open-meteo.com/v1/search?${new URLSearchParams({ name: name.split(",")[0], count: "1", format: "json" })}`);
      const hit = g.results?.[0];
      if (!hit?.timezone) throw new Error(`Couldn't find a place called "${name}"`);
      place = [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ");
      zone = hit.timezone;
    } else if (ctx?.location) {
      const q = new URLSearchParams({ latitude: ctx.location.latitude, longitude: ctx.location.longitude, current: "temperature_2m", timezone: "auto" });
      zone = (await getJson(`https://api.open-meteo.com/v1/forecast?${q}`)).timezone;
      place = ctx.location.name || "your location";
    } else {
      zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      place = "this device";
    }
    const now = new Date();
    const fmt = (o) => new Intl.DateTimeFormat("en-GB", { timeZone: zone, ...o }).format(now);
    const time = fmt({ hour: "2-digit", minute: "2-digit", hour12: false });
    const date = fmt({ weekday: "long", day: "numeric", month: "long", year: "numeric" });
    const offset = new Intl.DateTimeFormat("en-US", { timeZone: zone, timeZoneName: "longOffset" }).formatToParts(now).find((p) => p.type === "timeZoneName")?.value.replace("GMT", "UTC") || "";
    const result = { location: place, timezone: zone, time, date, utc_offset: offset === "UTC" ? "UTC+00:00" : offset };
    return { result, display: extraDisplay("get_time", result) };
  },

  async wikipedia(input) {
    const topic = text(input.topic, 200);
    if (!topic) throw new Error("'topic' is required");
    const q = new URLSearchParams({ action: "query", list: "search", srsearch: topic, format: "json", srlimit: "1", origin: "*" });
    const title = (await getJson(`https://en.wikipedia.org/w/api.php?${q}`)).query?.search?.[0]?.title;
    if (!title) throw new Error(`No Wikipedia article found for "${topic}"`);
    const s = await getJson(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, "_"))}`);
    const url = s.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`;
    const result = { title: s.title || title, description: s.description, summary: String(s.extract || "").slice(0, 1500), url };
    return { result, display: extraDisplay("wikipedia", result, { image: s.thumbnail?.source }) };
  },

  async define_word(input) {
    const word = text(input.word, 60);
    if (!word) throw new Error("'word' is required");
    let data;
    try {
      data = await getJson(`https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word.toLowerCase())}`);
    } catch (err) {
      if (/HTTP 404/.test(err.message)) throw new Error(`No dictionary entry for "${word}"`);
      throw err;
    }
    const e = data[0];
    const meanings = (e.meanings || []).slice(0, 3).map((m) => ({
      part_of_speech: m.partOfSpeech,
      definitions: (m.definitions || []).slice(0, 2).map((d) => ({ definition: d.definition, example: d.example })),
      synonyms: (m.synonyms || []).slice(0, 5),
    }));
    const phonetic = e.phonetic || e.phonetics?.find((p) => p.text)?.text;
    const result = { word: e.word, phonetic, meanings };
    return { result, display: extraDisplay("define_word", result) };
  },

  async translate(input) {
    const t = text(input.text, 480);
    if (!t) throw new Error("'text' is required");
    const to = langCode(input.to);
    const from = input.from ? langCode(input.from) : "en";
    if (!to) throw new Error(`Unknown target language "${input.to}"`);
    if (!from) throw new Error(`Unknown source language "${input.from}"`);
    const data = await getJson(`https://api.mymemory.translated.net/get?${new URLSearchParams({ q: t, langpair: `${from}|${to}` })}`);
    const translated = data.responseData?.translatedText;
    if (!translated || data.responseStatus >= 400) throw new Error(data.responseDetails || "Translation failed");
    const result = { text: t, from, to, translation: translated };
    return { result, display: extraDisplay("translate", result) };
  },

  async create_qr_code(input) {
    const data = text(input.text, 900);
    if (!data) throw new Error("'text' is required");
    const url = `https://api.qrserver.com/v1/create-qr-code/?${new URLSearchParams({ size: "480x480", margin: "12", data })}`;
    return { result: { created: true, shown_to_user: true, text: data }, display: { kind: "image", prompt: `QR code: ${data}`, url, width: 480, height: 480 } };
  },
};

/* ───────────── tool definitions (Claude format) ───────────── */

const s = { type: "string" };
const def = (name, description, properties, required = []) => ({ name, description, input_schema: { type: "object", properties, required } });

export const EXTRA_TOOLS = [
  def("calculate", "Exact arithmetic and math: + - * / ^, mod, n!, percentages ('15% of 240'), sqrt, sin/cos/tan in degrees, log, ln, pi, e. Use for any calculation instead of doing it in your head.", { expression: s }, ["expression"]),
  def("convert_units", "Convert between units of length, weight, temperature, volume, speed, area, data size, time, energy or pressure.", { value: { type: "number" }, from: s, to: s }, ["value", "from", "to"]),
  def("convert_currency", "Convert money between currencies at today's exchange rate. Use 3-letter codes (USD, EUR, INR, GBP, JPY…).", { amount: { type: "number" }, from: s, to: s }, ["from", "to"]),
  def("get_time", "The current local time and date in a city or country (or the user's location if omitted).", { location: s }),
  def("wikipedia", "A short encyclopedia summary of a person, place, thing or event from Wikipedia.", { topic: s }, ["topic"]),
  def("define_word", "Dictionary meanings, pronunciation, examples and synonyms of an English word.", { word: s }, ["word"]),
  def("translate", "Translate text into another language. 'to' and 'from' are language names or codes (from defaults to English).", { text: s, to: s, from: s }, ["text", "to"]),
  def("create_qr_code", "Make a QR code image for a link, Wi-Fi details, contact or any text.", { text: s }, ["text"]),
];

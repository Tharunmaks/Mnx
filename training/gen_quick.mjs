// Training conversations for Mnx's 65 quick tools (tools-more.js), written
// in exactly the format the Python generator uses. Offline tools run for
// real, so every result is genuine; online tools get realistic stand-in
// results. Also teaches using remembered facts from the system prompt.
//
//   node training/gen_quick.mjs --train 20000 --test 200 [--out training/data] [--append]
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
process.env.MNX_MEMORY_FILE = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "mnx-gen-")), "memory.json");
const { MORE_TOOLS } = await import("../tools-more.js");
const P = JSON.parse(fs.readFileSync(path.join(here, "system_prompt.json"), "utf8"));
const arg = (n, d) => {
  const i = process.argv.indexOf(`--${n}`);
  return i > 0 ? process.argv[i + 1] : d;
};
const nTrain = Number(arg("train", 20000));
const nTest = Number(arg("test", 200));
const out = arg("out", path.join(here, "data"));
const append = process.argv.includes("--append");
const TOOL = Object.fromEntries(MORE_TOOLS.map((t) => [t.name, t]));

/* ───────────── deterministic randomness ───────────── */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let R = rng(11);
const rand = () => R();
const int = (a, b) => a + Math.floor(rand() * (b - a + 1));
const pick = (xs) => xs[Math.floor(rand() * xs.length)];
const chance = (p) => rand() < p;

/* ───────────── format helpers (must match generate.py / local.js) ───────────── */
// Python's json.dumps(ensure_ascii=False): ", " and ": " separators.
function pyJson(v) {
  if (Array.isArray(v)) return `[${v.map(pyJson).join(", ")}]`;
  if (v && typeof v === "object") return `{${Object.entries(v).filter(([, x]) => x !== undefined).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(", ")}}`;
  return JSON.stringify(v ?? null);
}
const NAMES = ["", "", "", "Tharun", "Priya", "Arjun", "Meera", "Sam", "Alex", "Fatima", "Kenji", "Lucas", "Ana", "Ravi", "Zara"];
function system(today, memories = []) {
  const name = pick(NAMES);
  const mem = memories.length ? `\n\nThings the user asked you to remember:\n${memories.map((m) => `- ${m}`).join("\n")}` : "";
  return { role: "system", content: `${P.stable_prompt}\n\nToday's date is ${today}.${name ? ` The user's name is ${name}.` : ""}${mem}` };
}
const think = (title, text) => `<think>**${title}**\n${text}</think>\n`;
const call = (name, args) => {
  if (!P.tools.includes(name)) throw new Error(`unknown tool ${name}`);
  return `<tool_call>\n${pyJson({ name, arguments: args })}\n</tool_call>`;
};
const response = (obj) => `<tool_response>\n${pyJson(obj)}\n</tool_response>`;

// Run a real tool as if today were `today` (so ages, countdowns, etc. match the prompt's date).
const RealDate = Date;
function runAt(today, name, args) {
  const base = new RealDate(`${today}T10:30:00`).getTime();
  class FakeDate extends RealDate {
    constructor(...a) {
      if (a.length) super(...a);
      else super(base);
    }
    static now() {
      return base;
    }
  }
  globalThis.Date = FakeDate;
  try {
    const r = TOOL[name].run(args, {});
    if (r instanceof Promise) throw new Error(`${name} is online; give a result`);
    return r.result;
  } finally {
    globalThis.Date = RealDate;
  }
}
const randomDay = () => {
  const d = new RealDate(Date.UTC(2025, 0, 1) + int(0, 640) * 864e5);
  return d.toISOString().slice(0, 10);
};
const n2 = (x) => Number(x).toLocaleString("en-US", { maximumFractionDigits: 2 });

/* ───────────── content ───────────── */
const TEXTS = ["The quick brown fox jumps over the lazy dog.", "Mnx runs on my phone and helps me plan my day, learn new things and write code.",
  "Climate change is one of the biggest challenges of our time. It affects weather, oceans and food. We can all help by saving energy.",
  "Dear team, the launch moves to Friday. Please finish testing by Thursday evening and send me your notes.",
  "Reading every day improves vocabulary, focus and empathy. Even ten pages a day adds up to many books a year.",
  "Water boils at 100 degrees Celsius at sea level. At higher altitudes it boils at lower temperatures because air pressure is lower."];
const TITLES = ["my first blog post", "10 Tips for Better Sleep!", "How to Learn Python in 30 Days", "Best Street Food in Chennai", "Café Menu: Summer Specials", "Why I Switched to Linux"];
const COUNTRIES = [
  { name: "India", official: "Republic of India", capital: "New Delhi", pop: 1417492000, area: 3287590, region: "Southern Asia, Asia", langs: ["Hindi", "English", "Tamil"], cur: ["Indian rupee (INR, ₹)"], code: "+91", flag: "🇮🇳", tz: ["UTC+05:30"] },
  { name: "Japan", official: "Japan", capital: "Tokyo", pop: 123210000, area: 377930, region: "Eastern Asia, Asia", langs: ["Japanese"], cur: ["Japanese yen (JPY, ¥)"], code: "+81", flag: "🇯🇵", tz: ["UTC+09:00"] },
  { name: "Brazil", official: "Federative Republic of Brazil", capital: "Brasília", pop: 212559000, area: 8515767, region: "South America, Americas", langs: ["Portuguese"], cur: ["Brazilian real (BRL, R$)"], code: "+55", flag: "🇧🇷", tz: ["UTC-05:00", "UTC-04:00", "UTC-03:00"] },
  { name: "Germany", official: "Federal Republic of Germany", capital: "Berlin", pop: 83491000, area: 357114, region: "Western Europe, Europe", langs: ["German"], cur: ["Euro (EUR, €)"], code: "+49", flag: "🇩🇪", tz: ["UTC+01:00"] },
  { name: "Kenya", official: "Republic of Kenya", capital: "Nairobi", pop: 53771000, area: 580367, region: "Eastern Africa, Africa", langs: ["English", "Swahili"], cur: ["Kenyan shilling (KES, Sh)"], code: "+254", flag: "🇰🇪", tz: ["UTC+03:00"] },
  { name: "Australia", official: "Commonwealth of Australia", capital: "Canberra", pop: 26014000, area: 7692024, region: "Australia and New Zealand, Oceania", langs: ["English"], cur: ["Australian dollar (AUD, $)"], code: "+61", flag: "🇦🇺", tz: ["UTC+08:00", "UTC+10:00"] },
  { name: "Canada", official: "Canada", capital: "Ottawa", pop: 38005000, area: 9984670, region: "North America, Americas", langs: ["English", "French"], cur: ["Canadian dollar (CAD, $)"], code: "+1", flag: "🇨🇦", tz: ["UTC-08:00", "UTC-05:00"] },
  { name: "Egypt", official: "Arab Republic of Egypt", capital: "Cairo", pop: 102334000, area: 1002450, region: "Northern Africa, Africa", langs: ["Arabic"], cur: ["Egyptian pound (EGP, £)"], code: "+20", flag: "🇪🇬", tz: ["UTC+02:00"] },
  { name: "France", official: "French Republic", capital: "Paris", pop: 67391000, area: 551695, region: "Western Europe, Europe", langs: ["French"], cur: ["Euro (EUR, €)"], code: "+33", flag: "🇫🇷", tz: ["UTC+01:00"] },
  { name: "Mexico", official: "United Mexican States", capital: "Mexico City", pop: 128932000, area: 1964375, region: "North America, Americas", langs: ["Spanish"], cur: ["Mexican peso (MXN, $)"], code: "+52", flag: "🇲🇽", tz: ["UTC-06:00"] },
  { name: "Indonesia", official: "Republic of Indonesia", capital: "Jakarta", pop: 273524000, area: 1904569, region: "South-Eastern Asia, Asia", langs: ["Indonesian"], cur: ["Indonesian rupiah (IDR, Rp)"], code: "+62", flag: "🇮🇩", tz: ["UTC+07:00", "UTC+09:00"] },
  { name: "Sri Lanka", official: "Democratic Socialist Republic of Sri Lanka", capital: "Sri Jayawardenepura Kotte", pop: 21919000, area: 65610, region: "Southern Asia, Asia", langs: ["Sinhala", "Tamil"], cur: ["Sri Lankan rupee (LKR, Rs)"], code: "+94", flag: "🇱🇰", tz: ["UTC+05:30"] },
  { name: "Italy", official: "Italian Republic", capital: "Rome", pop: 59554000, area: 301336, region: "Southern Europe, Europe", langs: ["Italian"], cur: ["Euro (EUR, €)"], code: "+39", flag: "🇮🇹", tz: ["UTC+01:00"] },
  { name: "South Korea", official: "Republic of Korea", capital: "Seoul", pop: 51780000, area: 100210, region: "Eastern Asia, Asia", langs: ["Korean"], cur: ["South Korean won (KRW, ₩)"], code: "+82", flag: "🇰🇷", tz: ["UTC+09:00"] },
  { name: "Nigeria", official: "Federal Republic of Nigeria", capital: "Abuja", pop: 206140000, area: 923768, region: "Western Africa, Africa", langs: ["English"], cur: ["Nigerian naira (NGN, ₦)"], code: "+234", flag: "🇳🇬", tz: ["UTC+01:00"] },
  { name: "Argentina", official: "Argentine Republic", capital: "Buenos Aires", pop: 45377000, area: 2780400, region: "South America, Americas", langs: ["Spanish"], cur: ["Argentine peso (ARS, $)"], code: "+54", flag: "🇦🇷", tz: ["UTC-03:00"] },
  { name: "Norway", official: "Kingdom of Norway", capital: "Oslo", pop: 5379000, area: 323802, region: "Northern Europe, Europe", langs: ["Norwegian"], cur: ["Norwegian krone (NOK, kr)"], code: "+47", flag: "🇳🇴", tz: ["UTC+01:00"] },
  { name: "Vietnam", official: "Socialist Republic of Vietnam", capital: "Hanoi", pop: 97339000, area: 331212, region: "South-Eastern Asia, Asia", langs: ["Vietnamese"], cur: ["Vietnamese đồng (VND, ₫)"], code: "+84", flag: "🇻🇳", tz: ["UTC+07:00"] },
];
const COINS = [["bitcoin", "BTC", 60000, 110000], ["ethereum", "ETH", 2200, 4200], ["solana", "SOL", 90, 260], ["dogecoin", "DOGE", 0.08, 0.4], ["cardano", "ADA", 0.3, 1.1], ["ripple", "XRP", 0.4, 3]];
const BOOKS = {
  "atomic habits": [["Atomic Habits", "James Clear", 2018, 320]], "harry potter": [["Harry Potter and the Philosopher's Stone", "J. K. Rowling", 1997, 223], ["Harry Potter and the Chamber of Secrets", "J. K. Rowling", 1998, 251]],
  "books by agatha christie": [["Murder on the Orient Express", "Agatha Christie", 1934, 256], ["And Then There Were None", "Agatha Christie", 1939, 272], ["The Murder of Roger Ackroyd", "Agatha Christie", 1926, 312]],
  "the alchemist": [["The Alchemist", "Paulo Coelho", 1988, 197]], "books about stoicism": [["Meditations", "Marcus Aurelius", 180, 254], ["Letters from a Stoic", "Seneca", 65, 254], ["The Obstacle Is the Way", "Ryan Holiday", 2014, 201]],
  "wings of fire kalam": [["Wings of Fire", "A. P. J. Abdul Kalam, Arun Tiwari", 1999, 180]], "sapiens": [["Sapiens: A Brief History of Humankind", "Yuval Noah Harari", 2011, 443]],
  "books on python programming": [["Automate the Boring Stuff with Python", "Al Sweigart", 2015, 504], ["Python Crash Course", "Eric Matthes", 2015, 544], ["Fluent Python", "Luciano Ramalho", 2015, 792]],
};
const SYNONYMS = { happy: ["glad", "cheerful", "joyful", "content", "delighted"], big: ["large", "huge", "enormous", "vast", "giant"], fast: ["quick", "rapid", "swift", "speedy", "brisk"],
  smart: ["clever", "intelligent", "bright", "sharp", "wise"], beautiful: ["lovely", "pretty", "gorgeous", "stunning", "attractive"], important: ["significant", "crucial", "vital", "essential", "key"],
  tired: ["weary", "exhausted", "sleepy", "drained", "fatigued"], difficult: ["hard", "tough", "challenging", "demanding", "tricky"] };
const RHYMES = { cat: ["hat", "bat", "mat", "sat", "flat", "that"], light: ["night", "bright", "flight", "sight", "right", "might"], day: ["way", "say", "play", "stay", "may", "gray"],
  love: ["above", "dove", "glove", "of", "shove"], moon: ["soon", "tune", "june", "noon", "spoon", "balloon"], heart: ["start", "art", "part", "apart", "smart"] };
const ANTONYMS = { happy: ["sad", "unhappy", "miserable"], hot: ["cold", "cool", "chilly"], fast: ["slow", "sluggish"], strong: ["weak", "feeble"], early: ["late", "tardy"], begin: ["end", "finish", "stop"] };
const JOKES = ["Why don't skeletons fight each other? They don't have the guts.", "I'm reading a book about anti-gravity. It's impossible to put down.", "Why did the scarecrow win an award? He was outstanding in his field.",
  "What do you call a fake noodle? An impasta.", "Why don't eggs tell jokes? They'd crack each other up.", "I used to hate facial hair, but then it grew on me.", "What do you call a bear with no teeth? A gummy bear.",
  "Why did the math book look sad? It had too many problems.", "How does a penguin build its house? Igloos it together."];
const QUOTES = [["The only way to do great work is to love what you do.", "Steve Jobs"], ["It always seems impossible until it's done.", "Nelson Mandela"], ["Dream is not that which you see while sleeping, it is something that does not let you sleep.", "A. P. J. Abdul Kalam"],
  ["In the middle of every difficulty lies opportunity.", "Albert Einstein"], ["Be the change that you wish to see in the world.", "Mahatma Gandhi"], ["Well done is better than well said.", "Benjamin Franklin"], ["What we think, we become.", "Buddha"]];
const REPOS = [["facebook/react", "The library for web and native user interfaces.", 230000, 47000, "JavaScript", "MIT"], ["torvalds/linux", "Linux kernel source tree", 185000, 54000, "C", "NOASSERTION"],
  ["python/cpython", "The Python programming language", 65000, 31000, "Python", "NOASSERTION"], ["ggml-org/llama.cpp", "LLM inference in C/C++", 80000, 12000, "C++", "MIT"], ["microsoft/vscode", "Visual Studio Code", 170000, 31000, "TypeScript", "MIT"],
  ["tensorflow/tensorflow", "An Open Source Machine Learning Framework for Everyone", 187000, 74000, "C++", "Apache-2.0"], ["vercel/next.js", "The React Framework", 130000, 28000, "JavaScript", "MIT"]];
const PACKAGES = [["npm", "express", "5.1.0", "Fast, unopinionated, minimalist web framework"], ["npm", "react", "19.1.0", "React is a JavaScript library for building user interfaces."], ["npm", "lodash", "4.17.21", "Lodash modular utilities."],
  ["npm", "axios", "1.9.0", "Promise based HTTP client for the browser and node.js"], ["pypi", "requests", "2.32.3", "Python HTTP for Humans."], ["pypi", "numpy", "2.2.6", "Fundamental package for array computing in Python"],
  ["pypi", "pandas", "2.2.3", "Powerful data structures for data analysis, time series, and statistics"], ["pypi", "flask", "3.1.1", "A simple framework for building complex web applications."]];
const PLACES = [["Delhi", "India", 28.6], ["Mumbai", "India", 19.1], ["Chennai", "India", 13.1], ["Beijing", "China", 39.9], ["London", "United Kingdom", 51.5], ["Paris", "France", 48.9], ["Los Angeles", "United States", 34.1],
  ["Lagos", "Nigeria", 6.5], ["Sydney", "Australia", -33.9], ["Tokyo", "Japan", 35.7], ["Cairo", "Egypt", 30.0], ["Reykjavik", "Iceland", 64.1], ["Singapore", "Singapore", 1.3], ["Dhaka", "Bangladesh", 23.8]];
const HOLIDAYS = {
  US: (y) => [[`${y}-01-01`, "New Year's Day"], [nthWeekday(y, 0, 1, 3), "Martin Luther King, Jr. Day"], [nthWeekday(y, 1, 1, 3), "Presidents' Day (Washington's Birthday)"], [lastWeekday(y, 4, 1), "Memorial Day"],
    [`${y}-06-19`, "Juneteenth National Independence Day"], [`${y}-07-04`, "Independence Day"], [nthWeekday(y, 8, 1, 1), "Labor Day"], [nthWeekday(y, 9, 1, 2), "Columbus Day"], [`${y}-11-11`, "Veterans Day"], [nthWeekday(y, 10, 4, 4), "Thanksgiving Day"], [`${y}-12-25`, "Christmas Day"]],
  FR: (y) => [[`${y}-01-01`, "New Year's Day (Jour de l'an)"], [addDays(easter(y), 1), "Easter Monday (Lundi de Pâques)"], [`${y}-05-01`, "Labour Day (Fête du Travail)"], [`${y}-05-08`, "Victory in Europe Day (Victoire 1945)"], [addDays(easter(y), 39), "Ascension Day (Ascension)"],
    [addDays(easter(y), 50), "Whit Monday (Lundi de Pentecôte)"], [`${y}-07-14`, "Bastille Day (Fête nationale)"], [`${y}-08-15`, "Assumption Day (Assomption)"], [`${y}-11-01`, "All Saints' Day (Toussaint)"], [`${y}-11-11`, "Armistice Day (Armistice 1918)"], [`${y}-12-25`, "Christmas Day (Noël)"]],
  DE: (y) => [[`${y}-01-01`, "New Year's Day (Neujahr)"], [addDays(easter(y), -2), "Good Friday (Karfreitag)"], [addDays(easter(y), 1), "Easter Monday (Ostermontag)"], [`${y}-05-01`, "Labour Day (Tag der Arbeit)"], [addDays(easter(y), 39), "Ascension Day (Christi Himmelfahrt)"],
    [addDays(easter(y), 50), "Whit Monday (Pfingstmontag)"], [`${y}-10-03`, "German Unity Day (Tag der Deutschen Einheit)"], [`${y}-12-25`, "Christmas Day (Erster Weihnachtstag)"], [`${y}-12-26`, "St. Stephen's Day (Zweiter Weihnachtstag)"]],
};
const COUNTRY_CODES = { US: "the United States", FR: "France", DE: "Germany" };
function easter(y) {
  const a = y % 19, b = Math.floor(y / 100), c = y % 100, d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25), g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30, i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31), day = ((h + l - 7 * m + 114) % 31) + 1;
  return `${y}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}
function addDays(iso, n) {
  const d = new RealDate(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}
function nthWeekday(y, month, weekday, n) {
  const d = new RealDate(Date.UTC(y, month, 1));
  const shift = (weekday - d.getUTCDay() + 7) % 7;
  return new RealDate(Date.UTC(y, month, 1 + shift + 7 * (n - 1))).toISOString().slice(0, 10);
}
function lastWeekday(y, month, weekday) {
  const d = new RealDate(Date.UTC(y, month + 1, 0));
  return new RealDate(Date.UTC(y, month, d.getUTCDate() - ((d.getUTCDay() - weekday + 7) % 7))).toISOString().slice(0, 10);
}
const MEMORY_FACTS = [["I'm vegetarian", "food", "Since you're vegetarian"], ["My favourite color is teal", "color", "Teal, your favourite color"], ["I live in Chennai", "city", "Since you live in Chennai"],
  ["My sister's birthday is on 12 March", "birthday", "Your sister's birthday is on 12 March"], ["I'm learning Python", "learning", "Since you're learning Python"], ["My dog is called Bruno", "dog", "Bruno"],
  ["I work night shifts", "work", "Since you work night shifts"], ["I prefer short answers", "style", ""], ["My car does 15 km per litre", "car", "Your car does 15 km per litre"], ["I'm allergic to peanuts", "allergy", "Since you're allergic to peanuts"]];

/* ───────────── tasks: one per tool ───────────── */
// Each returns { user, args, title, why, answer(result), expect } (plus result for online tools).
const TASKS = {
  generate_password: () => {
    const len = pick([12, 14, 16, 20, 24, 32]);
    const sym = chance(0.8);
    return { user: pick([`generate a ${len} character password`, `I need a strong password${sym ? "" : " without symbols"}, ${len} characters`, `make me a secure password (${len} chars)`, `random password please, length ${len}`]),
      args: sym ? { length: len } : { length: len, symbols: false }, title: "Making a password", why: "I'll generate a random password on the device.",
      answer: (r) => `Here's a **${r.strength}** ${r.length}-character password:\n\n\`${r.password}\`\n\nStore it in a password manager, and don't reuse it on other sites.` };
  },
  password_strength: () => {
    const pw = pick(["password123", "Summer2024", "qwerty", "Tr0ub4dor&3", "correct-horse-battery-staple", "Mnx!2025rocks", "iloveyou", "G7#kp!2Lq9@zW", "abc12345", "Chennai@123"]);
    return { user: pick([`is "${pw}" a strong password?`, `check my password strength: ${pw}`, `how secure is ${pw}`]), args: { password: pw }, title: "Checking the password", why: "I'll measure how strong it is.",
      answer: (r) => `**${r.rating[0].toUpperCase() + r.rating.slice(1)}** (about ${r.entropy_bits} bits).${r.tips.length ? `\n\nTo make it stronger:\n${r.tips.map((t) => `- ${t[0].toUpperCase() + t.slice(1)}`).join("\n")}` : "\n\nThat's a solid password. Keep it unique to one site."}` };
  },
  uuid: () => {
    const c = pick([1, 1, 1, 3, 5]);
    return { user: c === 1 ? pick(["generate a uuid", "give me a random UUID", "new uuid v4"]) : `generate ${c} UUIDs`, args: c === 1 ? {} : { count: c }, title: "Generating IDs", why: "I'll create random UUIDs.",
      answer: (r) => (r.uuids.length === 1 ? `\`${r.uuids[0]}\`` : `\`\`\`text\n${r.uuids.join("\n")}\n\`\`\``) };
  },
  word_count: () => {
    const t = pick(TEXTS);
    return { user: pick([`count the words: ${t}`, `how many words and characters in this? "${t}"`, `word count for: ${t}`]), args: { text: t }, title: "Counting words", why: "I'll count exactly with the tool.",
      answer: (r) => `**${r.words} words**, ${r.characters} characters (${r.characters_no_spaces} without spaces), ${r.sentences} sentence${r.sentences === 1 ? "" : "s"}. Reading time: about ${r.reading_time_min} min.` };
  },
  change_case: () => {
    const c = pick(["upper", "lower", "title", "snake", "camel", "kebab", "sentence"]);
    const t = pick(["hello world from mnx", "user account settings", "THE QUICK BROWN FOX", "total order amount", "make the most of today", "get user profile data"]);
    const say = { upper: "uppercase", lower: "lowercase", title: "title case", snake: "snake_case", camel: "camelCase", kebab: "kebab-case", sentence: "sentence case" }[c];
    return { user: pick([`convert "${t}" to ${say}`, `${say}: ${t}`, `change this to ${say} - ${t}`]), args: { text: t, case: c }, title: "Changing case", why: `I'll convert it to ${say}.`, answer: (r) => `\`${r.text}\`` };
  },
  slugify: () => {
    const t = pick(TITLES);
    return { user: pick([`make a url slug for "${t}"`, `slugify: ${t}`]), args: { text: t }, title: "Making a slug", why: "I'll turn the title into a URL slug.", answer: (r) => `\`${r.slug}\`` };
  },
  base64: () => {
    const enc = chance(0.6);
    const plain = pick(["hello world", "Mnx is awesome", "user:password", "Chennai 2025", "{\"ok\": true}"]);
    const text = enc ? plain : Buffer.from(plain).toString("base64");
    return { user: enc ? pick([`base64 encode "${plain}"`, `encode this in base64: ${plain}`]) : pick([`decode this base64: ${text}`, `what does ${text} decode to?`]), args: enc ? { text } : { text, mode: "decode" },
      title: enc ? "Encoding" : "Decoding", why: `I'll ${enc ? "encode" : "decode"} it exactly.`, answer: (r) => (enc ? `In Base64:\n\n\`${r.output}\`` : `It decodes to:\n\n\`${r.output}\``) };
  },
  url_encode: () => {
    const t = pick(["chennai beach & food", "hello world?", "50% off!", "name=Ravi Kumar&city=Delhi", "café menu"]);
    return { user: pick([`url encode: ${t}`, `percent-encode "${t}" for a link`]), args: { text: t }, title: "Encoding for a URL", why: "I'll percent-encode it.", answer: (r) => `\`${r.output}\`` };
  },
  hash_text: () => {
    const a = pick(["sha256", "sha256", "md5", "sha1", "sha512"]);
    const t = pick(["hello", "Mnx", "password", "The quick brown fox", "abc123"]);
    return { user: pick([`${a} hash of "${t}"`, `hash "${t}" with ${a}`]), args: a === "sha256" && chance(0.5) ? { text: t } : { text: t, algorithm: a }, title: "Hashing", why: `I'll compute the ${a.toUpperCase()} hash.`,
      answer: (r) => `${r.algorithm.toUpperCase()} of "${t}":\n\n\`${r.hash}\`${r.algorithm === "md5" || r.algorithm === "sha1" ? "\n\nNote: MD5 and SHA-1 aren't safe for passwords or security; use SHA-256 or better." : ""}` };
  },
  json_format: () => {
    const good = chance(0.55);
    const j = good ? pick(['{"name":"Ravi","age":21,"skills":["python","js"]}', '[{"id":1,"done":false},{"id":2,"done":true}]', '{"city":"Chennai","temp":31.5}'])
      : pick(['{"name": "Ravi", "age": 21,}', "{'city': 'Chennai'}", '{"items": [1, 2, 3}', '{name: "Asha"}']);
    return { user: pick([`is this valid json? ${j}`, `format this json: ${j}`, `fix/check my JSON ${j}`]), args: { json: j }, title: "Checking JSON", why: "I'll parse it with the JSON tool.",
      answer: (r) => (r.valid ? `It's **valid JSON** (${r.type}). Formatted:\n\n\`\`\`json\n${r.formatted}\n\`\`\`` : `It's **not valid JSON**${r.where ? ` (${r.where})` : ""}: ${r.error}\n\nCommon causes: trailing commas, single quotes instead of double quotes, unquoted keys or missing brackets.`) };
  },
  regex_test: () => {
    const [pattern, text, what] = pick([["\\d+", "Order 66 shipped 3 items on 2025-05-04", "numbers"], ["[\\w.+-]+@[\\w-]+\\.\\w+", "Contact asha@mail.com or ravi.k@work.in", "emails"],
      ["\\b[A-Z][a-z]+\\b", "Priya met Arjun in Chennai", "capitalised words"], ["#\\w+", "Loving #summer and #travel today", "hashtags"]]);
    return { user: `test regex ${pattern} on "${text}"`, args: { pattern, text }, title: "Testing the regex", why: `I'll run the pattern to find ${what}.`,
      answer: (r) => `**${r.count} match${r.count === 1 ? "" : "es"}**: ${r.matches.map((m) => `\`${m.match}\``).join(", ")}` };
  },
  text_diff: () => {
    const a = pick(["milk\neggs\nbread\nrice", "line one\nline two\nline three", "apples\nbananas\ncherries"]);
    const lines = a.split("\n");
    lines[int(0, lines.length - 1)] = pick(["butter", "line 2 (edited)", "mangoes"]);
    const b = lines.join("\n");
    return { user: `what changed between these?\nOLD:\n${a}\nNEW:\n${b}`, args: { a, b }, title: "Comparing", why: "I'll diff the two texts line by line.",
      answer: (r) => `**${r.added} added, ${r.removed} removed**:\n\n\`\`\`diff\n${r.changes.join("\n")}\n\`\`\`` };
  },
  lorem_ipsum: () => {
    const p = pick([1, 2, 3]);
    return { user: pick([`give me ${p} paragraph${p > 1 ? "s" : ""} of lorem ipsum`, `placeholder text, ${p} paragraphs`]), args: { paragraphs: p }, title: "Placeholder text", why: "I'll generate lorem ipsum.", answer: (r) => r.text };
  },
  sort_lines: () => {
    const items = pick([["banana", "apple", "cherry", "apple", "mango"], ["Zara", "Arjun", "Meera", "Ravi", "Arjun"], ["10", "2", "33", "4"]]);
    const desc = chance(0.3);
    const uniq = items.length !== new Set(items).size && chance(0.6);
    return { user: `sort these${desc ? " Z to A" : ""}${uniq ? " and remove duplicates" : ""}:\n${items.join("\n")}`, args: { text: items.join("\n"), ...(desc ? { order: "desc" } : {}), ...(uniq ? { unique: true } : {}) },
      title: "Sorting", why: "I'll sort the lines.", answer: (r) => r.lines.map((l, k) => `${k + 1}. ${l}`).join("\n") };
  },
  extract_contacts: () => {
    const t = pick(["Call me at +91 98765 43210 or mail priya@example.com. Site: https://priya.dev #design", "Reach support@shop.in or visit https://shop.in/help, phone 044-2345-6789", "Follow @mnx #ai #opensource, docs at https://mnx.app/docs"]);
    return { user: `extract the emails, phones and links from: ${t}`, args: { text: t }, title: "Extracting contacts", why: "I'll pull out emails, phones, links and hashtags.",
      answer: (r) => Object.entries(r).filter(([, v]) => v.length).map(([k, v]) => `- **${k[0].toUpperCase() + k.slice(1)}:** ${v.join(", ")}`).join("\n") };
  },
  random_number: () => {
    const [lo, hi] = pick([[1, 10], [1, 100], [1, 6], [100, 999], [1, 50]]);
    const c = pick([1, 1, 3, 5]);
    return { user: c === 1 ? pick([`random number between ${lo} and ${hi}`, `pick a number from ${lo} to ${hi}`]) : `give me ${c} random numbers between ${lo} and ${hi}`, args: c === 1 ? { min: lo, max: hi } : { min: lo, max: hi, count: c },
      title: "Picking a number", why: "I'll use a true random number generator.", answer: (r) => `**${r.numbers.join(", ")}**` };
  },
  roll_dice: () => {
    const d = pick(["1d6", "2d6", "1d20", "3d6", "1d20+5", "4d6"]);
    return { user: pick([`roll ${d}`, `roll the dice: ${d}`, d === "1d6" ? "roll a die" : `throw ${d} for my game`]), args: { dice: d }, title: "Rolling dice", why: "I'll roll fairly.",
      answer: (r) => `🎲 **${r.total}**${r.rolls.length > 1 || /[+-]/.test(r.dice) ? ` (rolls: ${r.rolls.join(", ")}${/[+-]\d+$/.test(r.dice) ? `, ${r.dice.match(/[+-]\d+$/)[0]}` : ""})` : ""}` };
  },
  flip_coin: () => {
    const c = pick([1, 1, 1, 3, 10]);
    return { user: c === 1 ? pick(["flip a coin", "heads or tails?", "toss a coin for me"]) : `flip a coin ${c} times`, args: c === 1 ? {} : { count: c }, title: "Flipping", why: "I'll flip fairly.",
      answer: (r) => (r.flips.length === 1 ? `🪙 **${r.flips[0]}**!` : `**${r.heads} heads, ${r.tails} tails**: ${r.flips.join(", ")}`) };
  },
  pick_random: () => {
    const opts = pick([["pizza", "biryani", "dosa"], ["Priya", "Arjun", "Meera", "Sam"], ["beach", "mountains", "city"], ["Netflix", "a book", "a walk"]]);
    return { user: pick([`help me choose: ${opts.join(", ")}`, `pick one randomly from ${opts.join(" or ")}`, `who goes first? ${opts.join(", ")}`]), args: { options: opts }, title: "Choosing randomly", why: "I'll pick fairly at random.",
      answer: (r) => `The pick is **${r.picked[0]}**! 🎉` };
  },
  statistics: () => {
    const xs = Array.from({ length: int(5, 12) }, () => int(40, 99));
    return { user: pick([`find mean, median and mode of ${xs.join(", ")}`, `stats for these marks: ${xs.join(" ")}`, `average of ${xs.join(", ")}`]), args: { numbers: xs }, title: "Computing statistics", why: "I'll compute exact statistics.",
      answer: (r) => `| Measure | Value |\n|---|---|\n| Mean | **${n2(r.mean)}** |\n| Median | ${r.median} |\n| Mode | ${r.mode.length ? r.mode.join(", ") : "none"} |\n| Range | ${r.min}–${r.max} |\n| Std dev | ${n2(r.std_dev)} |` };
  },
  gcd_lcm: () => {
    const xs = pick([[12, 18], [24, 36, 60], [8, 14], [45, 75], [16, 40, 56], [7, 21]]);
    return { user: pick([`gcd and lcm of ${xs.join(", ")}`, `HCF and LCM of ${xs.join(" and ")}`]), args: { numbers: xs }, title: "GCD and LCM", why: "I'll compute both exactly.", answer: (r) => `- **GCD (HCF):** ${r.gcd}\n- **LCM:** ${r.lcm}` };
  },
  prime_factors: () => {
    const n = pick([97, 360, 1001, 7919, 1024, 2310, 9973, 123456, 600851, 97 * 89]);
    return { user: pick([`is ${n} prime?`, `prime factors of ${n}`, `factorise ${n}`]), args: { n }, title: "Factorising", why: "I'll factor it exactly.",
      answer: (r) => (r.is_prime ? `**Yes, ${n} is prime.**` : `**No.** ${n} = ${r.factors.join(" × ")}`) };
  },
  number_to_words: () => {
    const indian = chance(0.5);
    const n = pick([1234, 50000, 125000, 2500000, 1234567, 99999, 15075]);
    return { user: indian ? pick([`write ${n} in words (indian system)`, `${n} in words for a cheque, lakhs and crores`]) : pick([`write ${n} in words`, `how do you say ${n.toLocaleString("en-US")} in words?`]),
      args: indian ? { number: n, system: "indian" } : { number: n }, title: "Writing it in words", why: "I'll spell out the number.", answer: (r) => `**${r.words[0].toUpperCase() + r.words.slice(1)}**` };
  },
  roman_numerals: () => {
    const v = pick(["1994", "2025", "49", "3888", "MCMXC", "XLII", "MMXXIV", "14"]);
    return { user: /^\d/.test(v) ? pick([`${v} in roman numerals`, `convert ${v} to roman`]) : pick([`what number is ${v}?`, `convert roman ${v} to a number`]), args: { value: v }, title: "Roman numerals", why: "I'll convert exactly.",
      answer: (r) => (/^\d/.test(v) ? `${v} = **${r.roman}**` : `${r.roman} = **${r.number}**`) };
  },
  number_base: () => {
    const [v, from, to, say] = pick([["255", 10, 16, "255 to hex"], ["42", 10, 2, "42 to binary"], ["1011", 2, 10, "binary 1011 to decimal"], ["FF", 16, 10, "hex FF to decimal"], ["777", 8, 10, "octal 777 to decimal"], ["1000", 10, 2, "1000 in binary"]]);
    return { user: `convert ${say}`, args: { value: v, from_base: from, to_base: to }, title: "Converting bases", why: "I'll convert between number bases.", answer: (r) => `**${r.output}**\n\n(binary ${r.binary}, octal ${r.octal}, decimal ${r.decimal}, hex ${r.hex})` };
  },
  percentage_change: () => {
    const a = int(50, 5000);
    const b = Math.round(a * (0.5 + rand()));
    return { user: pick([`percentage change from ${a} to ${b}`, `my salary went from ${a} to ${b}, what % is that?`, `${a} to ${b} is how many percent increase?`]), args: { from: a, to: b }, title: "Percentage change", why: "I'll compute it exactly.",
      answer: (r) => `**${r.percent >= 0 ? "+" : ""}${r.percent}%** (${r.change >= 0 ? "up" : "down"} ${Math.abs(r.change)} from ${a} to ${b}).` };
  },
  discount: () => {
    const p = pick([999, 1499, 2499, 4999, 120, 59.99, 15000]);
    const pc = pick([10, 15, 20, 25, 30, 40, 50, 70]);
    return { user: pick([`${pc}% off on ${p}, what's the final price?`, `price ${p} with ${pc} percent discount`, `how much do I save on a ${p} item at ${pc}% off?`]), args: { price: p, percent: pc }, title: "Applying the discount", why: "I'll calculate the sale price.",
      answer: (r) => `Final price: **${n2(r.final_price)}** (you save ${n2(r.you_save)}).` };
  },
  tax: () => {
    const inc = chance(0.4);
    const a = pick([1000, 2500, 1180, 5900, 750, 12000]);
    const rate = pick([5, 12, 18, 28, 20, 7.5]);
    return { user: inc ? `price ${a} includes ${rate}% GST, how much is the tax?` : pick([`add ${rate}% GST to ${a}`, `${a} plus ${rate}% tax`, `what is ${a} with ${rate}% VAT?`]), args: inc ? { amount: a, rate, inclusive: true } : { amount: a, rate },
      title: "Calculating tax", why: "I'll compute the tax exactly.", answer: (r) => (inc ? `Tax inside ${a}: **${n2(r.tax)}** (price before tax ${n2(r.net)}).` : `**${n2(r.gross)}** in total (${n2(r.net)} + ${n2(r.tax)} tax).`) };
  },
  tip_split: () => {
    const bill = pick([675, 1240, 89.5, 3200, 450, 2875]);
    const tip = pick([10, 15, 18, 20]);
    const ppl = pick([2, 3, 4, 5, 6, 7]);
    return { user: pick([`bill is ${bill}, ${tip}% tip, split between ${ppl}`, `${ppl} of us, bill ${bill}, tip ${tip} percent. each pays?`]), args: { bill, tip_percent: tip, people: ppl }, title: "Splitting the bill", why: "I'll work out the tip and each share.",
      answer: (r) => `Each person pays **${n2(r.each)}**.\n\n- Tip (${tip}%): ${n2(r.tip)}\n- Total: ${n2(r.total)}` };
  },
  loan_emi: () => {
    const P = pick([200000, 500000, 1000000, 2500000, 50000, 3500000]);
    const rate = pick([7.5, 8.5, 9, 10.5, 12, 13.75]);
    const months = pick([12, 24, 36, 60, 120, 240]);
    return { user: pick([`EMI for a loan of ${P} at ${rate}% for ${months} months`, `I'm taking a ${P} loan, ${rate} percent interest, ${months / 12} years. monthly payment?`]), args: { principal: P, annual_rate: rate, months },
      title: "Calculating the EMI", why: "I'll use the standard EMI formula.", answer: (r) => `Your EMI is **${n2(r.emi)} per month**.\n\n| | Amount |\n|---|---|\n| Total interest | ${n2(r.total_interest)} |\n| Total paid | ${n2(r.total_paid)} |` };
  },
  compound_interest: () => {
    const P = pick([10000, 50000, 100000, 5000]);
    const rate = pick([6, 7, 8, 10, 12]);
    const years = pick([5, 10, 15, 20]);
    const add = chance(0.5) ? pick([500, 1000, 5000]) : 0;
    return { user: add ? `if I invest ${P} at ${rate}% and add ${add} every month for ${years} years, how much will I have?` : pick([`compound interest on ${P} at ${rate}% for ${years} years`, `${P} at ${rate}% compounded monthly for ${years} years`]),
      args: add ? { principal: P, rate, years, monthly_add: add } : { principal: P, rate, years }, title: "Projecting growth", why: "I'll compound it month by month.",
      answer: (r) => `After ${years} years you'd have about **${n2(r.final_amount)}**.\n\n- You put in: ${n2(r.deposited)}\n- Interest earned: ${n2(r.interest_earned)}\n\nReal returns vary; this assumes a steady ${rate}% a year.` };
  },
  fuel_cost: () => {
    const d = pick([120, 350, 500, 900, 45]);
    const eff = pick([12, 15, 18, 22]);
    const price = pick([95, 102, 105, 1.6, 1.85]);
    return { user: `fuel cost for a ${d} km trip, car does ${eff} km per litre, petrol ${price} per litre`, args: { distance_km: d, km_per_liter: eff, price_per_liter: price }, title: "Trip fuel cost", why: "I'll work out fuel and cost.",
      answer: (r) => `About **${n2(r.cost)}** for fuel (${n2(r.liters)} litres).` };
  },
  pace: () => {
    const d = pick([5, 10, 21.1, 42.2, 3]);
    const t = pick(d <= 5 ? ["25:00", "30:15", "22:40"] : d <= 10 ? ["52:30", "1:01:20", "45:00"] : ["1:58:00", "2:15:30", "4:05:00"]);
    return { user: `I ran ${d} km in ${t}, what's my pace?`, args: { distance_km: d, time: t }, title: "Working out pace", why: "I'll compute pace and speed.", answer: (r) => `**${r.pace_per_km} per km** (${r.speed_kmh} km/h). Nice run! 🏃` };
  },
  bmi: () => {
    const w = int(45, 110);
    const h = int(150, 190);
    return { user: pick([`my bmi? ${w} kg, ${h} cm`, `calculate BMI for weight ${w}kg height ${h}cm`, `I'm ${h} cm and ${w} kg, is that healthy?`]), args: { weight_kg: w, height_cm: h }, title: "Calculating BMI", why: "I'll compute BMI and its category.",
      answer: (r) => `Your BMI is **${r.bmi}** (${r.category}).\n\nA healthy weight for your height is about ${r.healthy_weight_range_kg[0]}–${r.healthy_weight_range_kg[1]} kg. BMI doesn't account for muscle, so treat it as a rough guide.` };
  },
  bmr: () => {
    const sex = pick(["male", "female"]);
    const w = int(50, 95);
    const h = int(152, 188);
    const age = int(18, 60);
    const act = pick(["sedentary", "light", "moderate", "active"]);
    return { user: `how many calories do I need? ${sex}, ${age} years, ${w} kg, ${h} cm, ${act === "sedentary" ? "desk job" : `${act} exercise`}`, args: { weight_kg: w, height_cm: h, age, sex, activity: act },
      title: "Estimating calories", why: "I'll use the Mifflin–St Jeor formula.", answer: (r) => `About **${r.daily_calories} kcal a day** to maintain your weight (BMR ${r.bmr} kcal at rest).\n\nEat roughly 300–500 kcal less to lose weight slowly, or more to gain.` };
  },
  age: (today) => {
    const b = `${int(1960, 2015)}-${String(int(1, 12)).padStart(2, "0")}-${String(int(1, 28)).padStart(2, "0")}`;
    return { user: pick([`how old am I if I was born on ${b}?`, `age from birthdate ${b}`, `born ${b}, exact age?`]), args: { birthdate: b }, title: "Working out the age", why: "I'll compute the exact age from today's date.",
      answer: (r) => `You're **${r.years} years, ${r.months} month${r.months === 1 ? "" : "s"} and ${r.days} day${r.days === 1 ? "" : "s"}** old (${r.days_lived.toLocaleString("en-US")} days). Next birthday in ${r.next_birthday_in_days} days 🎂` };
  },
  date_diff: () => {
    const a = randomDay();
    const b = addDays(a, int(10, 900));
    return { user: pick([`days between ${a} and ${b}`, `how long from ${a} to ${b}?`]), args: { from: a, to: b }, title: "Counting days", why: "I'll count the exact days.", answer: (r) => `**${r.total_days} days** (${r.weeks} weeks, or ${r.years_months_days}).` };
  },
  add_to_date: () => {
    const d = randomDay();
    const [k, n] = pick([["days", 90], ["days", 45], ["weeks", 6], ["months", 3], ["days", -30], ["years", 1]]);
    return { user: n < 0 ? `what date was ${-n} ${k} before ${d}?` : `what date is ${n} ${k} after ${d}?`, args: { date: d, [k]: n }, title: "Date maths", why: "I'll add it on the calendar.", answer: (r) => `**${r.weekday}, ${r.date}**` };
  },
  day_of_week: () => {
    const d = pick(["1947-08-15", "1969-07-20", "2000-01-01", "1990-12-25", randomDay(), randomDay()]);
    return { user: pick([`what day of the week was ${d}?`, `which day is ${d}?`]), args: { date: d }, title: "Finding the weekday", why: "I'll look up the weekday.", answer: (r) => `${d} is a **${r.weekday}**.` };
  },
  countdown: (today) => {
    const [ev, date] = pick([["New Year", `${Number(today.slice(0, 4)) + 1}-01-01`], ["my exam", addDays(today, int(5, 120))], ["Christmas", `${today.slice(0, 4)}-12-25`], ["our trip", addDays(today, int(10, 200))]]);
    return { user: pick([`how many days until ${ev}${ev.startsWith("my") || ev.startsWith("our") ? ` on ${date}` : ""}?`, `countdown to ${ev} (${date})`]), args: { date, event: ev }, title: "Counting down", why: "I'll count the days from today.",
      answer: (r) => (r.days >= 0 ? `**${r.days} days** to go until ${ev} (${r.date}).` : `${ev} was **${-r.days} days ago** (${r.date}).`) };
  },
  unix_time: () => {
    const v = pick(["1700000000", "1609459200", "1735689600", "2000-01-01", "1999-12-31T23:59:59Z"]);
    return { user: /^\d+$/.test(v) ? `convert unix timestamp ${v} to a date` : `unix timestamp for ${v}`, args: { value: v }, title: "Converting time", why: "I'll convert the timestamp.",
      answer: (r) => (/^\d+$/.test(v) ? `**${r.iso_utc.replace("T", " ").replace(".000Z", " UTC")}**` : `**${r.unix}**`) };
  },
  timezone_convert: (today) => {
    const [time, from, to] = pick([["9:30 pm", "Asia/Kolkata", "America/New_York"], ["10:00", "Europe/London", "Asia/Kolkata"], ["8 am", "America/Los_Angeles", "Europe/Berlin"], ["14:00", "Asia/Tokyo", "UTC"], ["6:30 pm", "Asia/Dubai", "Asia/Singapore"]]);
    return { user: pick([`what's ${time} ${from} in ${to}?`, `convert ${time} from ${from} to ${to}`]), args: { time, from_zone: from, to_zone: to }, title: "Converting time zones", why: "I'll convert using real time-zone rules (including daylight saving).",
      answer: (r) => `${r.input} in ${from} is **${r.time}** in ${to}.` };
  },
  color_convert: () => {
    const c = pick(["#1e90ff", "coral", "rgb(255, 99, 71)", "teal", "#ffcc00", "hsl(210, 100%, 56%)", "#8a2be2", "gold"]);
    return { user: pick([`convert ${c} to hex, rgb and hsl`, `what's ${c} in rgb?`, `color codes for ${c}`]), args: { color: c }, title: "Converting the color", why: "I'll convert between color formats.",
      answer: (r) => `| Format | Value |\n|---|---|\n| HEX | \`${r.hex}\` |\n| RGB | \`${r.rgb}\` |\n| HSL | \`${r.hsl}\` |` };
  },
  public_holidays: (today) => {
    const cc = pick(["US", "FR", "DE"]);
    const y = Number(today.slice(0, 4));
    const list = HOLIDAYS[cc](y).sort((a, b) => a[0].localeCompare(b[0])).map(([date, name]) => ({ date, name }));
    const next = list.find((h) => h.date >= today) || null;
    return { user: pick([`public holidays in ${COUNTRY_CODES[cc]} ${y}`, `next public holiday in ${COUNTRY_CODES[cc]}?`, `${cc} holidays this year`]), args: { country_code: cc, year: y }, result: { country: cc, year: y, holidays: list, next },
      title: "Looking up holidays", why: "I'll fetch the official public holidays.", answer: (r) => `${next ? `The next one is **${next.name} on ${next.date}**.\n\n` : ""}| Date | Holiday |\n|---|---|\n${r.holidays.map((h) => `| ${h.date} | ${h.name} |`).join("\n")}` };
  },
  country_info: () => {
    const c = pick(COUNTRIES);
    const r = { name: c.name, official_name: c.official, capital: c.capital, population: c.pop, area_km2: c.area, region: c.region, languages: c.langs, currencies: c.cur, calling_code: c.code, timezones: c.tz, flag: c.flag };
    return { user: pick([`tell me about ${c.name}`, `capital and currency of ${c.name}`, `${c.name} country info`, `what's the population of ${c.name}?`]), args: { country: c.name }, result: r, title: "Looking up the country", why: "I'll get the country's key facts.",
      answer: (x) => `${x.flag} **${x.name}** (${x.official_name})\n\n| | |\n|---|---|\n| Capital | ${x.capital} |\n| Population | ${x.population.toLocaleString("en-US")} |\n| Region | ${x.region} |\n| Languages | ${x.languages.join(", ")} |\n| Currency | ${x.currencies.join(", ")} |\n| Calling code | ${x.calling_code} |` };
  },
  crypto_price: () => {
    const [coin, sym, lo, hi] = pick(COINS);
    const cur = pick(["usd", "usd", "inr", "eur"]);
    const mult = { usd: 1, inr: 83.4, eur: 0.92 }[cur];
    const price = Number((((lo + rand() * (hi - lo)) * mult) / 1).toPrecision(6));
    const ch = Math.round((rand() * 12 - 6) * 100) / 100;
    return { user: pick([`${sym} price${cur === "usd" ? "" : ` in ${cur.toUpperCase()}`}`, `how much is ${coin} now${cur === "usd" ? "" : ` in ${cur}`}?`]), args: cur === "usd" ? { coin } : { coin, currency: cur },
      result: { coin, currency: cur.toUpperCase(), price, change_24h_percent: ch }, title: "Checking the price", why: "Crypto prices move constantly, so I'll get the live price.",
      answer: (r) => `**${sym}: ${r.price.toLocaleString("en-US")} ${r.currency}** (${r.change_24h_percent >= 0 ? "▲ +" : "▼ "}${r.change_24h_percent}% in 24 h).\n\nPrices are volatile; this isn't financial advice.` };
  },
  book_search: () => {
    const q = pick(Object.keys(BOOKS));
    const books = BOOKS[q].map(([title, author, year, pages]) => ({ title, author, year, pages, url: `https://openlibrary.org/works/OL${int(10000, 99999)}W` }));
    return { user: pick([`find ${q}`, `search books: ${q}`, `recommend ${q}`]), args: { query: q }, result: { books }, title: "Searching books", why: "I'll search Open Library.",
      answer: (r) => r.books.map((b, k) => `${k + 1}. **${b.title}** by ${b.author} (${b.year}${b.pages ? `, ${b.pages} pages` : ""}) · [Open Library](${b.url})`).join("\n") };
  },
  find_words: () => {
    const kind = pick(["synonyms", "synonyms", "rhymes", "antonyms"]);
    const bank = { synonyms: SYNONYMS, rhymes: RHYMES, antonyms: ANTONYMS }[kind];
    const w = pick(Object.keys(bank));
    return { user: pick([`${kind} for ${w}`, `words that ${kind === "rhymes" ? "rhyme with" : kind === "antonyms" ? "mean the opposite of" : "mean the same as"} ${w}`]), args: kind === "synonyms" ? { word: w } : { word: w, kind },
      result: { word: w, kind, words: bank[w] }, title: "Finding words", why: `I'll look up ${kind}.`, answer: (r) => `${kind[0].toUpperCase() + kind.slice(1)} for **${w}**: ${r.words.join(", ")}.` };
  },
  air_quality: () => {
    const [city, country] = pick(PLACES);
    const aqi = int(15, 260);
    const level = aqi <= 50 ? "Good" : aqi <= 100 ? "Moderate" : aqi <= 150 ? "Unhealthy for sensitive groups" : aqi <= 200 ? "Unhealthy" : "Very unhealthy";
    const pm25 = Math.round(aqi * 0.42 * 10) / 10;
    const r = { location: `${city}, ${country}`, us_aqi: aqi, level, pm2_5: pm25, pm10: Math.round(pm25 * 1.7 * 10) / 10, ozone: int(20, 120), no2: int(5, 60) };
    return { user: pick([`air quality in ${city}`, `is the air good in ${city} today?`, `AQI ${city}`]), args: { location: city }, result: r, title: "Checking air quality", why: "I'll get the live air quality.",
      answer: (x) => `Air quality in ${city} is **${x.level}** (US AQI ${x.us_aqi}, PM2.5 ${x.pm2_5} µg/m³).${x.us_aqi > 100 ? "\n\nLimit long outdoor exercise, and consider a mask (N95) if you're sensitive." : "\n\nFine for outdoor activities."}` };
  },
  sunrise_sunset: (today) => {
    const [city, country, lat] = pick(PLACES);
    const doy = Number(new RealDate(today).getUTCMonth()) * 30;
    const dayLen = 12 + (lat / 90) * 6 * Math.sin(((doy - 80) / 365) * 2 * Math.PI);
    const mid = 12 * 60 + int(-30, 40);
    const rise = mid - (dayLen * 60) / 2;
    const set = mid + (dayLen * 60) / 2;
    const hm = (m) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`.replace(":60", ":59");
    const r = { location: `${city}, ${country}`, date: today, sunrise: hm(rise), sunset: hm(set), day_length: `${Math.floor(dayLen)} h ${Math.round((dayLen % 1) * 60)} min` };
    return { user: pick([`sunrise and sunset in ${city} today`, `when is sunset in ${city}?`, `what time does the sun rise in ${city}`]), args: { location: city }, result: r, title: "Checking the sun times", why: "I'll get today's sunrise and sunset.",
      answer: (x) => `In ${city} today: 🌅 sunrise **${x.sunrise}**, 🌇 sunset **${x.sunset}** (${x.day_length} of daylight).` };
  },
  recent_earthquakes: (today) => {
    const qs = Array.from({ length: int(3, 6) }, () => ({ magnitude: Math.round((4.5 + rand() * 2.5) * 10) / 10, place: pick(["120 km SE of Hualien City, Taiwan", "South of the Fiji Islands", "45 km W of Iquique, Chile", "Kermadec Islands, New Zealand", "Off the east coast of Honshu, Japan", "Banda Sea", "68 km N of Tobelo, Indonesia"]), time: `${addDays(today, -int(0, 6))} ${String(int(0, 23)).padStart(2, "0")}:${String(int(0, 59)).padStart(2, "0")} UTC` })).sort((a, b) => b.magnitude - a.magnitude);
    return { user: pick(["any big earthquakes this week?", "recent earthquakes", "latest earthquakes worldwide"]), args: {}, result: { count: qs.length, earthquakes: qs }, title: "Checking earthquakes", why: "I'll get the latest USGS earthquake feed.",
      answer: (r) => `${r.count} earthquakes of magnitude 4.5+ in the last 7 days. The strongest:\n\n${r.earthquakes.slice(0, 5).map((q) => `- **M${q.magnitude}**, ${q.place} (${q.time})`).join("\n")}` };
  },
  random_joke: () => {
    const j = pick(JOKES);
    return { user: pick(["tell me a joke", "make me laugh", "joke please", "got a dad joke?"]), args: {}, result: { joke: j }, title: "Finding a joke", why: "I'll fetch a clean joke.", answer: (r) => `${r.joke} 😄` };
  },
  quote: () => {
    const [q, a] = pick(QUOTES);
    return { user: pick(["give me a motivational quote", "inspire me", "quote of the day"]), args: {}, result: { quote: q, author: a }, title: "Finding a quote", why: "I'll fetch an inspiring quote.", answer: (r) => `> ${r.quote}\n>\n> — **${r.author}**` };
  },
  github_repo: () => {
    const [repo, desc, stars, forks, lang, lic] = pick(REPOS);
    const r = { name: repo, description: desc, stars: stars + int(0, 900), forks: forks + int(0, 300), open_issues: int(100, 9000), language: lang, license: lic, updated: "2025-0" + int(1, 9) + "-1" + int(0, 9), url: `https://github.com/${repo}` };
    return { user: pick([`github stats for ${repo}`, `how many stars does ${repo} have?`, `tell me about the ${repo} repo`]), args: { repo }, result: r, title: "Checking GitHub", why: "I'll get the repository's live stats.",
      answer: (x) => `**[${x.name}](${x.url})**: ${x.description}\n\n- ⭐ ${x.stars.toLocaleString("en-US")} stars · 🍴 ${x.forks.toLocaleString("en-US")} forks\n- Language: ${x.language} · License: ${x.license}` };
  },
  package_info: () => {
    const [reg, name, ver, desc] = pick(PACKAGES);
    const r = reg === "npm" ? { name, version: ver, description: desc, install: `npm install ${name}`, url: `https://www.npmjs.com/package/${name}` } : { name, version: ver, summary: desc, install: `pip install ${name}`, url: `https://pypi.org/project/${name}/` };
    return { user: pick([`latest version of ${name}${reg === "pypi" ? " on pypi" : ""}`, `what is the ${name} ${reg === "npm" ? "npm" : "python"} package?`]), args: reg === "npm" ? { name } : { name, registry: "pypi" }, result: r, title: "Checking the package", why: "I'll look up the registry.",
      answer: (x) => `**${x.name} ${x.version}**: ${x.description || x.summary}\n\nInstall with:\n\n\`\`\`bash\n${x.install}\n\`\`\`` };
  },
  remember: () => {
    const [fact] = pick(MEMORY_FACTS);
    return { user: pick([`remember that ${fact.replace(/^I'm/, "I'm").replace(/^My/, "my")}`, `please remember: ${fact}`, `note to self, remember ${fact.toLowerCase()}`]), args: { fact }, result: { saved: true, fact, total_memories: int(1, 12) },
      title: "Saving to memory", why: "The user wants me to remember this, so I'll save it.", answer: () => `Got it. I'll remember that. 🧠` };
  },
  recall: () => {
    const facts = Array.from(new Set(Array.from({ length: int(2, 5) }, () => pick(MEMORY_FACTS)[0])));
    return { user: pick(["what do you remember about me?", "what have I told you to remember?", "show my memories"]), args: {}, result: { memories: facts }, title: "Checking memory", why: "I'll list what I've been asked to remember.",
      answer: (r) => `Here's what I remember:\n\n${r.memories.map((m) => `- ${m}`).join("\n")}\n\nSay "forget …" to remove anything.` };
  },
  forget: () => {
    const [fact, key] = pick(MEMORY_FACTS);
    return { user: pick([`forget that ${fact.toLowerCase()}`, `forget about my ${key}`]), args: { fact: key }, result: { removed: 1, remaining: int(0, 6) }, title: "Forgetting", why: "I'll delete that memory.", answer: () => "Done, I've forgotten that." };
  },
  set_timer: () => {
    const [m, label] = pick([[5, "Tea"], [10, "Pasta"], [25, "Focus session"], [15, "Break"], [3, "Eggs"], [45, "Study"], [1, "Plank"]]);
    return { user: pick([`set a ${m} minute timer for ${label.toLowerCase()}`, `timer ${m} min`, `remind me in ${m} minutes (${label.toLowerCase()})`]), args: { minutes: m, label }, title: "Setting a timer", why: "I'll start a countdown in the app.",
      result: { set: true, minutes: m, label, ends_at: "" }, answer: () => `⏱️ **${m}-minute timer** started for ${label.toLowerCase()}. I'll ring when it's done. Keep Mnx open.` };
  },
  save_note: () => {
    const [title, text] = pick([["Shopping", "milk\neggs\nbread\ntomatoes"], ["Ideas", "build a budget app\nlearn guitar"], ["Packing", "charger\npassport\ntoothbrush"], ["Gift ideas", "book for Amma\nwatch for Ravi"]]);
    return { user: `save a note called ${title.toLowerCase()}: ${text.replace(/\n/g, ", ")}`, args: { title, text }, result: { saved: true, title, total_notes: int(1, 9) }, title: "Saving a note", why: "I'll save the note on this device.",
      answer: () => `Saved your **${title}** note. Ask "show my notes" anytime.` };
  },
  list_notes: () => {
    const notes = [{ title: "Shopping", text: "milk\neggs\nbread" }, { title: "Ideas", text: "build a budget app" }].slice(0, int(1, 2));
    return { user: pick(["show my notes", "what notes do I have?", "read my shopping note"]), args: {}, result: { notes }, title: "Opening notes", why: "I'll list the saved notes.",
      answer: (r) => r.notes.map((n) => `**${n.title}**\n${n.text.split("\n").map((l) => `- ${l}`).join("\n")}`).join("\n\n") };
  },
  cron_explain: () => {
    const e = pick(["0 9 * * 1-5", "*/15 * * * *", "30 2 * * *", "0 0 1 * *", "0 18 * * 5", "0 */6 * * *"]);
    return { user: pick([`what does the cron ${e} mean?`, `explain cron expression "${e}"`]), args: { expression: e }, title: "Reading the cron", why: "I'll translate the schedule.", answer: (r) => `\`${e}\` runs **${r.meaning}**.` };
  },
  http_status: () => {
    const c = pick([200, 201, 301, 400, 401, 403, 404, 409, 422, 429, 500, 502, 503]);
    return { user: pick([`what does HTTP ${c} mean?`, `getting a ${c} error from my API, why?`]), args: { code: c }, title: "HTTP status", why: "I'll explain the status code.", answer: (r) => `**${r.code} ${r.name}**: ${r.meaning}` };
  },
  jwt_decode: () => {
    const payload = { sub: String(int(100, 999)), name: pick(["Asha", "Ravi", "Sam"]), iat: 1735689600, exp: pick([1735693200, 1893456000]) };
    const tok = `${Buffer.from('{"alg":"HS256","typ":"JWT"}').toString("base64url")}.${Buffer.from(JSON.stringify(payload)).toString("base64url")}.sig`;
    return { user: `decode this jwt ${tok}`, args: { token: tok }, title: "Decoding the token", why: "I'll decode the header and payload.",
      answer: (r) => `Algorithm **${r.header.alg}**. Payload:\n\n\`\`\`json\n${JSON.stringify(r.payload, null, 2)}\n\`\`\`\n\n${r.expired ? "⚠️ This token has **expired**." : "It's still valid until " + r.expires.slice(0, 10) + "."} (Signature not verified.)` };
  },
  ip_subnet: () => {
    const c = pick(["192.168.1.0/24", "10.0.0.0/16", "172.16.5.77/28", "192.168.10.130/26", "10.10.0.0/22"]);
    return { user: `subnet details for ${c}`, args: { cidr: c }, title: "Calculating the subnet", why: "I'll compute the network range.",
      answer: (r) => `| | |\n|---|---|\n| Network | ${r.network} |\n| Netmask | ${r.netmask} |\n| Hosts | ${r.first_host} – ${r.last_host} (${r.usable_hosts.toLocaleString("en-US")} usable) |\n| Broadcast | ${r.broadcast} |` };
  },
};
const missing = MORE_TOOLS.map((t) => t.name).filter((n) => !TASKS[n]);
if (missing.length) throw new Error(`No training task for: ${missing.join(", ")}`);

/* ───────────── conversations ───────────── */
// Casual typing (lowercase), but only when it can't change text the tool needs verbatim.
function lowerish(t, args) {
  const low = t.toLowerCase().replace(/[?.!]$/, "");
  const safe = Object.values(args).every((v) => typeof v !== "string" || low.includes(v));
  return safe && chance(0.25) ? low : t;
}
function build(name, testSet) {
  const today = randomDay();
  const task = TASKS[name](today);
  const result = task.result ?? runAt(today, name, task.args);
  const memories = chance(0.1) ? [pick(MEMORY_FACTS)[0]] : [];
  // Test questions use openings never seen in training, so they check new phrasings.
  const user = testSet ? `${pick(["hey mnx, ", "Quick one: ", "Could you help? ", "I need this: ", "Mnx, ", "Question - "])}${task.user}` : lowerish(task.user, task.args);
  const msgs = [system(today, memories), { role: "user", content: user }];
  const train_from = 1;
  msgs.push({ role: "assistant", content: think(task.title, task.why) + call(name, task.args) });
  msgs.push({ role: "user", content: response(result) });
  msgs.push({ role: "assistant", content: think("Answering", "I'll give the result clearly, first.") + task.answer(result) });
  return { category: `quick_${name}`, messages: msgs, train_from, expect: { tool: name }, mock: { [name]: [result] } };
}
// Memory in action: the answer uses a remembered fact, no tool needed.
function memoryUse() {
  const today = randomDay();
  const [fact, key, lead] = pick(MEMORY_FACTS.filter((m) => m[2]));
  const q = { food: ["suggest a dinner recipe", "what should I eat tonight?"], color: ["what's my favourite color?", "pick a theme color for my site"], city: ["where do I live?", "suggest a weekend outing"],
    birthday: ["when is my sister's birthday?", "gift ideas for my sister"], learning: ["give me a small project idea", "what should I practise today?"], dog: ["what's my dog's name?", "fun name tag ideas for my dog"],
    work: ["best time for me to work out?", "tips for better sleep"], car: ["how far can I go on 40 litres?", "what's my car's mileage?"], allergy: ["suggest a snack", "is pad thai safe for me?"] }[key];
  const user = pick(q);
  const ans = {
    "what's my favourite color?": "Your favourite color is **teal**. 🩵", "where do I live?": "You live in **Chennai**.", "when is my sister's birthday?": "Your sister's birthday is on **12 March**. 🎂",
    "what's my dog's name?": "Your dog is called **Bruno**. 🐶", "how far can I go on 40 litres?": "At 15 km per litre, 40 litres takes you about **600 km**.", "what's my car's mileage?": "Your car does **15 km per litre**.",
  }[user] || `${lead}, here's something that fits:\n\n${{ food: "**Paneer tikka masala** with jeera rice: marinate paneer in yogurt and spices, grill it, then simmer in a tomato-cream gravy.", color: "**Teal (#008080)** as the main color, with white and a warm coral accent.", city: "Try **Marina Beach at sunrise** and breakfast at a local tiffin centre in Mylapore.",
    birthday: "- A personalised photo book\n- A plant and a handwritten card\n- A cooking class for two", learning: "Build a **to-do list app in Python** that saves tasks to a JSON file. It practises functions, lists and file handling.", dog: "- \"Bruno, the good boy\"\n- \"If lost, I'm Bruno. Please call my human!\"",
    work: "Exercise **after you wake up** (in the afternoon), before your shift, rather than right before sleeping.", allergy: "**Roasted chickpeas** or fruit with yogurt are good snacks. Always check labels for peanut traces." }[key]}`;
  const msgs = [system(today, [fact, ...(chance(0.4) ? [pick(MEMORY_FACTS)[0]] : [])].filter((v, k, a) => a.indexOf(v) === k)), { role: "user", content: user },
    { role: "assistant", content: think("Using what I remember", "The user told me something relevant earlier; I'll use it.") + ans }];
  return { category: "memory_use", messages: msgs, train_from: 1, expect: { tool: null }, mock: {} };
}

// Balanced: at most CAP examples per tool; low-variety tools are repeated
// (with a different date/name in the system prompt) up to FLOOR.
function generate(n, testSet) {
  R = rng(testSet ? 99 : 11);
  const names = [...MORE_TOOLS.map((t) => t.name), "memory_use"];
  const CAP = Math.ceil((n / names.length) * 1.4);
  const FLOOR = testSet ? 0 : Math.min(80, Math.floor(n / names.length));
  const seen = new Set();
  const count = Object.fromEntries(names.map((k) => [k, 0]));
  const outRows = [];
  let tries = 0;
  while (outRows.length < n && tries++ < n * 40) {
    const open = names.filter((k) => count[k] < CAP);
    if (!open.length) break;
    const name = pick(open);
    const ex = name === "memory_use" ? memoryUse() : build(name, testSet);
    const key = testSet ? ex.messages[1].content : JSON.stringify(ex.messages.slice(1));
    if (seen.has(key) && (testSet || count[name] >= FLOOR)) continue;
    seen.add(key);
    count[name]++;
    outRows.push(ex);
  }
  return outRows;
}

fs.mkdirSync(out, { recursive: true });
const train = generate(nTrain, false);
const trainPrompts = new Set(train.map((e) => e.messages[1].content.toLowerCase()));
const test = generate(nTest * 3, true).filter((e) => !trainPrompts.has(e.messages[1].content.toLowerCase())).slice(0, nTest);
const w = (file, rows, fn) => (append ? fs.appendFileSync : fs.writeFileSync)(path.join(out, file), rows.map(fn).join("\n") + "\n");
w("train.jsonl", train, (e) => JSON.stringify({ category: e.category, messages: e.messages, train_from: e.train_from }));
w("test.jsonl", test, (e) => JSON.stringify({ category: e.category, messages: e.messages.slice(0, e.train_from + 1), mock: e.mock, expect: e.expect, reference: e.messages.slice(e.train_from + 1) }));
console.log(`quick tools: ${train.length.toLocaleString()} training conversations, ${test.length} test cases (${MORE_TOOLS.length} tools) -> ${out}${append ? " (appended)" : ""}`);

// Client-side tools Mnx can call. Each runs on this server and returns
// { result } for the model plus optional { display } for the UI to render.
// Weather, geocoding, places and routing use free, key-less public APIs
// (Open-Meteo, OpenStreetMap Nominatim, OSRM).

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const UA = "Mnx-Assistant/1.0 (+https://github.com/Tharunmaks/Mnx)";
// The code runner only ever runs after the user approves each program in the
// chat. Turn it off completely with MNX_CODE_RUNNER=off.
export const CODE_RUNNER = !/^(0|off|false|no)$/i.test(process.env.MNX_CODE_RUNNER || "on");

export const TOOLS = [
  {
    name: "get_user_location",
    description:
      "Get the user's current location (coordinates and place name) as shared by their browser. Use when the user refers to 'here', 'near me', 'my area', or asks about local weather/places without naming a location.",
    eager_input_streaming: true,
    input_schema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "get_weather",
    description:
      "Get current weather and a daily forecast for a place. Provide either a place name or coordinates.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        location: { type: "string", description: "City or place name, e.g. 'Chennai' or 'Paris, France'" },
        latitude: { type: "number" },
        longitude: { type: "number" },
        days: { type: "integer", minimum: 1, maximum: 7, description: "Forecast days (default 5)" },
        units: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      additionalProperties: false,
    },
  },
  {
    name: "find_places",
    description:
      "Find places such as restaurants, cafes, shops, hotels, hospitals, attractions or addresses, optionally near a location. Returns names, addresses, coordinates and Google Maps links.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to look for, e.g. 'pizza restaurant', 'pharmacy', 'Eiffel Tower'" },
        near: { type: "string", description: "Place name to search around, e.g. 'Koramangala, Bangalore'" },
        latitude: { type: "number", description: "Search around these coordinates instead of 'near'" },
        longitude: { type: "number" },
        limit: { type: "integer", minimum: 1, maximum: 15 },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "get_directions",
    description:
      "Get a route between two places with distance, travel time, turn-by-turn steps and a Google Maps directions link. Use 'my location' as origin to start from the user's current location.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        origin: { type: "string" },
        destination: { type: "string" },
        mode: { type: "string", enum: ["driving", "walking", "cycling"] },
      },
      required: ["origin", "destination"],
      additionalProperties: false,
    },
  },
  {
    name: "create_file",
    description:
      "Create a downloadable file for the user: source code in any language (Python, JavaScript, HTML, CSS, Ruby, C, C++, Lua, Go, Rust, Java...), JSON, YAML, CSV, Markdown, plain text, SVG, etc. The user sees it as a file card with preview and download. Put the complete file contents in 'content'.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        filename: { type: "string", description: "File name with extension, e.g. 'snake_game.py'" },
        description: { type: "string", description: "One line describing the file" },
        content: { type: "string", description: "Full file contents" },
      },
      required: ["filename", "content"],
      additionalProperties: false,
    },
  },
  {
    name: "create_document",
    description:
      "Create a formatted document the user can download: a PDF, a Word document (docx), or a PowerPoint slide deck (pptx). For pdf/docx fill 'sections'; for pptx fill 'slides'. Body text supports simple markdown: paragraphs, '- ' bullets, '1. ' numbered items and **bold**.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        format: { type: "string", enum: ["pdf", "docx", "pptx"] },
        filename: { type: "string", description: "File name without extension" },
        title: { type: "string" },
        subtitle: { type: "string" },
        theme: { type: "string", enum: ["midnight", "ocean", "sunset", "forest", "paper"], description: "Color theme" },
        sections: {
          type: "array",
          items: {
            type: "object",
            properties: { heading: { type: "string" }, body: { type: "string" } },
            required: ["body"],
          },
        },
        slides: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              bullets: { type: "array", items: { type: "string" } },
              notes: { type: "string", description: "Speaker notes" },
            },
            required: ["title"],
          },
        },
      },
      required: ["format", "title"],
      additionalProperties: false,
    },
  },
  {
    name: "create_image",
    description:
      "Create an image from a text description (illustrations, photos, logos, art, wallpapers). Write a detailed English prompt describing subject, style, lighting and composition.",
    eager_input_streaming: true,
    input_schema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Detailed description of the image" },
        aspect: { type: "string", enum: ["square", "landscape", "portrait"] },
      },
      required: ["prompt"],
      additionalProperties: false,
    },
  },
  ...(CODE_RUNNER
    ? [
        {
          name: "run_code",
          description:
            "Run a short Python or JavaScript program on the user's device and get its output. The user sees the code and must approve it first. Use it for calculations, data processing, simulations and charts (Python: save charts with matplotlib to a .png in the current folder and they are shown to the user). Always print the results you need.",
          eager_input_streaming: true,
          input_schema: {
            type: "object",
            properties: {
              language: { type: "string", enum: ["python", "javascript"] },
              code: { type: "string", description: "The complete program" },
            },
            required: ["language", "code"],
            additionalProperties: false,
          },
        },
      ]
    : []),
];

// Extra tools only the local model needs (online models use Anthropic's
// built-in web search / fetch instead).
export const LOCAL_ONLY_TOOLS = [
  {
    name: "search_web",
    description: "Search the internet for current information, news, facts or anything you don't know.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  },
  {
    name: "read_webpage",
    description: "Read the text of a web page (for example a search result) to get details.",
    input_schema: {
      type: "object",
      properties: { url: { type: "string" } },
      required: ["url"],
    },
  },
];

const WEATHER_CODES = {
  0: ["Clear sky", "sun"], 1: ["Mainly clear", "sun"], 2: ["Partly cloudy", "partly"], 3: ["Overcast", "cloud"],
  45: ["Fog", "fog"], 48: ["Rime fog", "fog"],
  51: ["Light drizzle", "drizzle"], 53: ["Drizzle", "drizzle"], 55: ["Heavy drizzle", "drizzle"],
  56: ["Freezing drizzle", "drizzle"], 57: ["Freezing drizzle", "drizzle"],
  61: ["Light rain", "rain"], 63: ["Rain", "rain"], 65: ["Heavy rain", "rain"],
  66: ["Freezing rain", "rain"], 67: ["Freezing rain", "rain"],
  71: ["Light snow", "snow"], 73: ["Snow", "snow"], 75: ["Heavy snow", "snow"], 77: ["Snow grains", "snow"],
  80: ["Rain showers", "rain"], 81: ["Rain showers", "rain"], 82: ["Violent showers", "storm"],
  85: ["Snow showers", "snow"], 86: ["Heavy snow showers", "snow"],
  95: ["Thunderstorm", "storm"], 96: ["Thunderstorm with hail", "storm"], 99: ["Thunderstorm with hail", "storm"],
};

async function getJson(url) {
  const res = await fetch(url, {
    headers: { "user-agent": UA, accept: "application/json" },
    signal: AbortSignal.timeout(12000),
  });
  if (!res.ok) throw new Error(`${new URL(url).host} returned HTTP ${res.status}`);
  return res.json();
}

const str = (v, max = 500) => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : undefined);
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

async function geocode(query) {
  const q = new URLSearchParams({ q: query, format: "jsonv2", limit: "1", addressdetails: "1" });
  const [hit] = await getJson(`https://nominatim.openstreetmap.org/search?${q}`);
  if (!hit) throw new Error(`Couldn't find a place called "${query}"`);
  return { name: hit.display_name, latitude: Number(hit.lat), longitude: Number(hit.lon) };
}

async function reverseGeocode(lat, lon) {
  try {
    const q = new URLSearchParams({ lat, lon, format: "jsonv2", zoom: "14" });
    const r = await getJson(`https://nominatim.openstreetmap.org/reverse?${q}`);
    const a = r.address || {};
    const local = a.suburb || a.neighbourhood || a.city_district || a.town || a.village;
    const city = a.city || a.town || a.village || a.county;
    return [local !== city ? local : null, city, a.state, a.country].filter(Boolean).join(", ") || r.display_name;
  } catch {
    return null;
  }
}

function distanceKm(a, b) {
  const R = 6371;
  const dLat = ((b.latitude - a.latitude) * Math.PI) / 180;
  const dLon = ((b.longitude - a.longitude) * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((a.latitude * Math.PI) / 180) * Math.cos((b.latitude * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

async function userLocation(ctx) {
  if (!ctx.location) return null;
  const { latitude, longitude } = ctx.location;
  const name = ctx.location.name || (await reverseGeocode(latitude, longitude)) || `${latitude.toFixed(3)}, ${longitude.toFixed(3)}`;
  ctx.location.name = name;
  return { latitude, longitude, name };
}

const handlers = {
  async get_user_location(_input, ctx) {
    const loc = await userLocation(ctx);
    if (!loc)
      return {
        result: { available: false, message: "The user hasn't shared their location. Ask them which city they're in." },
        display: { kind: "location", available: false },
      };
    return { result: loc, display: { kind: "location", available: true, ...loc } };
  },

  async get_weather(input, ctx) {
    let place;
    const lat = num(input.latitude);
    const lon = num(input.longitude);
    const name = str(input.location, 200);
    if (lat !== undefined && lon !== undefined) place = { latitude: lat, longitude: lon, name: name || (await reverseGeocode(lat, lon)) };
    else if (name) {
      const q = new URLSearchParams({ name: name.split(",")[0], count: "5", format: "json" });
      const g = await getJson(`https://geocoding-api.open-meteo.com/v1/search?${q}`);
      let hit = g.results?.[0];
      // Prefer a match whose country/region appears in the query ("Paris, Texas").
      const extra = name.split(",").slice(1).join(",").toLowerCase().trim();
      if (extra && g.results) hit = g.results.find((r) => `${r.admin1} ${r.country}`.toLowerCase().includes(extra)) || hit;
      if (hit) place = { latitude: hit.latitude, longitude: hit.longitude, name: [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ") };
      else place = await geocode(name);
    } else {
      place = await userLocation(ctx);
      if (!place) throw new Error("No location given and the user hasn't shared theirs. Ask which city.");
    }
    const days = Math.min(7, Math.max(1, Math.round(num(input.days) ?? 5)));
    const f = input.units === "fahrenheit";
    const q = new URLSearchParams({
      latitude: place.latitude,
      longitude: place.longitude,
      current: "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,is_day,precipitation",
      daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset,uv_index_max",
      timezone: "auto",
      forecast_days: String(days),
      temperature_unit: f ? "fahrenheit" : "celsius",
      wind_speed_unit: f ? "mph" : "kmh",
    });
    const w = await getJson(`https://api.open-meteo.com/v1/forecast?${q}`);
    const c = w.current;
    const [cond, icon] = WEATHER_CODES[c.weather_code] || ["Unknown", "cloud"];
    const unit = f ? "°F" : "°C";
    const daily = w.daily.time.map((date, i) => {
      const [dc, di] = WEATHER_CODES[w.daily.weather_code[i]] || ["Unknown", "cloud"];
      return {
        date,
        condition: dc,
        icon: di,
        max: Math.round(w.daily.temperature_2m_max[i]),
        min: Math.round(w.daily.temperature_2m_min[i]),
        rain_chance: w.daily.precipitation_probability_max?.[i],
        uv_max: w.daily.uv_index_max?.[i],
      };
    });
    const result = {
      location: place.name,
      local_time: c.time,
      timezone: w.timezone,
      current: {
        condition: cond,
        temperature: `${Math.round(c.temperature_2m)}${unit}`,
        feels_like: `${Math.round(c.apparent_temperature)}${unit}`,
        humidity: `${c.relative_humidity_2m}%`,
        wind: `${Math.round(c.wind_speed_10m)} ${f ? "mph" : "km/h"}`,
        precipitation_mm: c.precipitation,
      },
      sunrise: w.daily.sunrise?.[0],
      sunset: w.daily.sunset?.[0],
      daily,
    };
    return {
      result,
      display: {
        kind: "weather",
        location: place.name,
        unit,
        condition: cond,
        icon: c.is_day === 0 && icon === "sun" ? "moon" : icon,
        temperature: Math.round(c.temperature_2m),
        feels_like: Math.round(c.apparent_temperature),
        humidity: c.relative_humidity_2m,
        wind: result.current.wind,
        sunrise: result.sunrise,
        sunset: result.sunset,
        daily,
      },
    };
  },

  async find_places(input, ctx) {
    const query = str(input.query, 200);
    if (!query) throw new Error("'query' is required");
    const limit = Math.min(15, Math.max(1, Math.round(num(input.limit) ?? 8)));
    let center = null;
    const lat = num(input.latitude);
    const lon = num(input.longitude);
    const near = str(input.near, 200);
    if (lat !== undefined && lon !== undefined) center = { latitude: lat, longitude: lon };
    else if (near && /^(me|here|my location|current location)$/i.test(near)) center = await userLocation(ctx);
    else if (near) center = await geocode(near);

    const params = { q: query, format: "jsonv2", limit: String(limit), addressdetails: "1", extratags: "1" };
    if (center) {
      const d = 0.08; // ~9 km box
      params.viewbox = [center.longitude - d, center.latitude + d, center.longitude + d, center.latitude - d].join(",");
      params.bounded = "1";
    }
    let hits = await getJson(`https://nominatim.openstreetmap.org/search?${new URLSearchParams(params)}`);
    if (!hits.length && center) {
      // Widen the box once before giving up.
      const d = 0.4;
      params.viewbox = [center.longitude - d, center.latitude + d, center.longitude + d, center.latitude - d].join(",");
      hits = await getJson(`https://nominatim.openstreetmap.org/search?${new URLSearchParams(params)}`);
    }
    const places = hits.map((h) => {
      const p = { latitude: Number(h.lat), longitude: Number(h.lon) };
      const a = h.address || {};
      const name = h.name || h.display_name.split(",")[0];
      const address = [a.house_number && a.road ? `${a.house_number} ${a.road}` : a.road, a.suburb || a.neighbourhood, a.city || a.town || a.village]
        .filter(Boolean)
        .join(", ") || h.display_name;
      return {
        name,
        type: (h.type || h.category || "").replace(/_/g, " "),
        address,
        latitude: p.latitude,
        longitude: p.longitude,
        distance_km: center ? Math.round(distanceKm(center, p) * 10) / 10 : undefined,
        opening_hours: h.extratags?.opening_hours,
        cuisine: h.extratags?.cuisine?.replace(/;/g, ", "),
        website: h.extratags?.website,
        phone: h.extratags?.phone,
        maps_url: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${name} ${address}`)}`,
      };
    });
    if (center) places.sort((x, y) => x.distance_km - y.distance_km);
    return {
      result: { query, near: center?.name || near, count: places.length, places },
      display: { kind: "places", query, near: center?.name || near, places },
    };
  },

  async get_directions(input, ctx) {
    const resolve = async (s) => {
      if (/^(me|here|my location|current location|my current location)$/i.test(s.trim())) {
        const loc = await userLocation(ctx);
        if (!loc) throw new Error("The user hasn't shared their location; ask where they are starting from.");
        return loc;
      }
      return geocode(s);
    };
    const originQ = str(input.origin, 200);
    const destQ = str(input.destination, 200);
    if (!originQ || !destQ) throw new Error("'origin' and 'destination' are required");
    const mode = ["walking", "cycling"].includes(input.mode) ? input.mode : "driving";
    const [o, d] = await Promise.all([resolve(originQ), resolve(destQ)]);
    const profile = { driving: "routed-car", walking: "routed-foot", cycling: "routed-bike" }[mode];
    const r = await getJson(
      `https://routing.openstreetmap.de/${profile}/route/v1/driving/${o.longitude},${o.latitude};${d.longitude},${d.latitude}?overview=false&steps=true`,
    );
    const route = r.routes?.[0];
    if (!route) throw new Error("No route found between those places");
    const steps = route.legs
      .flatMap((l) => l.steps)
      .map((s) => {
        const m = s.maneuver;
        const verb = m.type === "depart" ? "Head" : m.type === "arrive" ? "Arrive" : m.type === "roundabout" || m.type === "rotary" ? "Take the roundabout" : m.modifier ? `Turn ${m.modifier}` : m.type;
        return {
          instruction: `${verb[0].toUpperCase()}${verb.slice(1)}${s.name ? ` ${m.type === "arrive" ? "at" : "onto"} ${s.name}` : ""}`,
          distance_m: Math.round(s.distance),
        };
      })
      .filter((s, i, all) => s.distance_m > 0 || i === all.length - 1)
      .slice(0, 30);
    const travelmode = { driving: "driving", walking: "walking", cycling: "bicycling" }[mode];
    const mapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(`${o.latitude},${o.longitude}`)}&destination=${encodeURIComponent(`${d.latitude},${d.longitude}`)}&travelmode=${travelmode}`;
    const result = {
      origin: o.name,
      destination: d.name,
      mode,
      distance_km: Math.round(route.distance / 100) / 10,
      duration_min: Math.round(route.duration / 60),
      steps,
      maps_url: mapsUrl,
    };
    return { result, display: { kind: "directions", ...result } };
  },

  async create_file(input) {
    const filename = str(input.filename, 120);
    if (!filename || typeof input.content !== "string") throw new Error("'filename' and 'content' are required");
    const safe = filename.replace(/[\\/:*?"<>|]+/g, "_");
    return {
      result: { delivered: true, filename: safe, bytes: Buffer.byteLength(input.content) },
      display: { kind: "file", filename: safe, description: str(input.description, 300), content: input.content },
    };
  },

  async create_document(input) {
    const format = ["pdf", "docx", "pptx"].includes(input.format) ? input.format : null;
    const title = str(input.title, 200);
    if (!format || !title) throw new Error("'format' (pdf, docx or pptx) and 'title' are required");
    const sections = Array.isArray(input.sections)
      ? input.sections.filter((s) => s && typeof s.body === "string").map((s) => ({ heading: str(s.heading, 200), body: s.body }))
      : [];
    const slides = Array.isArray(input.slides)
      ? input.slides
          .filter((s) => s && typeof s.title === "string")
          .map((s) => ({
            title: s.title.slice(0, 200),
            bullets: Array.isArray(s.bullets) ? s.bullets.filter((b) => typeof b === "string").slice(0, 12) : [],
            notes: str(s.notes, 4000),
          }))
      : [];
    if (format === "pptx" && !slides.length) throw new Error("A pptx deck needs at least one slide in 'slides'");
    if (format !== "pptx" && !sections.length) throw new Error(`A ${format} document needs at least one entry in 'sections'`);
    const filename = (str(input.filename, 100) || title).replace(/[\\/:*?"<>|]+/g, "_").replace(/\.(pdf|docx|pptx)$/i, "");
    const doc = {
      kind: "document",
      format,
      filename: `${filename}.${format}`,
      title,
      subtitle: str(input.subtitle, 300),
      theme: ["midnight", "ocean", "sunset", "forest", "paper"].includes(input.theme) ? input.theme : "midnight",
      sections,
      slides,
    };
    return {
      result: { delivered: true, filename: doc.filename, format, pages_or_slides: format === "pptx" ? slides.length : sections.length },
      display: doc,
    };
  },

  async search_web(input) {
    const query = str(input.query, 300);
    if (!query) throw new Error("'query' is required");
    let results = [];
    try {
      results = await duckDuckGo(query);
    } catch (err) {
      console.warn("[search_web] DuckDuckGo failed:", err.message);
    }
    if (!results.length) results = await wikipediaSearch(query);
    if (!results.length) throw new Error("No search results");
    return { result: { query, results }, display: { kind: "sources", query, results } };
  },

  async read_webpage(input) {
    const url = str(input.url, 2000);
    if (!url || !/^https?:\/\//i.test(url)) throw new Error("'url' must be an http(s) URL");
    const res = await fetch(url, {
      headers: { "user-agent": "Mozilla/5.0 (Linux; Android 14) Mnx/1.0", accept: "text/html,text/plain;q=0.9,*/*;q=0.5" },
      signal: AbortSignal.timeout(15000),
      redirect: "follow",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const type = res.headers.get("content-type") || "";
    if (!/text|html|json|xml/.test(type)) throw new Error(`Can't read ${type || "this kind of"} content`);
    const raw = (await res.text()).slice(0, 2_000_000);
    const rawTitle = raw.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1];
    const title = rawTitle ? decodeEntities(rawTitle.trim()) : url;
    return {
      result: { url, title, text: htmlToText(raw).slice(0, 6000) },
      display: { kind: "sources", query: null, results: [{ title, url }] },
    };
  },

  async create_image(input) {
    const prompt = str(input.prompt, 1500);
    if (!prompt) throw new Error("'prompt' is required");
    const [width, height] = { landscape: [1280, 768], portrait: [768, 1280] }[input.aspect] || [1024, 1024];
    const seed = Math.floor(Math.random() * 1e9);
    // Pollinations: free, key-less image generation. The browser loads the image directly.
    const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=${width}&height=${height}&seed=${seed}&nologo=true`;
    return {
      result: { created: true, shown_to_user: true, prompt },
      display: { kind: "image", prompt, url, width, height },
    };
  },

  async run_code(input, ctx) {
    if (!CODE_RUNNER) throw new Error("The code runner is turned off on this device (MNX_CODE_RUNNER=off).");
    const language = input.language === "javascript" ? "javascript" : "python";
    if (typeof input.code !== "string" || !input.code.trim()) throw new Error("'code' is required");
    if (input.code.length > 100_000) throw new Error("The program is too long (100 KB max).");
    // Never run without an explicit OK from the user.
    const approved = typeof ctx?.requestApproval === "function" && (await ctx.requestApproval({ kind: "run_code", language, code: input.code }));
    if (!approved) {
      return {
        result: { ran: false, reason: "The user chose not to run this code." },
        display: { kind: "code_run", language, code: input.code, declined: true },
      };
    }
    return runCode(language, input.code);
  },
};

/* ───────────── Code runner ───────────── */
function findExe(names) {
  for (const name of names)
    for (const dir of (process.env.PATH || "").split(path.delimiter)) {
      const p = path.join(dir, name);
      try {
        fs.accessSync(p, fs.constants.X_OK);
        return p;
      } catch {
        /* keep looking */
      }
    }
  return null;
}

const IMAGE_MIME = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp" };
const OUTPUT_LIMIT = 20000;

async function runCode(language, code) {
  const exe = language === "python" ? findExe(["python3", "python"]) : process.execPath;
  if (!exe) throw new Error("Python isn't installed on this device. On Termux run: pkg install python");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mnx-run-"));
  const file = path.join(dir, language === "python" ? "main.py" : "main.mjs");
  fs.writeFileSync(file, code);
  const started = Date.now();
  try {
    const run = await new Promise((resolve) => {
      // A minimal environment: no API keys or other secrets from this server.
      const child = spawn(exe, [file], {
        cwd: dir,
        env: {
          PATH: process.env.PATH,
          HOME: process.env.HOME || dir,
          TMPDIR: dir,
          LANG: "C.UTF-8",
          PYTHONIOENCODING: "utf-8",
          PYTHONUNBUFFERED: "1",
          MPLBACKEND: "Agg",
          ...(process.env.PREFIX ? { PREFIX: process.env.PREFIX } : {}),
          ...(process.env.LD_PRELOAD ? { LD_PRELOAD: process.env.LD_PRELOAD } : {}),
        },
        stdio: ["ignore", "pipe", "pipe"],
      });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (d) => stdout.length < OUTPUT_LIMIT && (stdout += d));
      child.stderr.on("data", (d) => stderr.length < OUTPUT_LIMIT && (stderr += d));
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill("SIGKILL");
      }, Number(process.env.MNX_CODE_TIMEOUT_MS) || 30000);
      child.on("close", (exitCode) => {
        clearTimeout(timer);
        resolve({ stdout: stdout.slice(0, OUTPUT_LIMIT), stderr: stderr.slice(0, OUTPUT_LIMIT), exitCode, timedOut });
      });
      child.on("error", (e) => {
        clearTimeout(timer);
        resolve({ stdout: "", stderr: e.message, exitCode: -1, timedOut: false });
      });
    });
    // Show images the program saved (charts etc.).
    const images = [];
    for (const name of fs.readdirSync(dir).sort()) {
      const mime = IMAGE_MIME[path.extname(name).toLowerCase()];
      const full = path.join(dir, name);
      if (!mime || fs.statSync(full).size > 4 * 1048576) continue;
      images.push({ name, src: `data:${mime};base64,${fs.readFileSync(full).toString("base64")}` });
      if (images.length >= 4) break;
    }
    const seconds = Math.round((Date.now() - started) / 100) / 10;
    return {
      result: {
        ran: true,
        exit_code: run.exitCode,
        timed_out: run.timedOut,
        stdout: run.stdout.slice(0, 4000),
        stderr: run.stderr.slice(-2000),
        images_shown_to_user: images.map((i) => i.name),
      },
      display: { kind: "code_run", language, code, ...run, seconds, images },
    };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

/* ───────────── Free web search ───────────── */
function decodeEntities(s) {
  return s
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&quot;/g, '"')
    .replace(/&apos;|&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&");
}
const stripTags = (s) => decodeEntities(s.replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim();

export function htmlToText(html) {
  return decodeEntities(
    html
      .replace(/<(script|style|noscript|svg|nav|footer|header|form)[\s\S]*?<\/\1>/gi, " ")
      .replace(/<\/(p|div|li|h[1-6]|tr|section|article)>/gi, "\n")
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n+/g, "\n")
    .trim();
}

export function parseDuckDuckGo(html) {
  const results = [];
  const re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>([\s\S]*?)(?=<a[^>]+class="result__a"|$)/g;
  let m;
  while ((m = re.exec(html)) && results.length < 6) {
    let url = decodeEntities(m[1]);
    const real = url.match(/[?&]uddg=([^&]+)/);
    if (real) url = decodeURIComponent(real[1]);
    if (url.startsWith("//")) url = `https:${url}`;
    if (/duckduckgo\.com\/y\.js/.test(url)) continue; // ads
    const snip = m[3].match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/);
    results.push({ title: stripTags(m[2]), url, snippet: snip ? stripTags(snip[1]).slice(0, 300) : "" });
  }
  return results;
}

async function duckDuckGo(query) {
  const res = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, {
    headers: { "user-agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Mobile Safari/537.36", accept: "text/html" },
    signal: AbortSignal.timeout(12000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return parseDuckDuckGo(await res.text());
}

async function wikipediaSearch(query) {
  const q = new URLSearchParams({ action: "query", list: "search", srsearch: query, format: "json", srlimit: "5", origin: "*" });
  const data = await getJson(`https://en.wikipedia.org/w/api.php?${q}`);
  return (data.query?.search || []).map((r) => ({
    title: r.title,
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, "_"))}`,
    snippet: stripTags(r.snippet).slice(0, 300),
  }));
}

export async function runTool(name, input, ctx) {
  const handler = handlers[name];
  if (!handler) return { error: `Unknown tool: ${name}` };
  if (!input || typeof input !== "object" || Array.isArray(input)) return { error: "Tool input must be a JSON object" };
  try {
    return await handler(input, ctx);
  } catch (err) {
    return { error: err?.message || String(err) };
  }
}

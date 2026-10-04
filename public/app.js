// Mnx web client: chat UI, live streaming of thinking / tool activity /
// answers, rich result cards, chat history (IndexedDB) and MCP connectors.
(() => {
  "use strict";

  /* ───────────── Utilities ───────────── */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const host = (url) => {
    try {
      return new URL(url).hostname.replace(/^www\./, "");
    } catch {
      return url;
    }
  };
  const fmtBytes = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);

  function toast(msg, ms = 2200) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("show"), ms);
  }

  function download(blob, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied to clipboard");
    } catch {
      toast("Couldn't copy");
    }
  }

  const ICONS = {
    thinking: '<svg viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.4 1 1.1 1 1.8V16h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0 0 12 3z"/></svg>',
    search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    fetch: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>',
    location: '<svg viewBox="0 0 24 24"><path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg>',
    weather: '<svg viewBox="0 0 24 24"><circle cx="9" cy="9" r="3.5"/><path d="M9 2v1.5M9 14.5V16M2 9h1.5M3.9 3.9 5 5M13 5l1.1-1.1"/><path d="M10 20h8a3.5 3.5 0 0 0 0-7 5 5 0 0 0-9.4 1.5A3 3 0 0 0 10 20z"/></svg>',
    places: '<svg viewBox="0 0 24 24"><path d="M3 9l1.5-5h15L21 9"/><path d="M3 9a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0"/><path d="M5 12v8h14v-8M10 20v-5h4v5"/></svg>',
    route: '<svg viewBox="0 0 24 24"><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M8 19h8.5a3.5 3.5 0 0 0 0-7h-9a3.5 3.5 0 0 1 0-7H16"/></svg>',
    file: '<svg viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    doc: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4M7 9h6M7 12h10"/></svg>',
    mcp: '<svg viewBox="0 0 24 24"><path d="M9 7V3M15 7V3M7 7h10v4a5 5 0 0 1-10 0z"/><path d="M12 16v5"/></svg>',
    read: '<svg viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 12h6M9 15h6M9 18h4"/></svg>',
    code: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M13 15h4"/></svg>',
    image: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>',
    tool: '<svg viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z"/></svg>',
  };
  const CHEV = '<svg class="step-chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>';
  const COPY_IC = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
  const DL_IC = '<svg viewBox="0 0 24 24"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"/></svg>';
  const EYE_IC = '<svg viewBox="0 0 24 24"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
  const MAP_IC = '<svg viewBox="0 0 24 24"><path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2z"/><path d="M9 4v14M15 6v14"/></svg>';

  /* ───────────── Storage ───────────── */
  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v ? JSON.parse(v) : fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch {
        /* storage unavailable */
      }
    },
  };

  // Chats live in IndexedDB: histories with attachments outgrow localStorage.
  const db = (() => {
    let dbp = null;
    const open = () =>
      (dbp ||= new Promise((resolve, reject) => {
        const req = indexedDB.open("mnx", 1);
        req.onupgradeneeded = () => req.result.createObjectStore("chats", { keyPath: "id" });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }));
    const tx = async (mode, fn) => {
      try {
        const d = await open();
        return await new Promise((resolve, reject) => {
          const t = d.transaction("chats", mode);
          const r = fn(t.objectStore("chats"));
          t.oncomplete = () => resolve(r?.result);
          t.onerror = () => reject(t.error);
        });
      } catch (e) {
        console.warn("IndexedDB unavailable", e);
        return undefined;
      }
    };
    return {
      put: (chat) => tx("readwrite", (s) => s.put(chat)),
      get: (id) => tx("readonly", (s) => s.get(id)),
      all: async () => (await tx("readonly", (s) => s.getAll())) || [],
      del: (id) => tx("readwrite", (s) => s.delete(id)),
      clear: () => tx("readwrite", (s) => s.clear()),
    };
  })();

  /* ───────────── Settings & theme ───────────── */
  let MODELS = { "claude-opus-5-5": "Opus 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-fable-5-1": "Fable 5.1" };
  const settings = Object.assign({ name: "", model: "claude-opus-5-5", effort: "medium", theme: "system", location: false }, store.get("mnx.settings", {}));
  let mcpServers = store.get("mnx.mcp", []);

  function applyTheme() {
    const root = document.documentElement;
    if (settings.theme === "system") delete root.dataset.theme;
    else root.dataset.theme = settings.theme;
    const dark = settings.theme === "dark" || (settings.theme === "system" && !matchMedia("(prefers-color-scheme: light)").matches);
    $("#hljsDark").disabled = !dark;
    $("#hljsLight").disabled = dark;
  }
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", applyTheme);

  function applySettings() {
    applyTheme();
    $("#helloName").textContent = settings.name || "there";
    $("#profileBtn").textContent = (settings.name || "?").trim().charAt(0).toUpperCase() || "?";
    $("#modelLabel").textContent = MODELS[settings.model] || settings.model;
    $("#modelQuick").value = settings.model;
    $("#locBtn").setAttribute("aria-pressed", String(!!settings.location));
    $("#locBtn").title = settings.location ? "Location sharing on" : "Share location";
  }

  function fillModelSelects() {
    for (const sel of [$("#modelQuick"), $("#modelSelect")]) {
      sel.innerHTML = Object.entries(MODELS)
        .map(([id, label]) => `<option value="${esc(id)}">Mnx · ${esc(label)}</option>`)
        .join("");
    }
    if (!MODELS[settings.model]) settings.model = Object.keys(MODELS)[0];
  }

  function saveSettings() {
    store.set("mnx.settings", settings);
    applySettings();
  }

  /* ───────────── Markdown ───────────── */
  marked.setOptions({ gfm: true, breaks: false });
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.getAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
  function renderMarkdown(target, md, { final = true } = {}) {
    target.innerHTML = DOMPurify.sanitize(marked.parse(md || ""));
    $$("pre code", target).forEach((code) => {
      try {
        if (final || !code.dataset.hl) hljs.highlightElement(code);
      } catch {
        /* unknown language */
      }
      const pre = code.parentElement;
      if (!$(".copy-code", pre)) {
        const b = el("button", "copy-code", "Copy");
        b.type = "button";
        b.onclick = () => copyText(code.innerText);
        pre.appendChild(b);
      }
    });
  }
  const pending = new Map();
  function scheduleRender(key, fn) {
    if (pending.has(key)) {
      pending.set(key, fn);
      return;
    }
    pending.set(key, fn);
    requestAnimationFrame(() => {
      const f = pending.get(key);
      pending.delete(key);
      f();
    });
  }

  /* ───────────── Partial JSON helpers (live tool input) ───────────── */
  function partialString(json, key) {
    const m = new RegExp(`"${key}"\\s*:\\s*"`).exec(json);
    if (!m) return undefined;
    let out = "";
    for (let i = m.index + m[0].length; i < json.length; i++) {
      const c = json[i];
      if (c === '"') return out;
      if (c !== "\\") {
        out += c;
        continue;
      }
      const n = json[i + 1];
      if (n === undefined) break;
      if (n === "u") {
        const hex = json.slice(i + 2, i + 6);
        if (hex.length < 4) break;
        out += String.fromCharCode(parseInt(hex, 16));
        i += 5;
      } else {
        out += { n: "\n", t: "\t", r: "\r", b: "\b", f: "\f" }[n] ?? n;
        i++;
      }
    }
    return out;
  }
  function allStrings(json, key) {
    const re = new RegExp(`"${key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"`, "g");
    const out = [];
    let m;
    while ((m = re.exec(json))) {
      try {
        out.push(JSON.parse(`"${m[1]}"`));
      } catch {
        out.push(m[1]);
      }
    }
    return out;
  }

  /* ───────────── Step labels ───────────── */
  const TOOL_KIND = {
    get_user_location: "location",
    get_weather: "weather",
    find_places: "places",
    get_directions: "route",
    create_file: "file",
    create_document: "doc",
    web_search: "search",
    web_fetch: "fetch",
    search_web: "search",
    read_webpage: "fetch",
    create_image: "image",
    run_code: "code",
  };
  function toolLabel(name, input, phase, display) {
    const i = input || {};
    const done = phase === "done";
    switch (name) {
      case "get_user_location":
        return done ? (display?.available ? `Found your location · ${display.name}` : "Location not shared") : "Finding your location";
      case "get_weather": {
        const where = display?.location || i.location;
        return done ? `Got the weather${where ? ` for ${where.split(",")[0]}` : ""}` : `Checking the weather${where ? ` in ${where}` : ""}`;
      }
      case "find_places":
        return done
          ? `Found ${display?.places?.length ?? 0} places${i.query ? ` for “${i.query}”` : ""}`
          : `Finding ${i.query ? `“${i.query}”` : "places"}${i.near ? ` near ${i.near}` : ""}`;
      case "get_directions":
        return done ? `Mapped a route to ${(display?.destination || i.destination || "").split(",")[0]}` : `Mapping a route${i.destination ? ` to ${i.destination}` : ""}`;
      case "create_file":
        return done ? `Created ${display?.filename || i.filename || "file"}` : `Writing ${i.filename || "a file"}`;
      case "create_document": {
        const fmt = (i.format || "document").toUpperCase();
        return done ? `Created ${display?.filename || fmt}` : `Building ${i.format === "pptx" ? "slides" : fmt}${i.title ? ` · ${i.title}` : ""}`;
      }
      case "search_web":
        return done ? `Searched “${i.query || "the web"}” · ${display?.results?.length ?? 0} results` : `Searching the web${i.query ? ` · “${i.query}”` : ""}`;
      case "read_webpage":
        return done ? `Read ${display?.results?.[0]?.title || host(i.url || "")}` : `Reading ${i.url ? host(i.url) : "a web page"}`;
      case "run_code": {
        const lang = i.language === "javascript" ? "JavaScript" : "Python";
        if (!done) return `Running ${lang} code`;
        if (display?.declined) return "Didn't run the code (you said no)";
        if (display?.timedOut) return `${lang} code timed out`;
        return display?.exitCode === 0 ? `Ran ${lang} code · ${display.seconds}s` : `${lang} code failed (exit ${display?.exitCode})`;
      }
      case "create_image":
        return done ? "Created an image" : `Painting${i.prompt ? ` “${i.prompt.length > 48 ? `${i.prompt.slice(0, 48)}…` : i.prompt}”` : " an image"}`;
      case "web_search":
        return done ? `Searched${i.query ? ` “${i.query}”` : " the web"}` : `Searching the web${i.query ? ` · “${i.query}”` : ""}`;
      case "web_fetch":
        return done ? `Read ${i.url ? host(i.url) : "page"}` : `Reading ${i.url ? host(i.url) : "a web page"}`;
      default:
        return done ? `Used ${name}` : `Using ${name}`;
    }
  }
  function thinkingHeading(text) {
    const heads = [...String(text).matchAll(/(?:^|\n)\s*\*\*([^*\n]{3,90})\*\*/g)];
    return heads.length ? heads[heads.length - 1][1].trim() : null;
  }

  /* ───────────── Loading phrases ───────────── */
  const PHRASES_START = [
    "Warming up the neurons",
    "Understanding your request",
    "Untangling the question",
    "Consulting the knowledge orbit",
    "Connecting the dots",
    "Lining up the facts",
    "Sharpening the pencil",
    "Brewing a fresh answer",
  ];
  const PHRASES_AFTER_TOOL = ["Reading the results", "Making sense of it all", "Cross-checking the details", "Piecing it together", "Polishing the answer"];

  /* ───────────── Weather glyphs ───────────── */
  function wxIcon(kind, cls = "w-icon") {
    const cloud = (dark = false, x = 0, y = 0, s = 1) =>
      `<g class="wx-cloud ${dark ? "dark" : ""} wx-drift" transform="translate(${x} ${y}) scale(${s})"><path d="M30 78h44a16 16 0 0 0 0-32 22 22 0 0 0-42 6 13 13 0 0 0-2 26z"/></g>`;
    const sun = (x = 0, y = 0, s = 1) =>
      `<g class="wx-sun" transform="translate(${x} ${y}) scale(${s})"><g class="rays">${[...Array(8)]
        .map((_, k) => {
          const a = (k * Math.PI) / 4;
          return `<line x1="${50 + Math.cos(a) * 28}" y1="${50 + Math.sin(a) * 28}" x2="${50 + Math.cos(a) * 38}" y2="${50 + Math.sin(a) * 38}"/>`;
        })
        .join("")}</g><circle class="core" cx="50" cy="50" r="19"/></g>`;
    const drops = (cls2, n) =>
      [...Array(n)].map((_, k) => (cls2 === "wx-flake" ? `<circle class="wx-flake" cx="${34 + k * 14}" cy="86" r="3.5" style="animation-delay:${k * 0.35}s"/>` : `<line class="wx-drop" x1="${36 + k * 13}" y1="82" x2="${32 + k * 13}" y2="92" style="animation-delay:${k * 0.25}s"/>`)).join("");
    let inner;
    switch (kind) {
      case "sun": inner = sun(); break;
      case "moon": inner = `<g class="wx-moon"><path d="M62 18a32 32 0 1 0 22 50A26 26 0 0 1 62 18z"/></g>`; break;
      case "partly": inner = sun(-14, -14, 0.85) + cloud(false, 6, 6, 0.95); break;
      case "cloud": inner = cloud(true, -12, -10, 0.8) + cloud(false, 4, 4); break;
      case "fog": inner = cloud(true, 0, -14) + `<g class="wx-fog"><line x1="22" y1="78" x2="78" y2="78"/><line x1="30" y1="90" x2="86" y2="90" style="animation-delay:.6s"/></g>`; break;
      case "drizzle": inner = cloud(false, 0, -12) + drops("wx-drop", 3); break;
      case "rain": inner = cloud(true, 0, -12) + drops("wx-drop", 5); break;
      case "snow": inner = cloud(false, 0, -12) + drops("wx-flake", 4); break;
      case "storm": inner = cloud(true, 0, -14) + `<path class="wx-bolt" d="M52 64 40 84h10l-4 14 16-22H52l6-12z"/>`; break;
      default: inner = cloud();
    }
    return `<svg class="${cls}" viewBox="0 0 100 100" aria-hidden="true">${inner}</svg>`;
  }

  /* ───────────── Cards ───────────── */
  const EXT_MIME = {
    html: "text/html", htm: "text/html", css: "text/css", js: "text/javascript", mjs: "text/javascript", json: "application/json",
    svg: "image/svg+xml", md: "text/markdown", csv: "text/csv", xml: "application/xml", py: "text/x-python",
  };
  const EXT_LANG = {
    py: "python", js: "javascript", mjs: "javascript", ts: "typescript", tsx: "typescript", jsx: "javascript", html: "xml", htm: "xml",
    svg: "xml", xml: "xml", rb: "ruby", c: "c", h: "c", cpp: "cpp", cc: "cpp", hpp: "cpp", lua: "lua", json: "json", md: "markdown",
    css: "css", scss: "scss", sh: "bash", bash: "bash", go: "go", rs: "rust", java: "java", kt: "kotlin", yml: "yaml", yaml: "yaml",
    sql: "sql", php: "php", swift: "swift", cs: "csharp", r: "r", pl: "perl", ini: "ini", toml: "ini",
  };

  function cardWeather(d) {
    const c = el("div", "card weather");
    const day = (iso) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { weekday: "short" });
    c.innerHTML = `
      <div class="w-main">
        ${wxIcon(d.icon)}
        <div>
          <div class="w-temp">${esc(d.temperature)}<sup>${esc(d.unit)}</sup></div>
        </div>
        <div>
          <div class="w-cond">${esc(d.condition)}</div>
          <div class="w-loc">${esc(d.location)}</div>
        </div>
      </div>
      <div class="w-stats">
        <span>Feels like <b>${esc(d.feels_like)}${esc(d.unit)}</b></span>
        <span>Humidity <b>${esc(d.humidity)}%</b></span>
        <span>Wind <b>${esc(d.wind)}</b></span>
        ${d.sunrise ? `<span>Sunrise <b>${esc(d.sunrise.slice(11))}</b></span>` : ""}
        ${d.sunset ? `<span>Sunset <b>${esc(d.sunset.slice(11))}</b></span>` : ""}
      </div>
      <div class="w-days">${(d.daily || [])
        .map(
          (x, k) => `<div class="w-day" style="animation-delay:${k * 70}ms" title="${esc(x.condition)}">
            <div>${k === 0 ? "Today" : esc(day(x.date))}</div>${wxIcon(x.icon, "")}
            <div><span class="hi">${esc(x.max)}°</span> <span class="lo">${esc(x.min)}°</span></div>
            ${x.rain_chance != null ? `<div class="lo">💧${esc(x.rain_chance)}%</div>` : ""}
          </div>`,
        )
        .join("")}</div>`;
    return c;
  }

  function osmEmbed(points) {
    if (!points.length) return "";
    const lats = points.map((p) => p.latitude);
    const lons = points.map((p) => p.longitude);
    const pad = 0.01;
    const bbox = [Math.min(...lons) - pad, Math.min(...lats) - pad, Math.max(...lons) + pad, Math.max(...lats) + pad].join(",");
    const m = points[0];
    return `<iframe class="map-embed" loading="lazy" title="Map" src="https://www.openstreetmap.org/export/embed.html?bbox=${encodeURIComponent(bbox)}&layer=mapnik&marker=${m.latitude},${m.longitude}"></iframe>`;
  }

  function cardPlaces(d) {
    const c = el("div", "card");
    const list = d.places || [];
    c.innerHTML = `
      <div class="card-head"><div class="step-ic" style="position:static">${ICONS.places}</div>
        <div><div class="card-title">${esc(d.query)}</div><div class="card-sub">${list.length} result${list.length === 1 ? "" : "s"}${d.near ? ` near ${esc(d.near)}` : ""}</div></div></div>
      ${list.length ? osmEmbed(list) : ""}
      <div class="places">${list
        .map(
          (p, k) => `<div class="place" style="animation-delay:${k * 60}ms"><div class="place-n">${k + 1}</div><div class="place-info">
            <div class="place-name">${esc(p.name)}</div>
            <div class="place-meta">${[p.type, p.cuisine, p.distance_km != null ? `${p.distance_km} km` : null].filter(Boolean).map(esc).join(" · ")}</div>
            <div class="place-meta">${esc(p.address)}</div>
            ${p.opening_hours ? `<div class="place-meta">🕒 ${esc(p.opening_hours)}</div>` : ""}
            <div class="place-links">
              <a href="${esc(p.maps_url)}" target="_blank" rel="noopener">Open in Google Maps</a>
              <a href="https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(`${p.latitude},${p.longitude}`)}" target="_blank" rel="noopener">Directions</a>
              ${p.website && /^https?:/i.test(p.website) ? `<a href="${esc(p.website)}" target="_blank" rel="noopener">Website</a>` : ""}
              ${p.phone ? `<a href="tel:${esc(p.phone)}">Call</a>` : ""}
            </div></div></div>`,
        )
        .join("")}</div>`;
    return c;
  }

  function cardDirections(d) {
    const c = el("div", "card");
    const h = Math.floor(d.duration_min / 60);
    const mins = d.duration_min % 60;
    const mode = { driving: "🚗 Driving", walking: "🚶 Walking", cycling: "🚲 Cycling" }[d.mode] || d.mode;
    c.innerHTML = `
      <div class="card-head"><div class="step-ic" style="position:static">${ICONS.route}</div>
        <div><div class="card-title">${esc(d.destination.split(",")[0])}</div><div class="card-sub">${esc(mode)}</div></div></div>
      <div class="route-line"><span>${esc(d.origin.split(",")[0])}</span><span class="dash"></span><span>${esc(d.destination.split(",")[0])}</span></div>
      <div class="route-stats">
        <div class="route-stat"><b>${h ? `${h} h ` : ""}${mins} min</b><span>Travel time</span></div>
        <div class="route-stat"><b>${esc(d.distance_km)} km</b><span>Distance</span></div>
      </div>
      <details class="card-more"><summary>Show ${d.steps.length} steps</summary>
        <ol class="route-steps">${d.steps.map((s) => `<li>${esc(s.instruction)} <em>${s.distance_m >= 1000 ? `${(s.distance_m / 1000).toFixed(1)} km` : `${s.distance_m} m`}</em></li>`).join("")}</ol>
      </details>
      <div class="card-actions"><a class="btn primary" href="${esc(d.maps_url)}" target="_blank" rel="noopener">${MAP_IC} Open in Google Maps</a></div>`;
    return c;
  }

  function cardLocation(d) {
    const c = el("div", "card");
    c.innerHTML = d.available
      ? `<div class="loc-card"><span class="loc-pulse"></span><span>You're near <b>${esc(d.name)}</b></span></div>`
      : `<div class="loc-card"><span>📍 Location isn't shared. Turn it on with the pin button next to the message box.</span></div>`;
    return c;
  }

  function cardFile(d) {
    const ext = (d.filename.split(".").pop() || "txt").toLowerCase();
    const lines = d.content.split("\n").length;
    const c = el("div", "card");
    c.innerHTML = `
      <div class="card-head"><div class="file-ic">${esc(ext.slice(0, 4))}</div>
        <div style="min-width:0"><div class="card-title">${esc(d.filename)}</div>
        <div class="card-sub">${d.description ? `${esc(d.description)} · ` : ""}${lines} lines · ${fmtBytes(new Blob([d.content]).size)}</div></div></div>
      <div class="card-actions">
        <button class="btn" data-act="copy">${COPY_IC} Copy</button>
        <button class="btn primary" data-act="dl">${DL_IC} Download</button>
        ${["html", "htm", "svg"].includes(ext) ? `<button class="btn" data-act="preview">${EYE_IC} Preview</button>` : ""}
      </div>
      <div class="file-code"><pre><code></code></pre></div>`;
    const code = $("code", c);
    code.textContent = d.content;
    const lang = EXT_LANG[ext];
    if (lang && hljs.getLanguage(lang)) code.innerHTML = hljs.highlight(d.content, { language: lang }).value;
    code.classList.add("hljs");
    c.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "copy") copyText(d.content);
      if (act === "dl") download(new Blob([d.content], { type: EXT_MIME[ext] || "text/plain" }), d.filename);
      if (act === "preview") {
        $("#previewTitle").textContent = d.filename;
        $("#previewFrame").srcdoc = d.content;
        $("#previewDialog").showModal();
      }
    });
    return c;
  }

  function cardDocument(d) {
    const c = el("div", "card");
    const count = d.format === "pptx" ? `${d.slides.length} slides + title` : `${d.sections.length} sections`;
    c.innerHTML = `
      <div class="card-head"><div class="file-ic ${esc(d.format)}">${esc(d.format)}</div>
        <div style="min-width:0"><div class="card-title">${esc(d.title)}</div><div class="card-sub">${esc(d.filename)} · ${count}</div></div></div>`;
    c.appendChild(MnxDocs.preview(d));
    const actions = el("div", "card-actions", `<button class="btn primary" data-act="dl">${DL_IC} Download ${esc(d.format.toUpperCase())}</button>`);
    c.appendChild(actions);
    actions.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      btn.disabled = true;
      const old = btn.innerHTML;
      btn.textContent = "Building…";
      try {
        download(await MnxDocs.build(d), d.filename);
      } catch (err) {
        console.error(err);
        toast(`Couldn't build the file: ${err.message}`, 4000);
      } finally {
        btn.disabled = false;
        btn.innerHTML = old;
      }
    });
    return c;
  }

  function cardSources(d) {
    const c = el("div", "card");
    const list = d.results || [];
    c.innerHTML = `${d.query ? `<div class="card-head"><div class="step-ic" style="position:static">${ICONS.search}</div>
        <div><div class="card-title">${esc(d.query)}</div><div class="card-sub">${list.length} result${list.length === 1 ? "" : "s"}</div></div></div>` : ""}
      <div class="places">${list
        .map(
          (r, k) => `<div class="place" style="animation-delay:${k * 50}ms"><div class="place-n">${k + 1}</div><div class="place-info">
            <a class="place-name" href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title || r.url)}</a>
            <div class="place-meta">${esc(host(r.url))}</div>
            ${r.snippet ? `<div class="place-meta">${esc(r.snippet)}</div>` : ""}</div></div>`,
        )
        .join("")}</div>`;
    return c;
  }

  function cardImage(d) {
    const c = el("div", "card image-card");
    c.innerHTML = `<div class="img-wrap" style="aspect-ratio:${d.width}/${d.height}"><div class="img-shimmer"><span class="orb"><i></i><i></i><i></i></span><span>Painting your image…</span></div><img alt="${esc(d.prompt)}" /></div>
      <div class="card-head"><div style="min-width:0"><div class="card-sub">${esc(d.prompt)}</div></div></div>
      <div class="card-actions"><a class="btn primary" href="${esc(d.url)}" target="_blank" rel="noopener" download="mnx-image.jpg">${DL_IC} Open / save</a></div>`;
    const img = $("img", c);
    img.onload = () => c.classList.add("loaded");
    img.onerror = () => ($(".img-shimmer span:last-child", c).textContent = "Couldn't load the image — check your connection.");
    img.src = d.url;
    return c;
  }

  function codeBlock(code, language) {
    const pre = el("pre", "");
    const c = el("code", "hljs");
    const lang = language === "javascript" ? "javascript" : "python";
    c.innerHTML = hljs.getLanguage(lang) ? hljs.highlight(code, { language: lang }).value : esc(code);
    pre.appendChild(c);
    return pre;
  }

  function cardCodeRun(d) {
    const c = el("div", "card code-run");
    const lang = d.language === "javascript" ? "JavaScript" : "Python";
    if (d.declined) {
      c.innerHTML = `<div class="card-head"><div class="file-ic code">${d.language === "javascript" ? "JS" : "PY"}</div>
        <div><div class="card-title">${lang} code not run</div><div class="card-sub">You chose not to run it.</div></div></div>`;
      return c;
    }
    const status = d.timedOut ? ["timeout", "Timed out"] : d.exitCode === 0 ? ["ok", "Success"] : ["err", `Exit ${d.exitCode}`];
    c.innerHTML = `
      <div class="card-head"><div class="file-ic code">${d.language === "javascript" ? "JS" : "PY"}</div>
        <div style="flex:1;min-width:0"><div class="card-title">${lang} output</div><div class="card-sub">Ran in ${esc(d.seconds)}s on this device</div></div>
        <span class="badge ${status[0]}">${status[1]}</span></div>
      <div class="term"></div>
      <div class="run-images"></div>
      <details class="card-more"><summary>Show code</summary><div class="file-code"></div></details>
      <div class="card-actions"><button class="btn" data-act="out">${COPY_IC} Copy output</button><button class="btn" data-act="code">${COPY_IC} Copy code</button></div>`;
    const term = $(".term", c);
    if (d.stdout) term.appendChild(document.createTextNode(d.stdout));
    if (d.stderr) {
      const e = el("span", "err");
      e.textContent = (d.stdout && !d.stdout.endsWith("\n") ? "\n" : "") + d.stderr;
      term.appendChild(e);
    }
    if (!d.stdout && !d.stderr) term.appendChild(el("span", "muted", "(no output)"));
    for (const img of d.images || []) {
      const fig = el("figure", "");
      fig.innerHTML = `<img alt="${esc(img.name)}"><figcaption>${esc(img.name)}</figcaption>`;
      $("img", fig).src = img.src;
      $(".run-images", c).appendChild(fig);
    }
    $(".file-code", c).appendChild(codeBlock(d.code, d.language));
    c.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "out") copyText([d.stdout, d.stderr].filter(Boolean).join("\n"));
      if (act === "code") copyText(d.code);
    });
    return c;
  }

  function cardApproval(part, live) {
    const c = el("div", "card approval");
    const lang = part.language === "javascript" ? "JavaScript" : "Python";
    c.innerHTML = `
      <div class="card-head"><div class="file-ic code">${part.language === "javascript" ? "JS" : "PY"}</div>
        <div style="min-width:0"><div class="card-title">Run this ${lang} code?</div>
        <div class="card-sub">It runs on this device with access to your files and the internet. Check it first.</div></div></div>
      <div class="file-code"></div>
      <div class="card-actions approval-actions">
        <button class="btn primary" data-a="run"><svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg> Run</button>
        <button class="btn" data-a="skip">Don't run</button>
      </div>
      <div class="approval-state"></div>`;
    $(".file-code", c).appendChild(codeBlock(part.code, part.language));
    const paint = () => {
      const st = part.state === "pending" && !live ? "expired" : part.state;
      c.dataset.state = st;
      $(".approval-state", c).textContent =
        { approved: "✓ Approved — running on your device", declined: "✕ Not run", expired: "This request expired", sending: "Sending…" }[st] || "";
    };
    c.addEventListener("click", async (e) => {
      const a = e.target.closest("[data-a]")?.dataset.a;
      if (!a || part.state !== "pending") return;
      part.state = "sending";
      paint();
      try {
        const r = await fetch("/api/approve", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ id: part.id, token: part.token, approve: a === "run" }),
        });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
        part.state = a === "run" ? "approved" : "declined";
      } catch (err) {
        part.state = "expired";
        toast(err.message);
      }
      delete part.token;
      paint();
    });
    part._paint = paint;
    paint();
    return c;
  }

  function renderCard(d) {
    switch (d?.kind) {
      case "code_run": return cardCodeRun(d);
      case "sources": return cardSources(d);
      case "image": return cardImage(d);
      case "weather": return cardWeather(d);
      case "places": return cardPlaces(d);
      case "directions": return cardDirections(d);
      case "location": return cardLocation(d);
      case "file": return cardFile(d);
      case "document": return cardDocument(d);
      default: return null;
    }
  }

  /* ───────────── Chat state ───────────── */
  const thread = $("#thread");
  const scroller = $("#scroll");
  const main = $("#main");
  let chat = null; // { id, title, created, updated, api: [], view: [] }
  let busy = false;
  let controller = null;

  const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140;
  let stick = true;
  scroller.addEventListener("scroll", () => (stick = nearBottom()), { passive: true });
  const follow = () => {
    if (stick) scroller.scrollTop = scroller.scrollHeight;
  };

  function newChat() {
    if (busy) stop();
    chat = null;
    thread.innerHTML = "";
    main.classList.add("empty");
    $("#chatTitle").textContent = "New chat";
    highlightHistory();
    $("#input").focus();
  }

  async function saveChat() {
    if (!chat) return;
    chat.updated = Date.now();
    // Strip runtime-only fields (DOM refs are prefixed with "_").
    await db.put(JSON.parse(JSON.stringify(chat, (k, v) => (k.startsWith("_") || k === "token" ? undefined : v))));
    refreshHistory();
  }

  /* ───────────── Rendering turns ───────────── */
  function renderUserTurn(turn) {
    const m = el("div", "msg user");
    const col = el("div", "col");
    if (turn.files?.length) {
      const f = el("div", "files");
      for (const file of turn.files)
        f.appendChild(el("span", "att", `${file.thumb ? `<img src="${esc(file.thumb)}" alt="">` : "📄"}<span class="att-name">${esc(file.name)}</span>`));
      col.appendChild(f);
    }
    if (turn.text) {
      const b = el("div", "bubble");
      b.textContent = turn.text;
      col.appendChild(b);
    }
    m.appendChild(col);
    thread.appendChild(m);
    return m;
  }

  class AssistantView {
    constructor(turn, live) {
      this.turn = turn;
      this.live = live;
      this.root = el("div", `msg assistant${live ? " live" : ""}`);
      this.root.innerHTML = `<div class="mnx-badge"><span class="wordmark">Mn<span class="x">x</span></span></div>`;
      this.body = el("div", "body");
      this.parts = el("div", "body");
      this.parts.style.padding = "0";
      this.loader = el("div", "loader", `<span class="orb"><i></i><i></i><i></i></span><span class="phrase"></span>`);
      this.loader.hidden = true;
      this.actions = el("div", "msg-actions", `<button class="icon-btn" title="Copy answer" data-act="copy">${COPY_IC}</button>`);
      this.actions.addEventListener("click", () => {
        const text = this.turn.parts.filter((p) => p.type === "text").map((p) => p.md).join("\n\n");
        copyText(text);
      });
      this.body.append(this.parts, this.loader, this.actions);
      this.root.appendChild(this.body);
      thread.appendChild(this.root);
      this.stepEls = new Map();
      for (const part of turn.parts) this.mountPart(part);
    }

    mountPart(part) {
      let node;
      if (part.type === "steps") {
        node = el("div", "steps");
        for (const s of part.steps) node.appendChild(this.stepEl(s));
      } else if (part.type === "text") {
        node = el("div", "md");
        renderMarkdown(node, part.md);
      } else if (part.type === "card") {
        node = renderCard(part.display) || el("div");
      } else if (part.type === "approval") {
        node = cardApproval(part, this.live);
      } else if (part.type === "notice") {
        node = el("div", `notice ${part.level || ""}`);
        node.textContent = part.text;
      }
      part._el = node;
      this.parts.appendChild(node);
      return node;
    }

    addPart(part) {
      this.turn.parts.push(part);
      return this.mountPart(part);
    }

    lastSteps() {
      const last = this.turn.parts[this.turn.parts.length - 1];
      if (last?.type === "steps") return last;
      const part = { type: "steps", steps: [] };
      this.addPart(part);
      return part;
    }

    addStep(step) {
      const part = this.lastSteps();
      part.steps.push(step);
      const node = this.stepEl(step);
      part._el.appendChild(node);
      follow();
      return step;
    }

    stepEl(step) {
      const node = el("div", "step");
      node.innerHTML = `<div class="step-ic">${ICONS[step.kind] || ICONS.tool}</div>
        <button class="step-head" type="button"><span class="step-label"></span><span class="step-meta"></span>${CHEV}</button>
        <div class="step-body"><div><div class="step-inner"></div></div></div>`;
      $(".step-head", node).addEventListener("click", () => {
        if (!node.classList.contains("has-body")) return;
        step.open = !node.classList.contains("open");
        node.classList.toggle("open", step.open);
      });
      this.stepEls.set(step, node);
      this.paintStep(step);
      return node;
    }

    paintStep(step) {
      const node = this.stepEls.get(step);
      if (!node) return;
      node.className = `step ${step.status || "done"}`;
      $(".step-label", node).textContent = step.label;
      $(".step-meta", node).textContent = step.meta || "";
      const inner = $(".step-inner", node);
      const b = step.body;
      const hasBody = !!b && ((b.text && b.text.trim()) || b.items?.length);
      node.classList.toggle("has-body", !!hasBody);
      node.classList.toggle("open", !!hasBody && !!step.open);
      if (!hasBody) return;
      if (b.type === "thinking" || b.type === "markdown") {
        let box = $(".thinking-text", inner);
        if (!box) {
          inner.innerHTML = "";
          box = inner.appendChild(el("div", "thinking-text md"));
        }
        renderMarkdown(box, b.text, { final: step.status !== "running" });
        if (step.status === "running") box.scrollTop = box.scrollHeight;
      } else if (b.type === "code") {
        let pre = $(".live-code", inner);
        if (!pre) {
          inner.innerHTML = "";
          pre = inner.appendChild(el("pre", "live-code"));
        }
        pre.textContent = b.text;
        pre.scrollTop = pre.scrollHeight;
      } else if (b.type === "sources") {
        inner.innerHTML = `<div class="sources">${b.items
          .map((r, k) => `<a class="source" style="animation-delay:${k * 50}ms" href="${esc(r.url)}" target="_blank" rel="noopener"><b>${esc(host(r.url))}</b><span>${esc(r.title || "")}</span></a>`)
          .join("")}</div>`;
      } else if (b.type === "outline") {
        inner.innerHTML = `<ol class="outline">${b.items.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>`;
      } else if (b.type === "text") {
        inner.textContent = b.text;
      }
    }

    updateStep(step, patch) {
      Object.assign(step, patch);
      scheduleRender(step, () => {
        this.paintStep(step);
        follow();
      });
    }
  }

  function renderChat(c) {
    thread.innerHTML = "";
    for (const turn of c.view) {
      if (turn.role === "user") renderUserTurn(turn);
      else {
        // Steps that were still running when the page closed are finished now.
        for (const part of turn.parts)
          if (part.type === "steps") for (const st of part.steps) if (st.status === "running") Object.assign(st, { status: "done", open: false });
        new AssistantView(turn, false);
      }
    }
    main.classList.toggle("empty", !c.view.length);
    $("#chatTitle").textContent = c.title || "Chat";
    requestAnimationFrame(() => (scroller.scrollTop = scroller.scrollHeight));
  }

  /* ───────────── Sending & streaming ───────────── */
  let attachments = [];

  async function getLocation() {
    if (!settings.location || !navigator.geolocation) return null;
    const cached = getLocation.cache;
    if (cached && Date.now() - cached.at < 10 * 60 * 1000) return cached.loc;
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (p) => {
          const loc = { latitude: p.coords.latitude, longitude: p.coords.longitude, accuracy_m: Math.round(p.coords.accuracy) };
          getLocation.cache = { at: Date.now(), loc };
          resolve(loc);
        },
        () => resolve(null),
        { timeout: 8000, maximumAge: 600000 },
      );
    });
  }

  function buildUserContent(text, files) {
    const content = [];
    for (const f of files) {
      if (f.kind === "image") content.push({ type: "image", source: { type: "base64", media_type: f.type, data: f.data } });
      else if (f.kind === "pdf") content.push({ type: "document", source: { type: "base64", media_type: "application/pdf", data: f.data }, title: f.name });
      else content.push({ type: "text", text: `<file name="${f.name}">\n${f.text}\n</file>` });
    }
    content.push({ type: "text", text: text || "Please look at the attached file(s)." });
    return content;
  }

  // If a run stopped between a tool call and its result, answer the dangling
  // tool calls so the conversation stays valid for the next turn.
  function closeDanglingTools() {
    const last = chat?.api[chat.api.length - 1];
    if (last?.role !== "assistant" || !Array.isArray(last.content)) return;
    const uses = last.content.filter((b) => b.type === "tool_use");
    if (!uses.length) return;
    chat.api.push({
      role: "user",
      content: uses.map((u) => ({ type: "tool_result", tool_use_id: u.id, is_error: true, content: "Cancelled by the user before the tool ran." })),
    });
  }

  function setBusy(v) {
    busy = v;
    main.classList.toggle("busy", v);
    updateSendState();
  }

  function stop() {
    controller?.abort();
  }

  async function send(text) {
    text = (text || "").trim();
    if (busy) return;
    if (!text && !attachments.length) return;

    const files = attachments;
    attachments = [];
    renderAttachments();
    $("#input").value = "";
    autosize();

    if (!chat) {
      chat = { id: uid(), title: (text || files[0]?.name || "New chat").slice(0, 70), created: Date.now(), updated: Date.now(), api: [], view: [] };
      $("#chatTitle").textContent = chat.title;
    }
    main.classList.remove("empty");
    closeDanglingTools();

    const userTurn = { role: "user", text, files: files.map((f) => ({ name: f.name, kind: f.kind, thumb: f.kind === "image" && f.size < 400000 ? `data:${f.type};base64,${f.data}` : null })) };
    chat.view.push(userTurn);
    chat.api.push({ role: "user", content: buildUserContent(text, files) });
    renderUserTurn(userTurn);
    saveChat(); // save right away so a reload or closed tab never loses the message
    stick = true;

    const turn = { role: "assistant", parts: [] };
    chat.view.push(turn);
    const view = new AssistantView(turn, true);
    follow();

    // Show which files are being read, as in the sketch's "Reading file" step.
    const readSteps = files.map((f) => view.addStep({ kind: "read", label: `Reading ${f.name}`, status: "running", meta: fmtBytes(f.size) }));

    setBusy(true);
    controller = new AbortController();
    const run = new RunState(view, readSteps);
    run.startPhrases(PHRASES_START);

    try {
      const location = await getLocation();
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          messages: chat.api,
          model: settings.model,
          effort: settings.effort,
          userName: settings.name,
          location,
          mcpServers: mcpServers.filter((s) => s.enabled).map(({ name, url, token }) => ({ name, url, token })),
        }),
      });
      if (!res.ok || !res.body) {
        let msg = `Server error (${res.status})`;
        try {
          msg = (await res.json()).error || msg;
        } catch {
          /* not JSON */
        }
        throw new Error(msg);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of chunk.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            let ev;
            try {
              ev = JSON.parse(line.slice(6));
            } catch {
              continue;
            }
            run.handle(ev);
          }
        }
      }
      if (!run.finished) run.finish();
    } catch (err) {
      if (err.name === "AbortError") {
        run.finish();
        view.addPart({ type: "notice", text: "Stopped." });
      } else {
        run.finish();
        view.addPart({ type: "notice", level: "error", text: navigator.onLine ? err.message : "You're offline." });
      }
    } finally {
      closeDanglingTools();
      view.root.classList.remove("live");
      setBusy(false);
      controller = null;
      await saveChat();
      $("#input").focus();
    }
  }

  // Applies streamed server events to an AssistantView.
  class RunState {
    constructor(view, readSteps) {
      this.view = view;
      this.blocks = new Map(); // "it:i" -> block state
      this.toolSteps = new Map(); // tool_use id -> step
      this.readSteps = readSteps;
      this.active = 0; // running steps + streaming texts
      this.finished = false;
      this.phraseTimer = null;
      this.phrases = [];
    }

    startPhrases(list) {
      this.phrases = list;
      this.phraseIdx = Math.floor(Math.random() * 2);
      this.setPhrase();
      clearInterval(this.phraseTimer);
      this.phraseTimer = setInterval(() => this.setPhrase(), 2400);
      this.syncLoader();
    }
    setPhrase() {
      const span = $(".phrase", this.view.loader);
      const next = el("span", "phrase");
      next.textContent = this.phrases[this.phraseIdx++ % this.phrases.length];
      span.replaceWith(next);
    }
    syncLoader() {
      const show = !this.finished && this.active <= 0;
      this.view.loader.hidden = !show;
      if (show) this.view.body.insertBefore(this.view.loader, this.view.actions);
      follow();
    }

    finishReads() {
      for (const s of this.readSteps) this.view.updateStep(s, { status: "done", label: s.label.replace(/^Reading/, "Read") });
      this.readSteps = [];
    }

    handle(ev) {
      if (this.readSteps.length && ev.t !== "iteration") this.finishReads();
      const key = `${ev.it}:${ev.i}`;
      switch (ev.t) {
        case "block_start": return this.blockStart(key, ev.block);
        case "thinking": {
          const b = this.blocks.get(key);
          if (!b) return;
          b.step.body.text += ev.text;
          const heading = thinkingHeading(b.step.body.text);
          this.view.updateStep(b.step, { label: heading || "Thinking" });
          return;
        }
        case "text": {
          const b = this.blocks.get(key);
          if (!b) return;
          b.part.md += ev.text;
          scheduleRender(b.part, () => {
            renderMarkdown(b.part._el, b.part.md, { final: false });
            follow();
          });
          return;
        }
        case "input": {
          const b = this.blocks.get(key);
          if (!b?.step) return;
          b.json += ev.json;
          this.liveInput(b);
          return;
        }
        case "block_stop": return this.blockStop(key);
        case "tool_run": {
          const step = this.toolSteps.get(ev.id);
          if (step) this.view.updateStep(step, { status: "running", label: toolLabel(ev.name, ev.input, "running"), input: ev.input });
          return;
        }
        case "tool_done": {
          saveChat();
          const step = this.toolSteps.get(ev.id);
          if (step) {
            if (step.status === "running") this.active--;
            const patch = { status: ev.error ? "error" : "done", label: ev.error ? `${toolLabel(ev.name, step.input, "running")} — failed` : toolLabel(ev.name, step.input, "done", ev.display) };
            if (["create_file", "create_document"].includes(ev.name)) patch.open = false;
            this.view.updateStep(step, patch);
          }
          if (ev.display) {
            this.view.addPart({ type: "card", display: ev.display });
            follow();
          }
          this.startPhrases(PHRASES_AFTER_TOOL);
          return;
        }
        case "assistant_turn":
          chat.api.push({ role: "assistant", content: ev.content });
          return;
        case "user_turn":
          chat.api.push({ role: "user", content: ev.content });
          saveChat();
          return;
        case "approval_request": {
          setTimeout(saveChat, 0);
          const step = this.toolSteps.get(ev.tool_use_id);
          if (step) this.view.updateStep(step, { label: "Waiting for your OK to run this code", open: false });
          this.view.addPart({ type: "approval", id: ev.id, token: ev.token, language: ev.language, code: ev.code, state: "pending" });
          follow();
          return;
        }
        case "approval_result": {
          const part = this.view.turn.parts.find((p) => p.type === "approval" && p.id === ev.id);
          if (part && ["pending", "sending"].includes(part.state)) {
            part.state = ev.approved ? "approved" : "expired";
            delete part.token;
            part._paint?.();
          }
          if (ev.approved) {
            const step = [...this.toolSteps.values()].find((s) => s.name === "run_code" && s.status === "running");
            if (step) this.view.updateStep(step, { label: toolLabel("run_code", step.input, "running") });
          }
          return;
        }
        case "phase":
          this.startPhrases(ev.phrases);
          return;
        case "notice":
          this.view.addPart({ type: "notice", level: ev.level, text: ev.text });
          return;
        case "error":
          this.view.addPart({ type: "notice", level: "error", text: ev.text });
          this.finish();
          return;
        case "done":
          this.finish();
          return;
      }
    }

    blockStart(key, block) {
      const v = this.view;
      if (block.type === "thinking") {
        const step = v.addStep({ kind: "thinking", label: "Thinking", status: "running", open: true, body: { type: "thinking", text: "" }, started: Date.now() });
        this.blocks.set(key, { type: "thinking", step });
        this.active++;
      } else if (block.type === "text") {
        const part = { type: "text", md: "" };
        v.addPart(part);
        part._el.classList.add("streaming");
        this.blocks.set(key, { type: "text", part });
        this.active++;
      } else if (block.type === "tool_use" || block.type === "server_tool_use") {
        const name = block.name;
        const step = v.addStep({
          kind: TOOL_KIND[name] || "tool",
          label: toolLabel(name, block.input, "running"),
          status: "running",
          input: block.input || {},
          server: block.type === "server_tool_use",
          name,
        });
        if (name === "run_code") Object.assign(step, { open: true, body: { type: "code", text: "" } });
        if (name === "create_file") Object.assign(step, { open: true, body: { type: "code", text: "" } });
        if (name === "create_document") Object.assign(step, { open: true, body: { type: "outline", items: [] } });
        this.blocks.set(key, { type: block.type, step, name, json: "" });
        this.toolSteps.set(block.id, step);
        this.active++;
      } else if (block.type === "mcp_tool_use") {
        const step = v.addStep({ kind: "mcp", label: `Using ${block.server} · ${block.name}`, status: "running", input: block.input || {}, name: block.name });
        this.blocks.set(key, { type: "mcp", step, json: "" });
        this.toolSteps.set(block.id, step);
        this.active++;
      } else if (block.type === "web_search_tool_result" || block.type === "web_fetch_tool_result" || block.type === "mcp_tool_result") {
        const step = this.toolSteps.get(block.tool_use_id);
        if (!step) return;
        const failed = block.type === "mcp_tool_result" ? block.error : !!block.error;
        const patch = { status: failed ? "error" : "done" };
        if (block.type === "web_search_tool_result") {
          patch.label = failed ? `Search failed (${block.error})` : toolLabel("web_search", step.input, "done");
          patch.meta = block.results?.length ? `${block.results.length} sources` : "";
          patch.body = { type: "sources", items: block.results || [] };
        } else if (block.type === "web_fetch_tool_result") {
          patch.label = failed ? `Couldn't read ${host(step.input.url || "")}` : `Read ${block.title || host(step.input.url || block.url || "")}`;
          if (block.url) patch.body = { type: "sources", items: [{ url: block.url, title: block.title }] };
        } else {
          patch.label = `${failed ? "Connector error" : "Used"} · ${step.name}`;
          if (block.preview) patch.body = { type: "text", text: block.preview };
        }
        if (step.status === "running") this.active--;
        v.updateStep(step, patch);
        this.startPhrases(PHRASES_AFTER_TOOL);
      }
      this.syncLoader();
    }

    liveInput(b) {
      const json = b.json;
      const input = { ...b.step.input };
      for (const k of ["query", "url", "location", "near", "destination", "origin", "filename", "format", "title", "prompt"]) {
        const v = partialString(json, k);
        if (v !== undefined) input[k] = v;
      }
      const patch = { input };
      if (b.type === "mcp") {
        patch.body = { type: "text", text: json.slice(-600) };
      } else {
        patch.label = toolLabel(b.name, input, "running");
        if (b.name === "run_code") {
          const code = partialString(json, "code") || "";
          patch.body = { type: "code", text: code };
          const lines = code.split("\n").length;
          patch.meta = code ? `${lines} line${lines === 1 ? "" : "s"}` : "";
        } else if (b.name === "create_image" && input.prompt) {
          patch.body = { type: "text", text: input.prompt };
        } else if (b.name === "create_file") {
          patch.body = { type: "code", text: partialString(json, "content") || "" };
          const lines = patch.body.text.split("\n").length;
          patch.meta = patch.body.text ? `${lines} line${lines === 1 ? "" : "s"}` : "";
        } else if (b.name === "create_document") {
          const items = input.format === "pptx" ? allStrings(json, "title").slice(1) : allStrings(json, "heading");
          patch.body = { type: "outline", items };
          patch.meta = items.length ? `${items.length} ${input.format === "pptx" ? "slides" : "sections"}` : "";
        }
      }
      this.view.updateStep(b.step, patch);
    }

    blockStop(key) {
      const b = this.blocks.get(key);
      if (!b) return;
      if (b.type === "thinking") {
        const secs = Math.max(1, Math.round((Date.now() - b.step.started) / 1000));
        const heading = thinkingHeading(b.step.body.text);
        this.view.updateStep(b.step, { status: "done", open: false, label: heading || "Thought it through", meta: `${secs}s` });
        this.active--;
      } else if (b.type === "text") {
        b.part._el.classList.remove("streaming");
        renderMarkdown(b.part._el, b.part.md);
        this.active--;
      } else if (b.type === "tool_use") {
        // Input complete: the server runs the tool next (tool_run / tool_done).
        try {
          b.step.input = JSON.parse(b.json || "{}");
        } catch {
          /* keep partial */
        }
        this.view.updateStep(b.step, { label: toolLabel(b.name, b.step.input, "running") });
      } else if (b.type === "server_tool_use") {
        try {
          b.step.input = JSON.parse(b.json || "{}");
        } catch {
          /* keep partial */
        }
        this.view.updateStep(b.step, { label: toolLabel(b.name, b.step.input, "running") });
      } else if (b.type === "mcp") {
        try {
          b.step.input = JSON.parse(b.json || "{}");
        } catch {
          /* keep partial */
        }
      }
      this.syncLoader();
    }

    finish() {
      if (this.finished) return;
      this.finished = true;
      clearInterval(this.phraseTimer);
      this.finishReads();
      for (const b of this.blocks.values()) {
        if (b.type === "text") {
          b.part._el?.classList.remove("streaming");
          if (b.part._el) renderMarkdown(b.part._el, b.part.md);
        } else if (b.step && b.step.status === "running") {
          this.view.updateStep(b.step, { status: "done", open: false });
        }
      }
      this.view.loader.hidden = true;
      if (!this.view.turn.parts.length) this.view.addPart({ type: "notice", text: "No response." });
    }
  }

  /* ───────────── Composer ───────────── */
  const input = $("#input");
  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
    updateSendState();
  }
  function updateSendState() {
    $("#sendBtn").disabled = !busy && !input.value.trim() && !attachments.length;
    $("#sendBtn").setAttribute("aria-label", busy ? "Stop" : "Send");
  }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!busy) send(input.value);
    }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) stop();
    else send(input.value);
  });
  $("#suggestions").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-prompt]");
    if (chip) send(chip.dataset.prompt);
  });

  const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|jsonl|xml|html?|css|js|mjs|cjs|ts|tsx|jsx|py|rb|c|h|cpp|cc|hpp|lua|go|rs|java|kt|swift|php|sh|bash|zsh|ya?ml|toml|ini|cfg|conf|sql|log|r|pl|scala|dart|vue|svelte|tex|env|gitignore)$/i;
  const readAs = (file, how) =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(r.error);
      how === "text" ? r.readAsText(file) : r.readAsDataURL(file);
    });

  async function addFiles(list) {
    for (const file of list) {
      if (attachments.length >= 10) {
        toast("Up to 10 files per message");
        break;
      }
      const isImage = /^image\/(png|jpe?g|gif|webp)$/.test(file.type);
      const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
      const isText = file.type.startsWith("text/") || TEXT_EXT.test(file.name) || file.type === "application/json";
      try {
        if (isImage) {
          if (file.size > 5 * 1048576) throw new Error("Images must be under 5 MB");
          const url = await readAs(file, "data");
          attachments.push({ name: file.name, size: file.size, type: file.type.replace("jpg", "jpeg"), kind: "image", data: url.split(",")[1], preview: url });
        } else if (isPdf) {
          if (file.size > 25 * 1048576) throw new Error("PDFs must be under 25 MB");
          const url = await readAs(file, "data");
          attachments.push({ name: file.name, size: file.size, type: "application/pdf", kind: "pdf", data: url.split(",")[1] });
        } else if (isText) {
          if (file.size > 2 * 1048576) throw new Error("Text files must be under 2 MB");
          attachments.push({ name: file.name, size: file.size, type: file.type, kind: "text", text: await readAs(file, "text") });
        } else throw new Error(`${file.name}: unsupported file type (use images, PDFs or text/code files)`);
      } catch (err) {
        toast(err.message, 3500);
      }
    }
    renderAttachments();
  }
  function renderAttachments() {
    const box = $("#attachments");
    box.innerHTML = "";
    attachments.forEach((a, k) => {
      const chip = el("span", "att", `${a.preview ? `<img src="${a.preview}" alt="">` : a.kind === "pdf" ? "📕" : "📄"}<span class="att-name">${esc(a.name)}</span><button type="button" class="icon-btn" aria-label="Remove">✕</button>`);
      $("button", chip).onclick = () => {
        attachments.splice(k, 1);
        renderAttachments();
      };
      box.appendChild(chip);
    });
    updateSendState();
  }
  $("#attachBtn").onclick = () => $("#fileInput").click();
  $("#fileInput").onchange = (e) => {
    addFiles([...e.target.files]);
    e.target.value = "";
  };
  input.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) {
      e.preventDefault();
      addFiles(files);
    }
  });
  let dragDepth = 0;
  main.addEventListener("dragenter", (e) => {
    if (![...e.dataTransfer.types].includes("Files")) return;
    dragDepth++;
    main.classList.add("dragging");
  });
  main.addEventListener("dragleave", () => {
    if (--dragDepth <= 0) {
      dragDepth = 0;
      main.classList.remove("dragging");
    }
  });
  main.addEventListener("dragover", (e) => e.preventDefault());
  main.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    main.classList.remove("dragging");
    addFiles([...e.dataTransfer.files]);
  });

  $("#locBtn").onclick = async () => {
    settings.location = !settings.location;
    saveSettings();
    if (settings.location) {
      getLocation.cache = null;
      const loc = await getLocation();
      toast(loc ? "Location sharing on" : "Couldn't get your location — check browser permissions", 3000);
      if (!loc) {
        settings.location = false;
        saveSettings();
      }
    } else toast("Location sharing off");
  };
  $("#modelQuick").onchange = (e) => {
    settings.model = e.target.value;
    saveSettings();
  };

  /* ───────────── Drawer: history & MCP ───────────── */
  const drawer = $("#drawer");
  function openDrawer(name) {
    const isOpen = drawer.classList.contains("open") && drawer.dataset.panel === name;
    closeDrawer();
    if (isOpen) return;
    drawer.dataset.panel = name;
    $$(".drawer-panel", drawer).forEach((p) => p.classList.toggle("active", p.dataset.panel === name));
    $$(".rail-btn[data-panel]").forEach((b) => b.classList.toggle("active", b.dataset.panel === name));
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    $("#scrim").classList.add("show");
    if (name === "history") refreshHistory();
    if (name === "mcp") renderMcp();
  }
  function closeDrawer() {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    $("#scrim").classList.remove("show");
    $$(".rail-btn[data-panel]").forEach((b) => b.classList.remove("active"));
  }
  $$(".rail-btn[data-panel]").forEach((b) => (b.onclick = () => openDrawer(b.dataset.panel)));
  $$("[data-close-drawer]").forEach((b) => (b.onclick = closeDrawer));
  $("#scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });

  async function refreshHistory() {
    const list = $("#historyList");
    const q = $("#historySearch").value.trim().toLowerCase();
    const chats = (await db.all()).sort((a, b) => b.updated - a.updated).filter((c) => !q || (c.title || "").toLowerCase().includes(q));
    list.innerHTML = chats.length ? "" : `<li class="history-empty">${q ? "No matching chats" : "No chats yet — say hi!"}</li>`;
    for (const c of chats) {
      const li = el("li", chat?.id === c.id ? "current" : "", `<span class="h-title"></span><span class="h-date">${new Date(c.updated).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span><button class="icon-btn" aria-label="Delete chat">✕</button>`);
      li.dataset.id = c.id;
      $(".h-title", li).textContent = c.title || "Untitled";
      li.onclick = async (e) => {
        if (e.target.closest(".icon-btn")) {
          e.stopPropagation();
          await db.del(c.id);
          if (chat?.id === c.id) newChat();
          refreshHistory();
          return;
        }
        if (busy) stop();
        const full = await db.get(c.id);
        if (!full) return;
        chat = full;
        renderChat(chat);
        closeDrawer();
      };
      list.appendChild(li);
    }
  }
  function highlightHistory() {
    $$("#historyList li").forEach((li) => li.classList.toggle("current", li.dataset.id === chat?.id));
  }
  $("#historySearch").addEventListener("input", refreshHistory);

  function renderMcp() {
    const list = $("#mcpList");
    list.innerHTML = mcpServers.length ? "" : `<li class="history-empty" style="border:0;background:none">No connectors yet.</li>`;
    for (const s of mcpServers) {
      const li = el("li", "", `<div class="m-info"><div class="m-name"></div><div class="m-url"></div></div>
        <label class="switch" title="Enable"><input type="checkbox" ${s.enabled ? "checked" : ""}><span></span></label>
        <button class="icon-btn" aria-label="Remove connector">✕</button>`);
      $(".m-name", li).textContent = s.name;
      $(".m-url", li).textContent = s.url;
      $("input", li).onchange = (e) => {
        s.enabled = e.target.checked;
        store.set("mnx.mcp", mcpServers);
      };
      $(".icon-btn", li).onclick = () => {
        mcpServers = mcpServers.filter((x) => x !== s);
        store.set("mnx.mcp", mcpServers);
        renderMcp();
      };
      list.appendChild(li);
    }
  }
  $("#mcpForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const url = String(f.get("url")).trim();
    if (!/^https:\/\//i.test(url)) return toast("Connector URL must start with https://");
    mcpServers.push({ id: uid(), name: String(f.get("name")).trim(), url, token: String(f.get("token") || "").trim(), enabled: true });
    store.set("mnx.mcp", mcpServers);
    e.target.reset();
    renderMcp();
    toast("Connector added");
  });

  /* ───────────── Settings dialog ───────────── */
  const dlg = $("#settingsDialog");
  function openSettings(focusName) {
    const f = $("#settingsForm");
    f.name.value = settings.name;
    f.model.value = settings.model;
    f.effort.value = settings.effort;
    f.theme.value = settings.theme;
    f.location.checked = settings.location;
    dlg.showModal();
    if (focusName) f.name.focus();
  }
  dlg.addEventListener("close", async () => {
    if (dlg.returnValue !== "save") return;
    const f = $("#settingsForm");
    const wantsLoc = f.location.checked && !settings.location;
    Object.assign(settings, { name: f.name.value.trim(), model: f.model.value, effort: f.effort.value, theme: f.theme.value, location: f.location.checked });
    saveSettings();
    if (wantsLoc) {
      getLocation.cache = null;
      if (!(await getLocation())) toast("Couldn't get your location — check browser permissions", 3000);
    }
  });
  $("#settingsBtn").onclick = () => openSettings(false);
  $("#profileBtn").onclick = () => openSettings(true);
  $("#clearAllBtn").onclick = async () => {
    if (!confirm("Delete all chats? This can't be undone.")) return;
    await db.clear();
    newChat();
    refreshHistory();
    dlg.close();
    toast("All chats deleted");
  };
  $$("[data-close-dialog]").forEach((b) => (b.onclick = () => b.closest("dialog").close()));
  $("#previewDialog").addEventListener("close", () => ($("#previewFrame").srcdoc = ""));

  $("#newChatBtn").onclick = () => {
    closeDrawer();
    newChat();
  };
  $("#brandBtn").onclick = () => {
    closeDrawer();
    newChat();
  };

  // Phones close background tabs without warning: save whenever we're hidden.
  document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && saveChat());
  window.addEventListener("pagehide", () => saveChat());

  /* ───────────── Boot ───────────── */
  fillModelSelects();
  applySettings();
  autosize();
  // Loading a model on a phone can take minutes. Say so, with progress.
  function watchLocalStatus(st) {
    const note = $("#readyNote");
    const show = (text, cls = "") => {
      note.hidden = !text;
      note.textContent = text;
      note.className = `ready-note ${cls}`;
    };
    if (!st || st.state === "ready" || st.state === "idle") return show("");
    if (st.state === "error") return show(`Your local model couldn't start: ${st.error}`, "err");
    show(st.state === "loading" ? "Loading your model into memory… keep Termux open." : `Getting Mnx ready · ${st.progress}% — the first start takes a few minutes. Keep Termux open.`);
    setTimeout(() => fetch("/api/status").then((r) => r.json()).then(watchLocalStatus, () => watchLocalStatus(st)), 1500);
  }

  fetch("/api/config")
    .then((r) => r.json())
    .then((cfg) => {
      if (cfg.models) {
        MODELS = cfg.models;
        fillModelSelects();
        applySettings();
      }
      // With no API key but a local model on the server, default to the local model.
      if (!cfg.hasKey && cfg.local && settings.model !== cfg.local) {
        settings.model = cfg.local;
        saveSettings();
        applySettings();
      }
      if (!cfg.hasKey && !cfg.local) toast("Heads up: the server has no ANTHROPIC_API_KEY set yet.", 6000);
      if (cfg.local) watchLocalStatus(cfg.localStatus);
    })
    .catch(() => {});
  if (!settings.name) setTimeout(() => toast("Tip: tap your avatar to tell Mnx your name", 3500), 1200);
  input.focus();
})();

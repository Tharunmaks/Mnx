// Mnx Hive — shared helpers: icons, settings, DOM, and the gateway client.
// No AI lives here. The web app only renders events the gateway sends.

export const ICONS = {
  hex: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l8.66 5v10L12 22l-8.66-5V7z"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  chat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M4 5h16v11H9l-5 4z"/></svg>',
  tool: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L3 17.8V21h3.2l6.3-6.3a4 4 0 0 0 5.2-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/></svg>',
  hive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M8 3l4 2.3v4.6L8 12.2 4 9.9V5.3zM16 3l4 2.3v4.6l-4 2.3-4-2.3V5.3zM12 11.8l4 2.3v4.6L12 21l-4-2.3v-4.6z"/></svg>',
  gear: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
  menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>',
  clip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5l-8.6 8.6a5 5 0 0 1-7.1-7.1l8.6-8.6a3.3 3.3 0 0 1 4.7 4.7l-8.6 8.6a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  back: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>',
  plug: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 7V3M15 7V3M6 7h12v4a6 6 0 0 1-12 0zM12 17v4"/></svg>',
  globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/></svg>',
  phone: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="6.5" y="2.5" width="11" height="19" rx="2.5"/><path d="M11 18.5h2"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>',
  fwd: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  // step icons, keyed by event "type"
  thinking: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><path d="M9 10h.01M15 10h.01M9 15c1.5 1 4.5 1 6 0"/></svg>',
  planning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/></svg>',
  searching: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/></svg>',
  reading: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M4 5h6a2 2 0 0 1 2 2v12a2 2 0 0 0-2-2H4zM20 5h-6a2 2 0 0 0-2 2v12a2 2 0 0 1 2-2h6z"/></svg>',
  web: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/></svg>',
  opening: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>',
  fetching: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  writing: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg>',
  running: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M7 5l12 7-12 7z"/></svg>',
  booking: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M9 3v4M15 3v4"/></svg>',
  waiting: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg>',
  memory: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/></svg>',
  done: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5h.01"/></svg>',
  user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>',
};

// Hexagon logo with an "M" inside.
export function logoSVG(cls = "logo") {
  return `<svg class="${cls}" viewBox="0 0 40 40" aria-hidden="true">
    <path d="M20 2.5l15.2 8.75v17.5L20 37.5 4.8 28.75v-17.5z" fill="#f5a623"/>
    <path d="M12.5 26.5V14l7.5 7.5 7.5-7.5v12.5" fill="none" stroke="#1a1203" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

// Hexagon avatar in a Bee's colour, letter inside.
export function beeHex(name, color, cls = "hex") {
  const letter = esc((name || "?").trim().charAt(0).toUpperCase());
  const c = color || colorFor(name);
  return `<svg class="${cls}" viewBox="0 0 40 40" aria-hidden="true">
    <path d="M20 2.5l15.2 8.75v17.5L20 37.5 4.8 28.75v-17.5z" fill="${c}" fill-opacity="0.16" stroke="${c}" stroke-width="1.5"/>
    <text x="20" y="25.5" text-anchor="middle" font-size="15" font-weight="700" fill="${c}" font-family="Inter, sans-serif">${letter}</text>
  </svg>`;
}

const BEE_COLORS = ["#f5a623", "#4c9bff", "#3ecf8e", "#c084fc", "#ff7a59", "#2dd4bf", "#f472b6", "#a3e635"];
export function colorFor(name = "") {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return BEE_COLORS[h % BEE_COLORS.length];
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function uid(prefix = "") {
  return prefix + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

// ---------- Settings (per-browser, localStorage) ----------
const SETTINGS_KEY = "mnx.settings";
const DEFAULTS = {
  name: "",
  model: "Mnx 3B",
  gateway: "", // empty = same origin as the page
  token: "", // access token printed by the gateway (data/token.txt)
  enterToSend: true,
};

export const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(key);
      return v == null ? fallback : JSON.parse(v);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked */ }
  },
};

export function getSettings() {
  return { ...DEFAULTS, ...store.get(SETTINGS_KEY, {}) };
}
export function saveSettings(s) {
  store.set(SETTINGS_KEY, { ...getSettings(), ...s });
}

export function gatewayBase() {
  const g = getSettings().gateway.trim().replace(/\/+$/, "");
  return g || location.origin;
}

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export async function api(path, opts = {}) {
  const res = await fetch(gatewayBase() + path, {
    ...opts,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${getSettings().token}` },
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    if (res.status === 401) window.dispatchEvent(new CustomEvent("mnx-auth"));
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

// ---------- Gateway client (WebSocket, auto-reconnect) ----------
// Server → client events: { task, bee, type, text, at, ... }  (see README "Event protocol")
// Client → server:        { type: "user_message" | "action" | "stop", ... }
export class HiveClient extends EventTarget {
  constructor() {
    super();
    this.ws = null;
    this.state = "connecting";
    this.queue = [];
    this.retry = 0;
    this.closed = false;
  }

  url() {
    const base = new URL(gatewayBase());
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    base.pathname = "/ws";
    base.search = "?token=" + encodeURIComponent(getSettings().token);
    return base.toString();
  }

  connect() {
    if (location.protocol === "file:" && !getSettings().gateway) {
      this.setState("offline");
      return;
    }
    this.setState("connecting");
    let ws;
    try { ws = new WebSocket(this.url()); } catch { this.scheduleReconnect(); return; }
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.setState("online");
      for (const m of this.queue.splice(0)) ws.send(JSON.stringify(m));
    };
    ws.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      this.dispatchEvent(new CustomEvent("event", { detail: ev }));
    };
    ws.onclose = (e) => {
      this.ws = null;
      if (e.code === 4401) {
        this.setState("auth");
        window.dispatchEvent(new CustomEvent("mnx-auth"));
        return; // wait for a new token from Settings
      }
      if (!this.closed) this.scheduleReconnect();
    };
    ws.onerror = () => ws.close();
  }

  scheduleReconnect() {
    this.setState("offline");
    const delay = Math.min(15000, 1000 * 2 ** this.retry++);
    setTimeout(() => this.connect(), delay);
  }

  reconnectNow() {
    this.retry = 0;
    if (this.ws) this.ws.close();
    else this.connect();
  }

  setState(s) {
    this.state = s;
    this.dispatchEvent(new CustomEvent("state", { detail: s }));
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }
}

export function timeShort(iso) {
  const d = iso ? new Date(iso) : new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ---------- shared page chrome ----------
export function connLabel(el, state) {
  el.className = `conn ${state === "auth" ? "offline" : state}`;
  el.querySelector("span").textContent =
    state === "online" ? "Online" : state === "offline" ? "Offline" : state === "auth" ? "Needs token" : "Connecting";
}

// Asks for the access token once when the gateway says 401.
let asking = false;
export function installTokenPrompt(onSaved) {
  window.addEventListener("mnx-auth", () => {
    if (asking) return;
    asking = true;
    setTimeout(() => {
      const t = prompt("Enter your Mnx Hive access token (printed by the server, also in data/token.txt):", "");
      asking = false;
      if (t && t.trim()) {
        saveSettings({ token: t.trim() });
        onSaved?.();
      }
    }, 50);
  });
}

export function fillIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });
}

// ---------- toasts: short, non-blocking messages ----------
export function toast(message, kind = "info", ms = 3800) {
  let box = document.getElementById("toasts");
  if (!box) {
    box = h("div", { id: "toasts", class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(box);
  }
  const el = h("div", { class: `toast ${kind}` }, String(message));
  box.append(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 300); }, ms);
}

// Human-friendly sizes and durations, shared by the Cell and Lab pages.
export function fmtSize(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}
export function fmtDuration(sec) {
  if (sec == null || sec < 0) return "—";
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

// Stream a file to the server (with progress); returns {id, name, size} for runs, imports and chat cards.
export function uploadFile(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `${gatewayBase()}/api/lab/uploads/${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Authorization", `Bearer ${getSettings().token}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText || "{}").detail || xhr.statusText)));
    xhr.onerror = () => reject(new Error("Upload failed (connection)"));
    xhr.send(file);
  });
}

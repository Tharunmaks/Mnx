// Mnx Hive — voice: Fish Audio speech and calls.
// A Bee rings you (the call screen pops up on every open Hive page), speaks with the Fish Audio voice,
// listens to you and does what you say. Talk mode is the same call, started by you from the composer.
// Without a Fish Audio key the browser's own voice and speech recognition stand in.

import { ICONS, api, beeHex, gatewayBase, getSettings, h, toast } from "./shared.js";

const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
let ui = null;          // the call screen
let current = null;     // the call shown on it
let player = null;      // the line being played
const queue = [];       // lines waiting to be played
let mic = null;         // { stream, ctx, analyser, rec } while listening with Fish Audio
let rec = null;         // browser speech recognition
let muted = false;
let ringer = null;
let tick = null;

// ---------- speaking ----------
async function audioBlob(path, opts = {}) {
  const res = await fetch(gatewayBase() + path, { ...opts, headers: { Authorization: `Bearer ${getSettings().token}`, ...(opts.headers || {}) } });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.blob();
}

function playBlob(blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const a = new Audio(url);
    player = a;
    a.onended = a.onerror = () => { URL.revokeObjectURL(url); player = null; resolve(); };
    a.play().catch(() => { player = null; resolve(); });
  });
}

function browserSay(text) {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) { resolve(); return; }
    const u = new SpeechSynthesisUtterance(text);
    const done = () => { clearTimeout(guard); resolve(); };
    // some browsers never fire "end" (no voices installed): don't wait longer than reading it would take
    const guard = setTimeout(done, 1500 + 450 * text.split(/\s+/).length);
    u.onend = u.onerror = done;
    speechSynthesis.speak(u);
  });
}

// Read text aloud with the Fish Audio voice (the browser's voice when no key is set). Used by "Speak replies".
export async function speak(text) {
  if (!text) return;
  try {
    await playBlob(await audioBlob("/api/voice/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }));
  } catch {
    await browserSay(text.slice(0, 600));
  }
}

export function stopSpeaking() {
  if (player) { player.pause(); player = null; }
  window.speechSynthesis?.cancel();
}

// ---------- starting calls ----------
// Talk mode: a call you start yourself, already picked up.
export async function startTalk(bee) {
  try {
    const call = await api("/api/calls", { method: "POST", body: { bee: bee || null, answer: true } });
    show(call);
  } catch (e) {
    toast(`Couldn't start talking: ${e.message}`, "error");
  }
}

// ---------- the call screen ----------
export function installVoice(client) {
  client.addEventListener("event", (e) => {
    const ev = e.detail;
    if (ev.type === "call") onCall(ev.call);
    else if (ev.type === "call_said" && current && ev.call === current.id) onSaid(ev);
    else if (ev.type === "call_heard" && current && ev.call === current.id) addLine("you", ev.text);
    else if (ev.type === "call_event" && current && ev.call === current.id) onStep(ev.event);
  });
  client.addEventListener("state", (e) => {
    if (e.detail !== "online") return;
    api("/api/calls").then((list) => { const ring = list.find((c) => c.state === "ringing" || c.state === "live"); if (ring) onCall(ring); }).catch(() => {});
  });
}

function onCall(call) {
  if (["ringing", "live"].includes(call.state)) {
    if (!current || current.id === call.id || current.state !== "live") show(call);
    return;
  }
  if (current && current.id === call.id) {
    current = call;
    if (["ended", "missed", "declined", "failed", "cancelled"].includes(call.state)) ended(call);
    else render();
  }
}

function show(call) {
  const fresh = !current || current.id !== call.id;
  current = call;
  if (!ui) build();
  ui.root.classList.remove("hidden");
  if (fresh) {
    ui.lines.replaceChildren();
    for (const l of call.transcript || []) addLine(l.who, l.text);
  }
  if (call.state === "ringing") startRinging(); else stopRinging();
  render();
}

function build() {
  const root = h("div", { class: "call-overlay hidden", role: "dialog", "aria-label": "Call" });
  const avatar = h("div", { class: "call-avatar" });
  const who = h("div", { class: "call-who" });
  const status = h("div", { class: "call-status" });
  const lines = h("div", { class: "call-lines", "aria-live": "polite" });
  const card = h("div", { class: "call-card-slot" });
  const listen = h("div", { class: "call-listen" });
  const typed = h("input", { class: "call-type", placeholder: "Or type what to do…", enterkeyhint: "send" });
  const form = h("form", { class: "call-form", onsubmit: (e) => { e.preventDefault(); const t = typed.value.trim(); if (t) { typed.value = ""; send(t); } } }, typed);
  const answer = h("button", { class: "call-btn answer", onclick: () => act("answer") }, h("span", { html: ICONS.phone || "" }), "Answer");
  const decline = h("button", { class: "call-btn hang", onclick: () => act("decline") }, "Decline");
  const mute = h("button", { class: "call-btn ghost", onclick: () => { muted = !muted; render(); if (muted) stopListening(); else maybeListen(); } }, "Mute");
  const hang = h("button", { class: "call-btn hang", onclick: () => act("hangup") }, "Hang up");
  const close = h("button", { class: "call-btn ghost", onclick: hide }, "Close");
  const buttons = h("div", { class: "call-buttons" }, answer, decline, mute, hang, close);
  root.append(h("div", { class: "call-box" }, h("div", { class: "call-top" }, avatar, h("div", {}, who, status)), lines, card, listen, form, buttons));
  document.body.append(root);
  ui = { root, avatar, who, status, lines, card, listen, form, answer, decline, mute, hang, close };
}

function render() {
  if (!ui || !current) return;
  const c = current;
  ui.avatar.innerHTML = beeHex(c.bee === "Mnx" ? "Mnx" : c.bee);
  ui.who.textContent = c.bee === "Mnx" ? "Your Hive" : `${c.bee} Bee`;
  const live = c.state === "live";
  const ringing = c.state === "ringing";
  const done = !live && !ringing;
  const about = c.request ? `about ${c.request}` : c.reason || "";
  let st = ringing ? (c.via === "phone" ? "Calling your phone…" : "Incoming call…") + (about ? ` · ${about}` : "") : live ? `On call · ${elapsed(c)}` : stateText(c);
  if (live && c.via === "phone") st += " · on your phone";
  ui.status.textContent = st;
  ui.answer.classList.toggle("hidden", !ringing || c.via === "phone");
  ui.decline.classList.toggle("hidden", !ringing);
  ui.mute.classList.toggle("hidden", !live || c.via === "phone");
  ui.mute.textContent = muted ? "Unmute" : "Mute";
  ui.hang.classList.toggle("hidden", !live);
  ui.close.classList.toggle("hidden", !done);
  ui.form.classList.toggle("hidden", !live || c.via === "phone");
  ui.listen.textContent = !live ? "" : c.via === "phone" ? "Talk on your phone; this screen follows along."
    : player || queue.length ? "Speaking…" : c.working ? "Working on it… (say “stop” to cancel)" : muted ? "Muted" : listeningText();
  ui.listen.className = `call-listen ${mic || rec ? "on" : ""}`;
  renderCard(c.card);
  clearInterval(tick);
  if (live) tick = setInterval(() => { if (current && current.state === "live") ui.status.textContent = `On call · ${elapsed(current)}` + (current.via === "phone" ? " · on your phone" : ""); }, 1000);
  if (live && c.via !== "phone") maybeListen();
  if (done) stopListening();
}

function listeningText() {
  if (current.voice === "fish") return mic ? "Listening (Fish Audio)…" : "Starting the microphone…";
  if (SpeechRec) return rec ? "Listening…" : "Starting the microphone…";
  return "Type below: this browser can't listen. Add a Fish Audio key in Settings → Voice.";
}

function stateText(c) {
  return { ended: "Call ended", missed: "Missed call", declined: "Declined", cancelled: "Cancelled", failed: `Couldn't call: ${c.error || "error"}` }[c.state] || c.state;
}

function elapsed(c) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - (c.answered_at || Date.now() / 1000)));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function addLine(who, text) {
  if (!ui) return;
  ui.lines.append(h("div", { class: `call-line ${who}` }, text));
  ui.lines.scrollTop = ui.lines.scrollHeight;
}

function onStep(ev) {
  if (ev.type === "step" || ev.type === "running" || ev.type === "searching") {
    if (ev.text && ev.status !== "done") ui.listen.textContent = `${ev.bee ? ev.bee + ": " : ""}${ev.text}`;
  }
}

function renderCard(card) {
  ui.card.replaceChildren();
  if (!card) return;
  const box = h("div", { class: "call-card" }, h("b", {}, card.title || "Your answer"));
  for (const row of (card.details || []).slice(0, 4)) box.append(h("div", { class: "row" }, h("span", {}, row[0]), h("span", {}, String(row[1]))));
  const values = {};
  for (const f of card.fields || []) {
    values[f.name] = f.value ?? (f.options && f.options[0] ? (f.options[0].value ?? f.options[0]) : "");
    if (f.input === "select") {
      const sel = h("select", { onchange: () => { values[f.name] = sel.value; } },
        ...(f.options || []).map((o) => h("option", { value: o.value ?? o }, o.label ?? o)));
      sel.value = values[f.name];
      box.append(h("label", { class: "row" }, h("span", {}, f.label || f.name), sel));
    } else if (f.input !== "file") {
      const inp = h("input", { value: values[f.name] ?? "", placeholder: f.placeholder || "", oninput: () => { values[f.name] = inp.value; } });
      box.append(h("label", { class: "row" }, h("span", {}, f.label || f.name), inp));
    }
  }
  const acts = h("div", { class: "acts" });
  for (const a of card.actions || []) {
    acts.append(h("button", { class: `btn ${a.style === "danger" ? "danger" : "primary"}`,
      onclick: () => api(`/api/calls/${current.id}/card`, { method: "POST", body: { action: a.id, values } }).catch((e) => toast(e.message, "error")) }, a.label || a.id));
  }
  box.append(acts);
  ui.card.append(box);
}

async function act(what) {
  if (!current) return;
  try {
    if (what === "answer") unlockAudio();
    const call = await api(`/api/calls/${current.id}/${what}`, { method: "POST" });
    stopRinging();
    onCall(call);
    if (what === "answer") show(call);
  } catch (e) {
    toast(e.message, "error");
  }
}

function hide() {
  stopListening();
  stopRinging();
  stopSpeaking();
  queue.length = 0;
  clearInterval(tick);
  ui?.root.classList.add("hidden");
  current = null;
}

function ended(call) {
  stopRinging();
  stopListening();
  render();
  setTimeout(() => { if (current && current.id === call.id && !player && !queue.length) hide(); }, call.state === "failed" ? 8000 : 4000);
}

async function send(text) {
  if (!current) return;
  try {
    await api(`/api/calls/${current.id}/say`, { method: "POST", body: { text } });
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------- the Bee's lines ----------
function onSaid(ev) {
  addLine("bee", ev.text);
  if (current.via === "phone") return; // it's playing on your phone
  queue.push(ev);
  if (queue.length === 1) playNext();
}

async function playNext() {
  const ev = queue[0];
  if (!ev) { render(); return; }
  stopListening(); // don't hear yourself
  render();
  try {
    if (ev.audio) await playBlob(await audioBlob(`/api/calls/${current.id}/audio/${ev.n}`));
    else await browserSay(ev.text);
  } catch { /* skip a line that won't play */ }
  queue.shift();
  if (queue.length) playNext();
  else render();
}

// ---------- listening ----------
function maybeListen() {
  if (!current || current.state !== "live" || current.via === "phone" || muted || player || queue.length) return;
  if (current.voice === "fish") listenFish();
  else if (SpeechRec) listenBrowser();
}

function stopListening() {
  if (rec) { try { rec.onend = null; rec.abort(); } catch { /* already stopped */ } rec = null; }
  if (mic) {
    if (mic.rec && mic.rec.state !== "inactive") { mic.cancel = true; mic.rec.stop(); }
    mic.stream.getTracks().forEach((t) => t.stop());
    mic.ctx.close().catch(() => {});
    cancelAnimationFrame(mic.raf);
    mic = null;
  }
}

function listenBrowser() {
  if (rec) return;
  rec = new SpeechRec();
  rec.lang = navigator.language || "en-IN";
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  rec.onresult = (e) => { const t = e.results[0][0].transcript; if (t.trim()) send(t); };
  rec.onend = () => { rec = null; setTimeout(maybeListen, 300); };
  rec.onerror = () => {};
  try { rec.start(); } catch { rec = null; }
  render();
}

// Fish Audio listening: record while you speak (a simple loudness gate), send each sentence when you pause.
async function listenFish() {
  if (mic) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch {
    ui.listen.textContent = "Microphone blocked. Allow it for this page, or type below.";
    return;
  }
  if (!current || current.state !== "live" || player || queue.length) { stream.getTracks().forEach((t) => t.stop()); return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const data = new Float32Array(analyser.fftSize);
  mic = { stream, ctx, analyser, rec: null, raf: 0, cancel: false };
  const me = mic;
  let talking = false, quietSince = 0, startedAt = 0, chunks = [];
  const begin = () => {
    chunks = [];
    const type = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"].find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || "";
    me.rec = new MediaRecorder(stream, type ? { mimeType: type } : {});
    me.rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    me.rec.onstop = () => {
      if (me.cancel) return;
      const blob = new Blob(chunks, { type: me.rec.mimeType || "audio/webm" });
      if (blob.size > 2000) upload(blob);
    };
    me.rec.start();
    startedAt = performance.now();
  };
  const loop = () => {
    if (mic !== me) return;
    analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    const loud = Math.sqrt(sum / data.length) > 0.02;
    const t = performance.now();
    if (loud && !talking) { talking = true; begin(); }
    if (talking) {
      if (loud) quietSince = t;
      if ((t - quietSince > 900 && t - startedAt > 400) || t - startedAt > 15000) { talking = false; me.rec.stop(); }
    }
    me.raf = requestAnimationFrame(loop);
  };
  loop();
  render();
}

async function upload(blob) {
  if (!current) return;
  ui.listen.textContent = "Hearing you…";
  try {
    const res = await fetch(`${gatewayBase()}/api/calls/${current.id}/listen`, {
      method: "POST", body: blob, headers: { Authorization: `Bearer ${getSettings().token}`, "Content-Type": blob.type || "audio/webm" } });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
  } catch (e) {
    toast(`Fish Audio couldn't hear that: ${e.message}`, "error");
  }
}

// ---------- ringing ----------
let audioCtx = null;
function unlockAudio() {
  try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch { /* no audio */ }
}
document.addEventListener("pointerdown", unlockAudio, { once: true });

function startRinging() {
  if (ringer) return;
  const beep = () => {
    navigator.vibrate?.([400, 200, 400]);
    try {
      unlockAudio();
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.frequency.value = 880;
      g.gain.setValueAtTime(0.0001, audioCtx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.25, audioCtx.currentTime + 0.05);
      g.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.9);
      o.connect(g).connect(audioCtx.destination);
      o.start();
      o.stop(audioCtx.currentTime + 1);
    } catch { /* audio blocked until you tap the page */ }
  };
  beep();
  ringer = setInterval(beep, 2200);
  if (document.hidden && window.Notification?.permission === "granted") {
    try { new Notification(`${current.bee} Bee is calling`, { body: current.reason || "Tap to answer", tag: current.id }); } catch { /* not allowed */ }
  }
}

function stopRinging() {
  clearInterval(ringer);
  ringer = null;
  navigator.vibrate?.(0);
}

// Settings → Voice asks once, so a call can show a notification while the page is in the background.
export function askNotifications() {
  if (window.Notification && Notification.permission === "default") Notification.requestPermission().catch(() => {});
}

// The Sandbox player — Mnx Hive's artifact view for what a Bee made (game, website, 3D model):
//   Play tab:  runs it (its private /d/ or /p/ link) in a locked-down frame.
//   Code tab:  the files the Bee wrote, to read and copy.
//   Bottom:    "change something, or paste an error" — sent to the chat, which changes this same game (not a new one).
// Chroma already serves those pages with a CSP sandbox (their own opaque origin); the frame's sandbox attribute is a second
// lock, so a page can never reach the Hive's pages, your token or your other sites.
import { api, h } from "./shared.js";

export const RUNNABLE = /^\/(?:d|p)\/[\w.-]+\/?/;

export function openRunner(url, title = "Sandbox") {
  if (!RUNNABLE.test(url)) return;
  document.querySelector(".runner")?.remove();
  const frame = h("iframe", {
    src: url, title, allow: "fullscreen; gamepad; autoplay",
    sandbox: "allow-scripts allow-forms allow-pointer-lock allow-popups allow-modals allow-downloads",
  });
  const code = h("div", { class: "runner-code", hidden: true }, h("p", { class: "runner-note" }, "Loading the code…"));
  const tabPlay = h("button", { type: "button", class: "btn on" }, "Play");
  const tabCode = h("button", { type: "button", class: "btn" }, "Code");
  const close = h("button", { type: "button", class: "btn", "aria-label": "Close" }, "Close");
  const full = h("button", { type: "button", class: "btn" }, "Full screen");
  const tab = h("a", { class: "btn", href: url, target: "_blank", rel: "noopener noreferrer" }, "New tab");
  const say = h("input", { type: "text", class: "runner-say", placeholder: "Change something, or paste an error here…", enterkeyhint: "send" });
  const send = h("button", { type: "button", class: "btn primary" }, "Send to the Bee");
  let prefix = "";
  const box = h("div", { class: "runner", role: "dialog", "aria-label": title },
    h("div", { class: "runner-win" },
      h("div", { class: "runner-bar" }, h("b", {}, title), h("span", { class: "runner-lock" }, "Sandboxed · private link"),
        tabPlay, tabCode, tab, full, close),
      frame, code,
      h("div", { class: "runner-ask" }, say, send)));
  const show = (which) => {
    frame.hidden = which !== "play"; code.hidden = which !== "code";
    tabPlay.classList.toggle("on", which === "play"); tabCode.classList.toggle("on", which === "code");
  };
  const shut = () => { box.remove(); removeEventListener("keydown", esc); };
  const esc = (e) => { if (e.key === "Escape" && document.fullscreenElement == null) shut(); };
  const ask = () => {
    const text = say.value.trim();
    if (!text) return;
    // an error pasted as it is goes as it is (the Hive knows it is an error); a wish is tied to this game by name
    const msg = /error|not defined|is not a function|cannot read/i.test(text) ? text : prefix + text;
    const ev = new CustomEvent("mnx-say", { detail: msg, cancelable: true });
    if (window.dispatchEvent(ev)) location.href = "/index.html?say=" + encodeURIComponent(msg);   // not in the chat: open it there
    shut();
  };
  tabPlay.addEventListener("click", () => show("play"));
  tabCode.addEventListener("click", () => show("code"));
  send.addEventListener("click", ask);
  say.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  close.addEventListener("click", shut);
  full.addEventListener("click", () => frame.requestFullscreen?.());
  box.addEventListener("click", (e) => { if (e.target === box) shut(); });
  addEventListener("keydown", esc);
  document.body.append(box);
  frame.focus();
  api("/artifact?url=" + encodeURIComponent(url)).then((a) => {
    prefix = a.change || "";
    const names = Object.keys(a.files);
    if (!names.length) { code.replaceChildren(h("p", { class: "runner-note" }, "The code isn't available here (it lives on the Bee's own computer).")); return; }
    const pre = h("pre", { class: "runner-pre" });
    const pick = h("select", { class: "runner-file", "aria-label": "File" }, ...names.map((n) => h("option", { value: n }, n)));
    const copy = h("button", { type: "button", class: "btn" }, "Copy");
    const sync = () => { pre.textContent = a.files[pick.value]; };
    pick.addEventListener("change", sync);
    copy.addEventListener("click", () => navigator.clipboard?.writeText(a.files[pick.value]).then(() => { copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy"; }, 1200); }));
    code.replaceChildren(h("div", { class: "runner-codebar" }, pick, copy, h("small", {}, `${a.bee} wrote this`)), pre);
    sync();
  }).catch(() => code.replaceChildren(h("p", { class: "runner-note" }, "The code isn't available for this one.")));
}

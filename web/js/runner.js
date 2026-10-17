// The Sandbox player: runs a Bee-made game, site or 3D model (its private /d/ or public /p/ link) in a locked-down frame.
// Chroma already serves those pages with a CSP sandbox (their own opaque origin); the frame's sandbox attribute is a second
// lock, so a page can never reach the Hive's pages, your token or your other sites.
import { h } from "./shared.js";

export const RUNNABLE = /^\/(?:d|p)\/[\w.-]+\/?/;

export function openRunner(url, title = "Sandbox") {
  if (!RUNNABLE.test(url)) return;
  document.querySelector(".runner")?.remove();
  const frame = h("iframe", {
    src: url, title, allow: "fullscreen; gamepad; autoplay",
    sandbox: "allow-scripts allow-forms allow-pointer-lock allow-popups allow-modals allow-downloads",
  });
  const close = h("button", { type: "button", class: "btn", "aria-label": "Close" }, "Close");
  const full = h("button", { type: "button", class: "btn" }, "Full screen");
  const tab = h("a", { class: "btn", href: url, target: "_blank", rel: "noopener noreferrer" }, "New tab");
  const box = h("div", { class: "runner", role: "dialog", "aria-label": title },
    h("div", { class: "runner-win" },
      h("div", { class: "runner-bar" }, h("b", {}, title), h("span", { class: "runner-lock" }, "Sandboxed · private link"), tab, full, close),
      frame));
  const shut = () => { box.remove(); removeEventListener("keydown", esc); };
  const esc = (e) => { if (e.key === "Escape" && document.fullscreenElement == null) shut(); };
  close.addEventListener("click", shut);
  full.addEventListener("click", () => frame.requestFullscreen?.());
  box.addEventListener("click", (e) => { if (e.target === box) shut(); });
  addEventListener("keydown", esc);
  document.body.append(box);
  frame.focus();
}

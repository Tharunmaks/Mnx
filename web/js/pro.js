// Mnx Hive — Plans: Free, or Pro paid with Razorpay Checkout. The Hive makes the order and checks the payment's
// signature; the page never sees a secret.

import { ICONS, api, connLabel, getSettings, h, installTokenPrompt, toast } from "./shared.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

let state = null;

const date = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
const spec = (p, re) => (p.computer.match(re) || [])[1];
const cpu = (p) => spec(p, /(\d+)\s*CPUs?/i);
const ram = (p) => spec(p, /(\d+)\s*GB\s*RAM/i);
const disk = (p) => spec(p, /(\d+)\s*GB\s*disk/i);
const vram = (p) => spec(p, /(\d+)\s*GB\s*VRAM/i);

function features(p) {
  return [
    h("li", {}, `${cpu(p)} CPUs, ${ram(p)} GB RAM and ${disk(p)} GB disk on every Bee`),
    h("li", {}, `Up to ${p.max_bees} Bees, each with its own chat`),
    h("li", { class: p.gpu ? "" : "no" }, p.gpu ? `The Hive's ${vram(p) || 16} GB GPU for your Bees' jobs` : "No GPU"),
    h("li", {}, "Vortex OS desktop on every Bee's computer"),
    h("li", {}, "Games, websites and 3D models in your private Sandbox"),
  ];
}

function compare(free, pro) {
  const rows = [
    ["CPUs per Bee", cpu(free), cpu(pro)], ["Memory per Bee", `${ram(free)} GB`, `${ram(pro)} GB`], ["Disk per Bee", `${disk(free)} GB`, `${disk(pro)} GB`],
    ["Bees", free.max_bees, pro.max_bees], ["GPU for jobs", free.gpu ? "Yes" : "No", pro.gpu ? `${vram(pro) || 16} GB` : "No"],
    ["Mnx Studio and Sandbox", "Yes", "Yes"], ["Apps, websites and games", "Yes", "Yes"],
  ];
  $("cmp").replaceChildren(
    h("div", { class: "cmp-row head" }, h("span", {}, "Feature"), h("span", {}, "Free"), h("span", {}, "Pro")),
    ...rows.map(([k, a, b]) => h("div", { class: "cmp-row" }, h("span", {}, k), h("span", {}, String(a)), h("span", { class: "pro-col" }, String(b)))));
}

async function load() {
  try { state = await api("/api/billing"); } catch (e) {
    $("status").replaceChildren(h("span", { class: "pill bad" }, h("i"), `Couldn't reach the Hive: ${e.message}`));
    connLabel($("conn"), "offline");
    return;
  }
  connLabel($("conn"), "online");
  const p = state.plan;
  const free = p.plans.find((x) => x.id === "free");
  const pro = p.plans.find((x) => x.id === "pro");
  $("freeList").replaceChildren(...features(free));
  $("proList").replaceChildren(...features(pro));
  $("proPrice").textContent = `₹${state.price_inr.toLocaleString("en-IN")}`;
  compare(free, pro);
  const isPro = p.plan === "pro";
  $("cardFree").classList.toggle("current", !isPro);
  $("cardPro").classList.toggle("current", isPro);
  $("freeNow").classList.toggle("hidden", isPro);
  $("proNow").classList.toggle("hidden", !isPro);

  const chips = [];
  const pay = $("pay");
  if (p.owner) {
    chips.push(h("span", { class: "pill honey" }, h("i"), "You own this Hive: always Pro"));
    pay.disabled = true;
    pay.textContent = "Always Pro";
  } else if (isPro) {
    const left = p.until ? Math.max(0, Math.ceil((p.until * 1000 - Date.now()) / 86_400_000)) : null;
    chips.push(h("span", { class: "pill honey" }, h("i"), p.until ? `Pro until ${date(p.until)}` : "You're on Pro"));
    if (left != null) chips.push(h("span", { class: "pill" }, `${left} day${left === 1 ? "" : "s"} left`));
    pay.disabled = !state.ready;
    pay.textContent = "Add 30 more days";
  } else {
    chips.push(h("span", { class: "pill ok" }, h("i"), "You're on Free"));
    pay.disabled = !state.ready;
    pay.textContent = `Get Pro · ₹${state.price_inr.toLocaleString("en-IN")}`;
  }
  if (state.test) chips.push(h("span", { class: "pill" }, "Test mode: no real money moves"));
  $("status").replaceChildren(...chips);
  $("payNote").textContent = state.ready ? "" : "Payments aren't switched on for this Hive yet.";
  $("ownerState").textContent = state.ready ? `Razorpay is set up (${state.test ? "test" : "live"} keys). Pro costs ₹${state.price_inr} for 30 days.`
    : "Razorpay isn't set up yet. On the Hive's machine, run:";
  $("hookUrl").textContent = `${location.origin}/razorpay/webhook`;
}

async function buy() {
  if (!window.Razorpay) { toast("Razorpay's checkout didn't load; check the connection and reload", "error"); return; }
  const pay = $("pay");
  pay.disabled = true;
  let order;
  try { order = await api("/api/billing/order", { method: "POST" }); } catch (e) { toast(e.message, "error"); pay.disabled = false; return; }
  const rzp = new window.Razorpay({
    key: order.key_id,
    amount: order.amount,
    currency: order.currency,
    order_id: order.order_id,
    name: "Mnx Hive",
    description: `Pro · ${order.days} days`,
    prefill: { name: getSettings().name || "" },
    theme: { color: "#f5a623" },
    handler: async (resp) => {
      try {
        const r = await api("/api/billing/confirm", { method: "POST", body: resp });
        toast(`You're on Pro until ${date(r.until)}. Your Bees are getting their Pro computers.`);
      } catch (e) {
        toast(`Paid, but the Hive couldn't confirm it yet: ${e.message}. It will be added when Razorpay tells the Hive.`, "error", 9000);
      }
      load();
    },
    modal: { ondismiss: () => { pay.disabled = false; } },
  });
  rzp.on("payment.failed", (resp) => {
    toast(`Payment didn't go through: ${resp.error?.description || "try again"}`, "error", 7000);
    pay.disabled = false;
  });
  rzp.open();
}

$("pay").addEventListener("click", buy);
installTokenPrompt(load);
load();

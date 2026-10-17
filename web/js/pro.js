// Mnx Hive — Plans: Free, or Pro paid with Razorpay Checkout. The Hive makes the order and checks the payment's
// signature; the page never sees a secret.

import { ICONS, api, connLabel, getSettings, h, installTokenPrompt, toast } from "./shared.js";

const $ = (id) => document.getElementById(id);
document.querySelectorAll("[data-icon]").forEach((el) => { el.outerHTML = ICONS[el.dataset.icon]; });

let state = null;

function date(ts) {
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function features(p) {
  return [`Every Bee: ${p.computer}`, `Up to ${p.max_bees} Bees, each with its own chat`,
    p.gpu ? "The Hive's GPU for your Bees' jobs" : "No GPU", "Vortex OS desktop on every Bee's computer"];
}

async function load() {
  try { state = await api("/api/billing"); } catch (e) { $("now").textContent = `Couldn't reach the Hive: ${e.message}`; connLabel($("conn"), "offline"); return; }
  connLabel($("conn"), "online");
  const p = state.plan;
  const free = p.plans.find((x) => x.id === "free");
  const pro = p.plans.find((x) => x.id === "pro");
  $("freeList").replaceChildren(...features(free).map((f) => h("li", {}, f)));
  $("proList").replaceChildren(...features(pro).map((f) => h("li", {}, f)));
  $("proPrice").textContent = `₹${state.price_inr.toLocaleString("en-IN")}`;
  $("cardFree").classList.toggle("current", p.plan === "free");
  $("cardPro").classList.toggle("current", p.plan === "pro");
  const pay = $("pay");
  if (p.owner) {
    $("now").textContent = "You own this Hive, so you're always on Pro.";
    pay.disabled = true;
    pay.textContent = "Always Pro";
  } else if (p.plan === "pro") {
    $("now").textContent = p.until ? `You're on Pro until ${date(p.until)}.` : "You're on Pro.";
    pay.disabled = !state.ready;
    pay.textContent = "Add 30 more days";
  } else {
    $("now").textContent = "You're on Free.";
    pay.disabled = !state.ready;
    pay.textContent = "Get Pro";
  }
  $("payNote").textContent = state.ready ? (state.test ? "Test mode: no real money moves." : "") : "Payments aren't switched on for this Hive yet.";
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

"""GPU Bee budget planner: "I have 1500 INR, which GPU should I rent?" answered with numbers.

For a money budget it lists what each rentable GPU buys: hours, the biggest model you could pretrain from scratch
in that time (compute-optimal, ~20 tokens per parameter, capped by memory), the biggest model a LoRA fine-tune fits,
and how many tokens a LoRA fine-tune of a given size gets through. Prices are RunPod's live prices when a
RUNPOD_API_KEY is saved, else a built-in table marked approximate. Exchange rates are approximate too.
"""

from __future__ import annotations

import math
import re

from . import architect

# ~35% of each GPU's dense bf16 tensor throughput, like architect.GPUS; the table there wins where both list a GPU.
EXTRA_GPUS = {
    "A40 48GB": (48, 52e12, 0.44),
    "RTX A6000 48GB": (48, 54e12, 0.49),
    "RTX A5000 24GB": (24, 39e12, 0.27),
    "L40 48GB": (48, 63e12, 0.89),
    "A10 24GB": (24, 44e12, 0.5),
}
TABLE = {**EXTRA_GPUS, **architect.GPUS}
# RunPod display names → our speed table (first match wins)
SPEED_RULES = [
    (r"\bH100\b.*\bPCIe\b", 265e12), (r"\bH100\b|\bH200\b", 350e12), (r"\bB200\b", 790e12), (r"\bA100\b", 110e12),
    (r"\bL40S\b", 120e12), (r"\bL40\b", 63e12), (r"RTX\s*6000\s*Ada", 100e12), (r"RTX\s*4090", 60e12), (r"RTX\s*3090", 30e12),
    (r"\bA40\b", 52e12), (r"RTX\s*A6000", 54e12), (r"RTX\s*A5000", 39e12), (r"\bA10G?\b", 44e12), (r"\bL4\b", 40e12), (r"\bT4\b", 12e12),
]
# approximate exchange rates to US dollars (only used to convert a budget; the card says so)
USD_PER = {"USD": 1.0, "INR": 1 / 85, "EUR": 1.10, "GBP": 1.30}
SYMBOL = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
SETUP_HOURS = 0.25        # boot, install, download the base model or data: paid but not training
LORA_BYTES_PER_PARAM = 2.5  # bf16 base weights plus LoRA state and activations at small batches
LORA_OVERHEAD_GB = 2.0
LORA_FLOPS_PER_TOKEN = 4  # forward 2N + backward through activations 2N (the base weights get no gradient)

_CUR = (r"(?P<cur>₹|rs\.?|inr|rupees?|\$|usd|us\s*dollars?|dollars?|bucks|€|eur|euros?|£|gbp|pounds?)")
_NUM = r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k|thousand|lakh|lakhs)?"
MONEY = re.compile(rf"{_CUR}\s*{_NUM}|{_NUM.replace('?P<num>', '?P<num2>').replace('?P<k>', '?P<k2>')}\s*{_CUR.replace('?P<cur>', '?P<cur2>')}\b", re.I)
ASKS_GPU = re.compile(r"\b(gpu|gpus|rent|renting|budget|afford|spend|which\s+(?:gpu|machine|server|card))\b", re.I)


def parse_money(text: str) -> tuple[float, str, tuple[int, int]] | None:
    """'1500 INR' → (1500.0, 'INR', span); '$20', '₹2k', '1.5 lakh rupees', '15 euros' work too."""
    m = MONEY.search(text or "")
    if not m:
        return None
    num = m.group("num") or m.group("num2")
    k = (m.group("k") or m.group("k2") or "").lower()
    cur = (m.group("cur") or m.group("cur2")).lower().replace(" ", "")
    amount = float(num.replace(",", "")) * {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5}.get(k, 1)
    code = ("INR" if cur.startswith(("₹", "rs", "inr", "rupee")) else "EUR" if cur.startswith(("€", "eur")) else
            "GBP" if cur.startswith(("£", "gbp", "pound")) else "USD")
    return amount, code, m.span()


def fmt_money(amount: float, code: str) -> str:
    sym = SYMBOL.get(code, "")
    return f"{sym}{amount:,.0f}" if amount >= 100 else f"{sym}{amount:,.2f}"


def speed_of(name: str) -> float | None:
    for pattern, speed in SPEED_RULES:
        if re.search(pattern, name, re.I):
            return speed
    return None


def table_gpus() -> list[dict]:
    return [{"name": n, "memory_gb": mem, "speed": sp, "price": price, "live": False} for n, (mem, sp, price) in TABLE.items()]


def lora_max_params(memory_gb: float) -> float:
    return max(0.0, (memory_gb * 0.9 - LORA_OVERHEAD_GB) * 1e9 / LORA_BYTES_PER_PARAM)


def pretrain_max_params(memory_gb: float) -> float:
    return memory_gb * 1e9 * 0.85 / architect.TRAIN_BYTES_PER_PARAM_GPU


def options(budget_usd: float, gpus: list[dict], lora_params: float | None = None) -> list[dict]:
    """What the budget buys on each GPU, most useful first (most training compute, then cheaper per hour)."""
    out = []
    for g in gpus:
        if not g.get("speed") or not g.get("price"):
            continue
        hours = budget_usd / g["price"]
        work = max(0.0, hours - SETUP_HOURS)
        flops = g["speed"] * work * 3600
        n_compute = math.sqrt(flops / (6 * architect.TOKENS_PER_PARAM)) if flops else 0.0
        n_mem = pretrain_max_params(g["memory_gb"])
        lora_n = lora_params or min(lora_max_params(g["memory_gb"]), 7e9)
        lora_fits = lora_n <= lora_max_params(g["memory_gb"])
        out.append({**g, "hours": hours, "work_hours": work, "flops": flops,
                    "pretrain_params": min(n_compute, n_mem), "pretrain_limit": "memory" if n_mem < n_compute else "time",
                    "lora_max": lora_max_params(g["memory_gb"]), "lora_params": lora_n, "lora_fits": lora_fits,
                    "lora_tokens": flops / (LORA_FLOPS_PER_TOKEN * lora_n) if lora_fits and lora_n else 0.0})
    out.sort(key=lambda o: (-o["flops"], o["price"]))
    return out


def recommend(opts: list[dict], need_gb: float = 0.0) -> dict | None:
    """The GPU to rent: among those with enough memory and at least an hour of real work, the cheapest per hour whose
    training compute is within 15% of the best (more hours leave room for mistakes and restarts)."""
    ok = [o for o in opts if o["memory_gb"] >= need_gb and o["work_hours"] >= 1]
    if not ok:
        return None
    best = max(o["flops"] for o in ok)
    close = [o for o in ok if o["flops"] >= 0.85 * best]
    return min(close, key=lambda o: (o["price"], -o["memory_gb"]))


def fmt_hours(h: float) -> str:
    return f"{h:.1f} h" if h < 10 else f"{h:.0f} h" if h < 72 else f"{h / 24:.0f} days"


def pretrain_cost(params: float, g: dict) -> dict:
    """Hours and dollars to pretrain `params` compute-optimally on one such GPU (None hours when it doesn't fit)."""
    if params > pretrain_max_params(g["memory_gb"]):
        return {"fits": False}
    hours = 6 * params * architect.TOKENS_PER_PARAM * params / g["speed"] / 3600 + SETUP_HOURS
    return {"fits": True, "hours": hours, "usd": hours * g["price"]}


# ---------------------------------------------------------------- the chat
async def on_budget(task, text: str) -> None:
    from . import providers, status
    from .lab import lab
    status.set_busy("gpu")
    try:
        bee = await task.emit("bee_created", bee="GPU", text="is pricing GPUs for your budget")
        money = parse_money(text)
        rest = text if not money else text[:money[2][0]] + " " + text[money[2][1]:]
        target = architect.parse_size(rest) if re.search(r"\b\d+(?:\.\d+)?\s*[mb]\b|param|model", rest, re.I) else None
        lora = bool(re.search(r"fine[\s-]?tun|finetun|lora|adapt", rest, re.I))
        key = lab._secrets.get("RUNPOD_API_KEY")
        gpus, live = table_gpus(), False
        s = await task.step("searching", "Asking RunPod for today's prices" if key else "Using the built-in price table (no RunPod key saved)",
                            bee="GPU", parent=bee)
        if key:
            try:
                found = await providers.runpod_gpu_types(key)
                rows = [{"name": f"{g['name']} {g['memory_gb']}GB", "memory_gb": g["memory_gb"], "speed": speed_of(g["name"]), "price": g["price"],
                         "live": True, "id": g["id"], "stock": g.get("stock")} for g in found]
                if any(r["speed"] for r in rows):
                    gpus, live = rows, True
                await task.finish_step(s, text=f"{len(found)} GPU types priced live, {sum(1 for r in rows if r['speed'])} with a known speed")
            except providers.ProviderError as exc:
                await task.finish_step(s, status="error", text=f"{exc}; using the built-in table instead"[:200])
        else:
            await task.finish_step(s, text=f"{len(gpus)} common GPUs at typical community-cloud prices (approximate)")
        if not money:
            per_hour = sorted((g for g in gpus if g["speed"]), key=lambda g: g["price"])
            await task.emit("result", parent=bee, service="GPU", title="What GPUs cost per hour" + ("" if live else " (approximate)"), badge="Prices",
                            details=[[g["name"], f"${g['price']:.2f}/h · {g['memory_gb']} GB"] for g in per_hour[:12]])
            await task.answer("Tell me your budget (for example “I have 1500 INR, which GPU should I rent?”) and I'll work out what each GPU buys you.")
            return
        amount, code, _ = money
        usd = amount * USD_PER[code]
        lora_params = target if (target and lora) else None
        opts = options(usd, gpus, lora_params)
        if not opts:
            await task.answer("I couldn't price any GPU I know the speed of.")
            return
        need_gb = 0.0
        if target:
            need_gb = ((target * LORA_BYTES_PER_PARAM / 1e9 + LORA_OVERHEAD_GB) / 0.9 if lora
                       else target * architect.TRAIN_BYTES_PER_PARAM_GPU / 1e9 / 0.85)
        pick = recommend(opts, need_gb)
        conv = "" if code == "USD" else f" ≈ ${usd:,.2f} (approximate rate)"
        rows = []
        for o in opts[:10]:
            line = (f"${o['price']:.2f}/h → {fmt_hours(o['hours'])} · pretrain up to {architect.fmt_params(o['pretrain_params'])} "
                    f"({'memory' if o['pretrain_limit'] == 'memory' else 'time'}-limited) · LoRA fine-tune up to {architect.fmt_params(o['lora_max'])}")
            if o["lora_fits"]:
                line += f" · LoRA on {architect.fmt_params(o['lora_params'])}: {architect.fmt_params(o['lora_tokens'])} tokens"
            rows.append([f"{o['name']}{'' if o['live'] else ''}", line])
        await task.emit("result", parent=bee, service="GPU", badge="Plan",
                        title=f"What {fmt_money(amount, code)}{conv} rents" + ("" if live else " (approximate prices)"),
                        details=rows, text=(f"Each line: price per hour, hours your budget buys, the largest model you could pretrain from scratch "
                                            f"in that time (about {architect.TOKENS_PER_PARAM} tokens per parameter), the largest a LoRA fine-tune fits, and how many "
                                            f"tokens a LoRA fine-tune gets through. {SETUP_HOURS * 60:.0f} minutes of each rental go to booting and downloads."))
        lines = []
        if pick:
            lines.append(f"Rent the {pick['name']}: {fmt_money(amount, code)} buys about {fmt_hours(pick['hours'])} at ${pick['price']:.2f}/h"
                         + ("" if live else " (typical price; check before renting)") + ".")
            lines.append(f"That's enough to pretrain a model of about {architect.fmt_params(pick['pretrain_params'])} from scratch, "
                         f"or to LoRA fine-tune a model up to {architect.fmt_params(pick['lora_max'])}"
                         + (f" (a {architect.fmt_params(pick['lora_params'])} model gets through about {architect.fmt_params(pick['lora_tokens'])} tokens of your data)." if pick["lora_fits"] else "."))
            bigger = [o for o in opts if o["memory_gb"] > pick["memory_gb"] and o["work_hours"] >= 1]
            if bigger:
                b = max(bigger, key=lambda o: o["flops"])
                lines.append(f"Pick the {b['name']} instead if you need more memory: it LoRA fine-tunes up to {architect.fmt_params(b['lora_max'])}, "
                             f"for about {fmt_hours(b['hours'])}.")
        else:
            lines.append("None of these GPUs gives at least an hour of real work with enough memory for that on this budget.")
        if target and lora:
            fits = [o for o in opts if o["lora_fits"]]
            if pick and pick["lora_fits"]:
                hours_per_m = LORA_FLOPS_PER_TOKEN * target * 1e6 / pick["speed"] / 3600
                lines.append(f"For a LoRA fine-tune of a {architect.fmt_params(target)} model the {pick['name']} handles about "
                             f"{architect.fmt_params(1e6 / hours_per_m)} tokens an hour: a few thousand example chats (a few million tokens) "
                             f"take {'under an hour' if hours_per_m * 3 < 1 else 'about ' + fmt_hours(hours_per_m * 3)}, so the budget leaves plenty of room to try again.")
            elif not fits:
                lines.append(f"A {architect.fmt_params(target)} model is too big for a LoRA fine-tune on any single GPU here.")
        if target and not lora:
            costs = [(o, pretrain_cost(target, o)) for o in opts]
            fit = [(o, c) for o, c in costs if c["fits"]]
            if fit:
                o, c = min(fit, key=lambda oc: oc[1]["usd"])
                verdict = "fits your budget" if c["usd"] <= usd else f"is about {c['usd'] / usd:.0f}× your budget"
                lines.append(f"Pretraining a {architect.fmt_params(target)} model properly ({architect.fmt_params(target * architect.TOKENS_PER_PARAM)} tokens) "
                             f"takes about {architect.fmt_time(c['hours'] * 3600)} on one {o['name']}, ${c['usd']:,.0f}: that {verdict}.")
                fo, fc = min(fit, key=lambda oc: oc[1]["hours"])
                if fo is not o:
                    lines.append(f"The fastest single GPU is the {fo['name']}: about {architect.fmt_time(fc['hours'] * 3600)} for ${fc['usd']:,.0f}.")
            else:
                lines.append(f"A {architect.fmt_params(target)} model doesn't fit one GPU for pretraining; it needs several (say “train a "
                             f"{architect.fmt_params(target)} model” and the GPU Bee plans a cluster).")
        lines.append("Billing runs while the machine is on: say “stop renting” the moment training ends."
                     + ("" if key else " Save RUNPOD_API_KEY in the Lab's Secrets and I'll use live prices and can rent it for you."))
        await task.answer(" ".join(lines))
    finally:
        status.clear("gpu")

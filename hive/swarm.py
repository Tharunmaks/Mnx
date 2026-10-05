"""Creating a model in chat with a swarm of Bees (the flow from the sketch).

"Create a 101.3 billion parameter model AI" →
  Queen       plans the work and hands it out
  Reader Bee  identifies the request; asks "Reason and what should the AI do?" when the purpose is missing
  Architect   designs the transformer, saves the blueprint in the Lab (always, so nothing is lost)
  Browser Bee searches for data and the code languages the model should know
  Data Bee    gathers the text (datasets, your files, web pages) and adds it to the model
  GPU Bee     opens cloud Docker and GPUs: this server, a connected server, all of them as a cluster, or rents one.
              It never stops at the blueprint: when nothing fits it asks for more servers, a rental, or a smaller size
  Coding Bee  writes the files of the run (train.py, serve.py, config, README) — the chat shows them with Show
  Cloud Bee   trains it (live card; keeps going when the app is closed)
  Eval Bee    tests it with N prompts when training ends and posts "Your model is ready"

The biggest model the Hive creates is 500B parameters (architect.MAX_PARAMS); bigger requests are capped.
"add data to my model" → Data Bee gathers more text, Cloud Bee continues training it.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path

from . import architect, status, system
from .cloud_chat import BEE as CLOUD, _gpu_targets, add_gpu_server, ask
from .core import Task
from .lab import MODELS, RECIPES, UPLOADS, lab

QUEEN, READER, ARCHITECT, BROWSER, DATA, GPU, CODING, EVAL = "Queen", "Reader", "Architect", "Browser", "Data", "GPU", "Coding", "Eval"
SWARM = ("reader", "architect", "browser", "data", "gpu", "coding", "cloud", "eval")
MAX_URLS = 15
MAX_ROUNDS = 6  # how many times the GPU Bee asks for more hardware before giving up

# Public datasets by purpose (name, config, text column). Streamed, so size doesn't matter.
LANG = {"tamil": "ta", "hindi": "hi", "malayalam": "ml", "telugu": "te", "kannada": "kn", "bengali": "bn", "marathi": "mr",
        "french": "fr", "spanish": "es", "german": "de", "japanese": "ja", "chinese": "zh", "arabic": "ar", "portuguese": "pt"}
CODE_LANGS = ["python", "javascript", "typescript", "html", "css", "java", "c", "c++", "c-sharp", "go", "rust", "php", "ruby",
              "sql", "shell", "kotlin", "swift", "dart", "scala", "lua", "perl", "julia", "haskell", "markdown", "json", "yaml",
              "dockerfile", "makefile", "cmake", "powershell"]
CODE_WORDS = re.compile(r"cod(e|ing|er)|program|python|javascript|software|developer|script|html|website|app builder", re.I)


def is_code(purpose: str) -> bool:
    return bool(CODE_WORDS.search(purpose or ""))


def languages_in(purpose: str) -> list[str]:
    """Code languages named in the request; a sensible default set for "a maximum coding AI"."""
    p = (purpose or "").lower()
    found = [l for l in CODE_LANGS if re.search(r"(?<![\w+#])" + re.escape(l) + r"(?![\w+#])", p)]
    if "c#" in p and "c-sharp" not in found:
        found.append("c-sharp")
    if "js" in p.split() and "javascript" not in found:
        found.append("javascript")
    return found or ["python", "javascript", "html"]


def suggest_dataset(purpose: str) -> tuple[str, str, str, str]:
    p = (purpose or "").lower()
    for word, code in LANG.items():
        if word in p:
            return "wikimedia/wikipedia", f"20231101.{code}", "text", f"{word.title()} Wikipedia"
    if re.search(r"stor(y|ies)|tale|kid|child|bedtime", p):
        return "roneneldan/TinyStories", "", "text", "TinyStories (simple children's stories)"
    if is_code(p):
        return "bigcode/the-stack-smol", "", "content", "The Stack (code, one folder per language)"
    if re.search(r"chat|assistant|helper|conversation|answer", p):
        return "HuggingFaceH4/no_robots", "", "messages", "Human-written chats (no_robots)"
    if re.search(r"poem|poetry|song|lyric", p):
        return "merve/poetry", "", "content", "Poetry"
    return "wikimedia/wikipedia", "20231101.en", "text", "English Wikipedia"


# ---------- hardware descriptions ----------
def _target_dict(t: dict) -> dict:
    """What the architect's estimator needs to know about a training target."""
    stats = system.stats()
    if t["id"] == "local":
        gpus = stats["gpus"] if lab.gpu_local else []
        return {"gpu": bool(gpus), "gpus": [{"name": g["name"], "memory_gb": g["memory_mb"] / 1024} for g in gpus],
                "cores": stats["cpus"], "ram_gb": (stats["memory"]["available"] or 0) / 1e9}
    info = (lab.targets.get(t["id"]) or {}).get("info") or {}
    gpus = []
    for g in info.get("gpus") or []:
        m = re.search(r"(\d+)\s*MiB", g)
        gpus.append({"name": g.split(",")[0], "memory_gb": int(m.group(1)) / 1024 if m else 0})
    ram = info.get("ram_mb")
    return {"gpu": bool(gpus), "gpus": gpus, "cores": int(info.get("cpus") or 1) if str(info.get("cpus", "")).isdigit() else 1,
            "ram_gb": int(ram) / 1024 if ram and str(ram).isdigit() else 0}


def _merge_targets(ts: list[dict]) -> dict:
    """Several servers trained together as one cluster: all their GPUs (or cores and RAM) added up."""
    gpus = [g for t in ts for g in t["gpus"]]
    return {"gpu": bool(gpus) and all(t["gpu"] for t in ts), "gpus": gpus,
            "cores": sum(t["cores"] for t in ts), "ram_gb": sum(t["ram_gb"] for t in ts)}


def _largest_that_fits(target: dict) -> tuple[str, float] | None:
    for label, n in reversed(architect.SIZES):
        if architect.estimate(architect.design(n), target=target)["here"]["fits"]:
            return label, n
    return None


def _short(label: str) -> str:
    return label.split(" · ")[0]


def _gpu_words(tinfo: dict) -> str:
    if tinfo["gpu"]:
        names = {}
        for g in tinfo["gpus"]:
            names[g["name"]] = names.get(g["name"], 0) + 1
        return ", ".join(f"{c}× {n}" for n, c in names.items())
    return f"{tinfo['cores']} CPU cores, {tinfo['ram_gb']:.0f} GB RAM"


# ---------- data gathering (Browser Bee + Data Bee) ----------
async def gather_data(task: Task, parent: str, purpose: str, arch_params: float | None, langs: list[str] | None = None) -> dict | None:
    """Ask what to learn from, collect it, and report how much there is.

    Returns {"uploads", "use_sample", "hf": {...}, "label", "chars", "files"} or None on cancel.
    """
    ds, cfg, col, ds_label = suggest_dataset(purpose)
    code = is_code(purpose) and ds == "bigcode/the-stack-smol"
    fields = [
        {"name": "source", "label": "Where should the text come from?", "input": "select", "value": "hf",
         "options": [{"value": "hf", "label": f"A public dataset · suggested: {ds_label}"},
                     {"value": "upload", "label": "My own file (.txt, .md, .jsonl, .csv)"},
                     {"value": "web", "label": "Web pages (the Browser Bee reads them)"},
                     {"value": "sample", "label": "Demo stories (built in, ~400 KB)"}]},
        {"name": "hf_dataset", "label": "Hugging Face dataset", "value": ds, "show_if": {"source": "hf"}},
    ]
    if code:
        fields.append({"name": "languages", "label": "Code languages (comma separated)", "value": ", ".join(langs or languages_in(purpose)),
                       "show_if": {"source": "hf"}})
    else:
        fields.append({"name": "hf_config", "label": "Config (optional)", "value": cfg, "show_if": {"source": "hf"}})
    fields += [
        {"name": "max_examples", "label": "Rows to use" + (" per language" if code else ""), "input": "number", "value": "20000",
         "show_if": {"source": "hf"}},
        {"name": "file", "label": "Your file", "input": "file", "accept": ".txt,.md,.jsonl,.csv", "show_if": {"source": "upload"}},
        {"name": "urls", "label": "Page addresses, one per line (up to 15)", "input": "textarea",
         "placeholder": "https://en.wikipedia.org/wiki/Kochi", "show_if": {"source": "web"}},
    ]
    reply = await ask(task, "clarify", parent=parent, title="Data Bee: what should the model learn from?", fields=fields,
                      text="More text makes a better model. A tiny model is fine with a few MB; bigger ones want millions of documents.",
                      actions=[{"id": "ok", "label": "Use this data"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "ok":
        return None
    v = reply["values"]
    source = v.get("source") or "hf"
    out = {"uploads": [], "use_sample": False, "hf": {}, "files": [], "chars": 0, "label": ""}
    if source == "hf":
        name = (v.get("hf_dataset") or "").strip()
        if not re.fullmatch(r"[\w.-]+(/[\w.-]+)?", name):
            await task.answer("That dataset name doesn't look like a Hugging Face id. Ask me again with a valid one.")
            return None
        rows = max(100, int(float(v.get("max_examples") or 20000)))
        if code and name == ds:
            chosen = [l.strip().lower() for l in (v.get("languages") or "").split(",") if l.strip()] or languages_in(purpose)
            chosen = [l for l in chosen if re.fullmatch(r"[\w+#.-]+", l)][:30]
            sources = [{"name": name, "data_dir": f"data/{l}", "column": col, "max_examples": rows} for l in chosen]
            out["hf"] = {"hf_dataset": "", "hf_config": "", "hf_datasets": json.dumps(sources), "max_examples": rows}
            for l in chosen:
                s = await task.step("fetching", f"Browser Bee: adding {l} code ({ds_label.split(' (')[0]}, up to {rows:,} files)",
                                    bee=BROWSER, parent=parent)
                await task.finish_step(s)
            out["label"] = f"{ds_label.split(' (')[0]}: {', '.join(chosen)} · up to {rows:,} files each"
            out["chars"] = rows * len(chosen) * 3000
        else:
            out["hf"] = {"hf_dataset": name, "hf_config": (v.get("hf_config") or "").strip(), "max_examples": rows,
                         "text_column": col if name == ds else ""}
            out["label"] = f"{name}{' (' + out['hf']['hf_config'] + ')' if out['hf']['hf_config'] else ''}, up to {rows:,} rows"
            out["chars"] = rows * 2500  # rough: a Wikipedia-style row
        s = await task.step("fetching", f"Will stream {out['label']} from Hugging Face during training", bee=DATA, parent=parent)
        await task.finish_step(s)
    elif source == "upload":
        if not v.get("file"):
            await task.answer("You chose to upload a file but didn't pick one. Ask again and choose the file.")
            return None
        path = lab.take_upload(v["file"])
        out["uploads"] = [v["file"]]
        out["chars"] = path.stat().st_size
        out["label"] = f"your file {path.name} ({out['chars'] / 1e6:.1f} MB)"
        s = await task.step("reading", f"Got {path.name}: {out['chars'] / 1e6:.1f} MB of text", bee=DATA, parent=parent)
        await task.finish_step(s)
    elif source == "web":
        urls = [u.strip() for u in (v.get("urls") or "").splitlines() if u.strip()][:MAX_URLS]
        urls = [u if "://" in u else "https://" + u for u in urls]
        if not urls:
            await task.answer("No page addresses were given. Ask again and paste some links.")
            return None
        folder = UPLOADS / ("web-" + task.id)
        folder.mkdir(parents=True, exist_ok=True)
        from .cells import cells
        browser = cells.get("browser", "Browser").browser
        got = 0
        for i, url in enumerate(urls, 1):
            s = await task.step("reading", f"Browser Bee is reading {url[:70]}", bee=BROWSER, parent=parent)
            try:
                page = await browser.do("goto", url=url)
                text = (await browser.do("read"))["text"]
                if len(text.strip()) < 200:
                    raise RuntimeError("almost no text on the page")
                (folder / f"page_{i:02d}.txt").write_text(f"# {page.get('title') or url}\n\n{text}")
                out["chars"] += len(text)
                got += 1
                await task.finish_step(s, text=f"Read {page.get('title') or url} ({len(text) / 1000:.0f}K characters)", bee=BROWSER)
            except Exception as exc:
                await task.finish_step(s, status="error", text=f"Couldn't read {url[:60]}", detail=str(exc)[:120])
        if not got:
            await task.answer("I couldn't read any of those pages. Check the links and ask again.")
            return None
        out["files"] = sorted(folder.iterdir())
        out["label"] = f"{got} web page{'s' if got > 1 else ''} ({out['chars'] / 1e6:.2f} MB)"
    else:
        out["use_sample"] = True
        out["chars"] = (RECIPES / "llm-pretrain" / "sample.txt").stat().st_size
        out["label"] = "the demo stories"
    tokens = out["chars"] / 4
    note = f"About {tokens / 1e6:.1f}M tokens" if tokens >= 1e6 else f"About {tokens / 1e3:.0f}K tokens"
    if arch_params:
        want = architect.TOKENS_PER_PARAM * arch_params
        passes = want / max(tokens, 1)
        note += (f" · a {architect.fmt_params(arch_params)} model would like ~{architect.fmt_params(want)} tokens, "
                 + (f"so it will make {passes:.0f} passes over this data (3 max)" if passes > 1 else "so one pass is plenty"))
    await task.step("done", f"Data ready: {out['label']}. {note}", bee=DATA, parent=parent)
    return out


# ---------- hardware (GPU Bee) ----------
async def acquire_hardware(task: Task, parent: str, arch: dict, allow_smaller: bool = True) -> tuple[dict | None, dict]:
    """Find hardware that holds the model: a server, every connected server as a cluster, or a rented GPU.

    Keeps asking (connect more, rent, cluster) until something fits or the user picks a smaller size / keeps the blueprint.
    Returns (plan, arch): plan = {"nodes", "target_id", "label", "tinfo", "gpu"} or None when nothing starts.
    """
    for _round in range(MAX_ROUNDS):
        targets = await asyncio.to_thread(_gpu_targets)
        ok = [t for t in targets if t["ok"]]
        ssh_ok = [t for t in ok if t["id"] != "local"]
        candidates: list[dict] = []
        for t in ok:
            tinfo = _target_dict(t)
            candidates.append({"nodes": None, "target_id": t["id"], "label": t["label"], "tinfo": tinfo, "gpu": t["gpu"]})
        if len(ssh_ok) >= 2:
            tinfo = _merge_targets([_target_dict(t) for t in ssh_ok])
            candidates.append({"nodes": [t["id"] for t in ssh_ok], "target_id": ssh_ok[0]["id"], "tinfo": tinfo, "gpu": tinfo["gpu"],
                               "label": f"All {len(ssh_ok)} connected servers together · "
                                        + (f"{len(tinfo['gpus'])} GPUs" if tinfo["gpu"] else "CPU only")})
        for c in candidates:
            c["fits"] = architect.estimate(arch, target=c["tinfo"])["here"]["fits"]
        fitting = [c for c in candidates if c["fits"]]
        options = [{"value": c["target_id"] if not c["nodes"] else "__cluster__",
                    "label": c["label"] + ("" if c["fits"] else " · too small")} for c in candidates]
        options.append({"value": "__add__", "label": "➕ Connect another GPU server (SSH)…"})
        if "RUNPOD_API_KEY" in lab.secret_names():
            options.append({"value": "__rent__", "label": "☁ Rent GPUs on RunPod…"})
        est = architect.estimate(arch)
        ref = est["reference"]
        need = architect.gpus_needed(arch["params"])
        machines = -(-need // 8)
        text = f"A {arch['params_text']} model needs about {est['train_memory_gpu']} of memory to train"
        if need > 1:
            text += f": {need}× 80 GB GPUs ({machines} machine{'s' if machines > 1 else ''} of 8)"
        text += f". A good setup is {ref['count']}× {ref['gpu']} for about {ref['time']} (~{ref['cost']} to rent)."
        if fitting:
            text += f" {_short(fitting[0]['label'])} can hold it."
        default = fitting[0] if fitting else candidates[0]
        reply = await ask(task, "clarify", parent=parent, title="GPU Bee: where should it train?", text=text,
                          fields=[{"name": "where", "label": "Train on", "input": "select",
                                   "value": "__cluster__" if default["nodes"] else default["target_id"], "options": options}],
                          actions=[{"id": "ok", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "ok":
            return None, arch
        choice = reply["values"].get("where") or options[0]["value"]
        if choice == "__add__":
            await add_gpu_server(task, parent)
            continue
        if choice == "__rent__":
            await rent_gpu(task, parent, arch)
            continue
        plan = next((c for c in candidates if (c["nodes"] and choice == "__cluster__") or (not c["nodes"] and c["target_id"] == choice)), default)
        if plan["fits"]:
            where = f"{len(plan['nodes'])} servers" if plan["nodes"] else _short(plan["label"])
            s = await task.step("opening", f"Opening Docker on {where} · {_gpu_words(plan['tinfo'])}", bee=GPU, parent=parent)
            await task.finish_step(s)
            return plan, arch

        # Doesn't fit: never stop at the blueprint — offer more hardware, a rental, or a smaller model.
        here = architect.estimate(arch, target=plan["tinfo"])["here"]
        alt = _largest_that_fits(plan["tinfo"]) if allow_smaller else None
        s = await task.step("error", f"{_short(plan['label'])} can't hold a {arch['params_text']} model yet", bee=GPU, parent=parent,
                            detail=f"needs {here['needed']}, has {here['memory']}")
        actions = [{"id": "more", "label": "Connect more servers"}]
        if "RUNPOD_API_KEY" in lab.secret_names():
            actions.append({"id": "rent", "label": "Rent GPUs"})
        if alt:
            actions.append({"id": "smaller", "label": f"Make it {alt[0]} instead"})
        actions.append({"id": "later", "label": "Keep the blueprint, train later"})
        actions.append({"id": "cancel", "label": "Cancel", "style": "danger"})
        reply = await ask(task, "confirm", parent=parent, title="Need more hardware", service=GPU,
                          text=(f"Training needs {here['needed']} but {_short(plan['label'])} has {here['memory']}. "
                                f"Connect more servers and I'll use them all together as one cluster, rent GPUs, or go smaller."),
                          details=[["Needs", here["needed"]], ["Available", here["memory"]],
                                   ["GPUs needed", f"{need}× 80 GB" + (f" ({machines} machines of 8)" if machines > 1 else "")]],
                          actions=actions)
        if reply["action"] == "more":
            await add_gpu_server(task, parent)
        elif reply["action"] == "rent":
            await rent_gpu(task, parent, arch)
        elif reply["action"] == "smaller" and alt:
            arch = architect.design(alt[1])
            await task.step("done", f"Architect redesigned it as a {arch['params_text']} model", bee=ARCHITECT, parent=parent)
            if architect.estimate(arch, target=plan["tinfo"])["here"]["fits"]:  # same hardware, smaller model: go
                where = f"{len(plan['nodes'])} servers" if plan["nodes"] else _short(plan["label"])
                s = await task.step("opening", f"Opening Docker on {where} · {_gpu_words(plan['tinfo'])}", bee=GPU, parent=parent)
                await task.finish_step(s)
                return plan, arch
        elif reply["action"] == "later":
            await task.answer(f"The blueprint is saved in the Lab. A {arch['params_text']} model needs {need}× 80 GB GPUs; "
                              f"connect them (Cloud Bee → Servers) and say “train the {arch['params_text']} blueprint”.")
            return None, arch
        else:
            return None, arch
    await task.answer("I asked a few times and still have nowhere to train. The blueprint is saved; connect GPUs and ask again.")
    return None, arch


async def rent_gpu(task: Task, parent: str, arch: dict) -> str | None:
    from . import providers
    key = lab._secrets.get("RUNPOD_API_KEY")
    s = await task.step("searching", "Asking RunPod what GPUs are available", bee=GPU, parent=parent)
    try:
        gpus = await providers.runpod_gpu_types(key)
    except providers.ProviderError as exc:
        await task.finish_step(s, status="error", text=str(exc))
        await task.answer("RunPod didn't answer. Check the API key under Cloud Bee → Servers → Secrets.")
        return None
    need_gb = architect.TRAIN_BYTES_PER_PARAM_GPU * arch["params"] / 1e9
    await task.finish_step(s, text=f"{len(gpus)} GPU types available")
    opts = []
    for g in gpus:
        count = max(1, -(-need_gb // (g["memory_gb"] * 0.85)))
        if count > 8:
            continue
        opts.append({"value": f"{g['id']}|{int(count)}", "label": f"{int(count)}× {g['name']} ({g['memory_gb']} GB) · ${g['price'] * count:.2f}/h"})
    if not opts:
        await task.answer("No single RunPod machine (up to 8 GPUs) can hold a model this big. Rent several and I'll cluster them, "
                          "or pick a smaller size.")
        return None
    reply = await ask(task, "clarify", parent=parent, title="Rent GPUs", fields=[
        {"name": "gpu", "label": "Machine", "input": "select", "value": opts[0]["value"], "options": opts},
        {"name": "hours", "label": "Expected hours (for the cost estimate)", "input": "number", "value": "2"}],
        text="Billing starts when the machine is ready and stops when you say 'stop renting' or press Stop in the Lab.",
        actions=[{"id": "rent", "label": "Rent it"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "rent":
        return None
    gpu_id, count = (reply["values"].get("gpu") or opts[0]["value"]).split("|")
    chosen = next(g for g in gpus if g["id"] == gpu_id)
    hours = max(0.5, float(reply["values"].get("hours") or 2))
    cost = chosen["price"] * int(count) * hours
    reply = await ask(task, "confirm", parent=parent, title="Confirm the rental", service=GPU,
                      details=[["Machine", f"{count}× {chosen['name']}"], ["Price", f"${chosen['price'] * int(count):.2f} per hour"],
                               ["Estimated", f"${cost:.2f} for {hours:g} hours"]],
                      text="This spends real money on your RunPod account.",
                      actions=[{"id": "go", "label": "Rent now"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "go":
        return None
    s = await task.step("running", f"Renting {count}× {chosen['name']}", bee=GPU, parent=parent)
    try:
        pod = await lab.rent_runpod(gpu_id, int(count))
        await task.finish_step(s, text=f"Rented pod {pod['id']} · waiting for it to boot")
        s = await task.step("waiting", "Waiting for the machine to accept SSH (usually 1–3 minutes)", bee=GPU, parent=parent)
        rec = await lab.attach_pod(pod["id"])
    except Exception as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer("Renting didn't work. Check runpod.io to make sure nothing is left running.")
        return None
    if rec["status"] != "ready":
        await task.finish_step(s, status="error", text="The machine is up but SSH login failed; see Cloud Bee → Servers")
        return None
    await task.finish_step(s, text="GPU machine connected")
    return rec["target"]


# ---------- the files the Coding Bee writes ----------
def _readme(name: str, purpose: str, arch: dict, data_label: str, where: str, tokens: int, test_prompts: int) -> str:
    lines = [f"# {name}", "", f"Purpose: {purpose or 'general language model'}", "",
             "## Architecture (Qwen2-style transformer)", ""]
    for k, v in architect.summary_lines(arch):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Training", "", f"- Data: {data_label}", f"- Tokens: {tokens:,}", f"- Hardware: {where}",
              f"- Test: {test_prompts:,} held-out prompts (next-token accuracy and perplexity)", "",
              "## Files", "", "- train.py — builds the tokenizer and the model, trains it (DDP/FSDP, multi-node), tests it, saves it",
              "- serve.py — the chat API used by Run it", "- params.json — the settings of this run",
              "- architect.py — the design helper", "- run.sh / launch.sh — how the Cloud Bee starts it in Docker or over SSH", ""]
    return "\n".join(lines)


# ---------- create a model ----------
async def create_model(task: Task, size: float | None, purpose: str, test_prompts: int | None = None) -> None:
    status.set_busy(*SWARM)
    started_run = False
    try:
        plan_step = await task.step("planning", "Plan: Reader → Architect → Browser & Data Bees → GPU Bee → Coding Bee → "
                                                "Cloud Bee trains → Eval Bee tests", bee=QUEEN)
        await task.finish_step(plan_step)

        # Reader Bee: understand the request, ask what's missing.
        reader = await task.emit("bee_created", bee=READER, text="is identifying the request")
        s = await task.step("thinking", "Identifying the request", bee=READER, parent=reader)
        await task.finish_step(s)
        purpose = (purpose or "").strip()
        if size and not purpose:
            reply = await ask(task, "clarify", parent=reader, title="That's a big one — I'll try. Reason and what should the AI do?",
                              text="Tell me what the model is for so the Bees can pick the right data and tests.",
                              fields=[{"name": "purpose", "label": "What should the AI do?", "placeholder": "A maximum coding AI",
                                       "required": True}],
                              actions=[{"id": "ok", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
            if reply["action"] != "ok":
                await task.answer("Okay, nothing started.")
                return
            purpose = (reply["values"].get("purpose") or "").strip()[:120]
        if not size:
            reply = await ask(task, "clarify", parent=reader, title="How big should the model be?", fields=[
                {"name": "size", "label": "Size", "input": "select", "value": "10M",
                 "options": [{"value": l, "label": f"{l} parameters"} for l, _ in architect.SIZES]},
                {"name": "purpose", "label": "What should the AI do?", "value": purpose, "placeholder": "A maximum coding AI"}],
                text="Tiny models (1M–50M) train on a CPU in minutes. 1B+ needs GPUs; above ~30B you need many (the GPU Bee finds them). "
                     "500B is the maximum.",
                actions=[{"id": "ok", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
            if reply["action"] != "ok":
                await task.answer("Okay, nothing started.")
                return
            size = architect.parse_size(reply["values"].get("size") or "10M") or 1e7
            purpose = (reply["values"].get("purpose") or purpose).strip()[:120]
        size, capped = architect.clamp_size(size)
        n_test = max(0, int(test_prompts or 1000))
        s = await task.step("done", f"Request: a {architect.fmt_params(size)} model · {purpose or 'general purpose'}"
                            + (" · you asked for more, but 500B is the maximum, so 500B it is" if capped else ""), bee=READER, parent=reader)
        status.clear("reader")

        # Architect: design it and keep the blueprint.
        arch_bee = await task.emit("bee_created", bee=ARCHITECT, text="is designing")
        s = await task.step("thinking", f"Designing a {architect.fmt_params(size)} transformer", bee=ARCHITECT, parent=arch_bee)
        arch = architect.design(size)
        targets = await asyncio.to_thread(_gpu_targets)
        best = next((t for t in targets if t["ok"]), targets[0])
        est = architect.estimate(arch, target=_target_dict(best))
        here = est["here"]
        await task.finish_step(s, text=f"Blueprint ready: {arch['params_text']} parameters, {arch['layers']} layers, width {arch['hidden']}")
        details = architect.summary_lines(arch) + [
            ["Weights", est["weights"]], ["Training memory", f"about {est['train_memory_gpu']} on GPUs"],
            ["Training tokens", f"{architect.fmt_params(est['tokens'])} (20 per parameter)"],
            [f"On {_short(best['label'])}", (f"fits · about {here['time']}" if here["fits"]
                                             else f"doesn't fit: needs {here['needed']}, has {here['memory']}")],
            ["Good setup", f"{est['reference']['count']}× {est['reference']['gpu']} · {est['reference']['time']} · ~{est['reference']['cost']}"],
        ]
        await task.emit("result", parent=arch_bee, service=ARCHITECT, title=f"Blueprint: {arch['params_text']} model", badge="Designed",
                        details=details, text=("A Qwen2-style transformer: RMSNorm, SwiGLU, rotary attention"
                                               + (", grouped-query attention" if arch["kv_heads"] != arch["heads"] else "") + "."))
        bp = lab.save_blueprint(f"{arch['params_text']} {purpose or 'model'}", arch, est, purpose)
        await task.step("done", "Blueprint saved in the Lab (Models tab)", bee=ARCHITECT, parent=arch_bee)
        status.clear("architect")

        # Browser Bee: find data (and the code languages) for this purpose.
        browser_bee = await task.emit("bee_created", bee=BROWSER, text="is searching the web")
        s = await task.step("searching", f"Searching for data about “{purpose or 'general text'}”", bee=BROWSER, parent=browser_bee)
        ds, cfg, col, ds_label = suggest_dataset(purpose)
        langs = languages_in(purpose) if is_code(purpose) else []
        await task.finish_step(s, text=f"Found {ds_label}" + (f" · code languages: {', '.join(langs)} (editable below)" if langs else ""))
        status.clear("browser")

        # Data Bee
        data_bee = await task.emit("bee_created", bee=DATA, text="is gathering text")
        data = await gather_data(task, data_bee, purpose, arch["params"], langs)
        if data is None:
            return
        status.clear("data")

        # GPU Bee: find hardware that holds it (never stops at the blueprint).
        gpu_bee = await task.emit("bee_created", bee=GPU, text="is opening cloud Docker and GPUs")
        plan, arch = await acquire_hardware(task, gpu_bee, arch)
        if plan is None:
            _drop(data)
            return
        status.clear("gpu")
        tinfo = plan["tinfo"]
        est = architect.estimate(arch, target=tinfo)
        tokens = min(est["tokens"], int(3 * data["chars"] / 4))
        here = architect.estimate(arch, tokens=tokens, target=tinfo)["here"]  # time for the tokens we'll actually use
        warnings = []
        if here["device"] == "CPU" and here["seconds"] > 6 * 3600:
            warnings.append(f"On CPU this would take about {here['time']}. A GPU server would be much faster.")
        if here["seconds"] > 30 * 86400:
            warnings.append(f"Even here it's about {here['time']}; connect more GPUs for a shorter run.")
        test_seconds = n_test * arch["seq_len"] * 2 * arch["params"] / (tinfo["cores"] * architect.CPU_FLOPS_PER_CORE if not tinfo["gpu"] else 1e14)
        if test_seconds > 3600:
            warnings.append(f"Testing with {n_test:,} prompts adds about {architect.fmt_time(test_seconds)}.")
        details = [["Model", f"{arch['params_text']} parameters, {arch['layers']} layers"], ["Data", data["label"]],
                   ["Training tokens", architect.fmt_params(tokens)], ["Train on", plan["label"]],
                   ["Estimated time", here["time"]], ["Memory needed", here["needed"]], ["Test", f"{n_test:,} held-out prompts"]]
        reply = await ask(task, "confirm", parent=gpu_bee, title="Start creating the model?", service=CLOUD, details=details,
                          text=" ".join(warnings) or "Looks good. Training keeps going on the server even if you close the app.",
                          actions=[{"id": "start", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "start":
            _drop(data)
            await task.answer("Okay, nothing started. The blueprint stays in the Lab.")
            return

        # Coding Bee: write the files of the run.
        coding_bee = await task.emit("bee_created", bee=CODING, text="is writing the files")
        s = await task.step("writing", "Writing train.py, serve.py, params.json, README.md", bee=CODING, parent=coding_bee)
        params = {"size": "custom", "custom_params": arch["params"], "tokenizer": "own", "tokens": tokens, "epochs_max": 3,
                  "seq_len": 0, "learning_rate": 0, "init_from": "", "max_examples": 20000, "hf_dataset": "", "hf_config": "",
                  "hf_datasets": "", "test_prompts": n_test,
                  "sample_prompts": "def |# " if is_code(purpose) else "Once upon a time|The", **data["hf"]}
        name = f"{arch['params_text']} {purpose.strip() or 'model'}"[:70]
        readme = _readme(name, purpose, arch, data["label"], plan["label"], tokens, n_test)
        try:
            files = [lab.take_upload(u) for u in data["uploads"]] + list(data["files"])
            run = lab.create_run("llm-pretrain", name, params, plan["target_id"], files, data["use_sample"], plan["gpu"], None, None,
                                 nodes=plan["nodes"], chat={"task": task.id, "chat": task.chat_id}, extra_files={"README.md": readme},
                                 blueprint=bp["id"])
        except ValueError as exc:
            await task.finish_step(s, status="error", text=f"Couldn't start: {exc}")
            await task.answer("Something about the setup didn't work. Fix it and ask me again; the blueprint is saved.")
            return
        finally:
            _drop(data)
        started_run = True
        job_files = lab.list_job_files(run["id"])
        shown = [f for f in job_files if not f["path"].startswith("data/")][:12]
        await task.finish_step(s, text=f"Wrote {len(job_files)} files: " + ", ".join(Path(f["path"]).name for f in shown))
        await task.emit("lab_files", parent=coding_bee, run=run["id"], files=shown)
        status.clear("coding")

        # Cloud Bee trains; Eval Bee reports when it ends (gateway → _report).
        cloud_bee = await task.emit("bee_created", bee=CLOUD, text="is training")
        where = f"a cluster of {len(plan['nodes'])} servers" if plan["nodes"] else _short(plan["label"])
        await task.step("running", f"Training started on {where}", bee=CLOUD, parent=cloud_bee, status="done")
        await task.emit("lab_run", id=f"lr-{run['id']}", parent=cloud_bee, run=run["id"])
        eval_bee = await task.emit("bee_created", bee=EVAL, text=f"will test it with {n_test:,} prompts when training ends")
        run["chat"]["eval_parent"] = eval_bee
        lab._save_run(run)
        await task.answer(f"Your {arch['params_text']} model is being created on {where}. The card updates live, training keeps going "
                          f"if you close the app, and the Eval Bee posts “your model is ready” here with the test results. "
                          "Then press Run it and talk to it.")
    finally:
        if started_run:
            status.clear("reader", "architect", "browser", "data", "gpu", "coding")
        else:
            status.clear(*SWARM)


def find_blueprint(query: str) -> dict | None:
    q = [w for w in re.sub(r"blueprint|model|the|my|train", " ", (query or "").lower()).split() if w]
    bps = [m for m in lab.models.values() if m.get("kind") == "blueprint" and not m.get("trained_model")]
    hits = [m for m in bps if all(w in m["name"].lower() for w in q)] or bps
    return max(hits, key=lambda m: m["created"]) if hits else None


# ---------- add data to a model ----------
def _find_model(query: str) -> dict | None:
    q = (query or "").lower().split()
    candidates = [m for m in lab.models.values() if m.get("kind") == "llm" and (MODELS / m["id"] / "files" / "model").is_dir()]
    if q:
        hits = [m for m in candidates if all(w in m["name"].lower() for w in q)]
        if hits:
            candidates = hits
    return max(candidates, key=lambda m: m["created"]) if candidates else None


async def add_data(task: Task, model_query: str, purpose: str) -> None:
    model = _find_model(model_query)
    if not model:
        await task.answer("I don't have a trained model to add data to yet. Create one first (“create a 10M model”) "
                          "or fine-tune one (“train Qwen 2.5 Coder”).")
        return
    status.set_busy("data", "gpu", "cloud", "eval")
    started_run = False
    try:
        plan_step = await task.step("planning", f"Plan: Data Bee gathers more text → GPU Bee finds hardware → Cloud Bee keeps training "
                                                f"{model['name']} → Eval Bee tests", bee=QUEEN)
        await task.finish_step(plan_step)
        arch = model.get("arch") or (model.get("result") or {})
        n_params = (model.get("result") or {}).get("params") or (arch.get("params") if isinstance(arch, dict) else None) or 1e7

        data_bee = await task.emit("bee_created", bee=DATA, text="is gathering text")
        data = await gather_data(task, data_bee, purpose or model.get("purpose") or "", n_params)
        if data is None:
            return
        status.clear("data")
        gpu_bee = await task.emit("bee_created", bee=GPU, text="is opening cloud Docker and GPUs")
        fake_arch = {"params": n_params, "params_text": architect.fmt_params(n_params), "seq_len": 512}
        plan, _ = await acquire_hardware(task, gpu_bee, fake_arch, allow_smaller=False)
        if plan is None:
            _drop(data)
            return
        status.clear("gpu")
        tokens = int(min(3 * data["chars"] / 4, n_params * 5))
        reply = await ask(task, "confirm", parent=gpu_bee, title=f"Keep training {model['name']}?", service=CLOUD,
                          details=[["Model", model["name"]], ["New data", data["label"]], ["Training tokens", architect.fmt_params(tokens)],
                                   ["Train on", plan["label"]]],
                          text="The model keeps everything it knows and learns the new text on top (continued training, lower learning rate).",
                          actions=[{"id": "start", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "start":
            _drop(data)
            await task.answer("Okay, nothing started.")
            return
        params = {"size": "custom", "custom_params": 0, "tokenizer": "own", "tokens": tokens, "epochs_max": 2, "seq_len": 0,
                  "learning_rate": 0, "max_examples": 20000, "hf_dataset": "", "hf_config": "", "hf_datasets": "",
                  "test_prompts": 200, "sample_prompts": "Once upon a time|The", **data["hf"]}
        cloud_bee = await task.emit("bee_created", bee=CLOUD, text="is training")
        s = await task.step("running", f"Continuing {model['name']} on {_short(plan['label'])}", bee=CLOUD, parent=cloud_bee)
        try:
            files = [lab.take_upload(u) for u in data["uploads"]] + list(data["files"])
            run = lab.create_run("llm-pretrain", f"{model['name']} + {data['label'].split(' (')[0]}"[:70], params, plan["target_id"], files,
                                 data["use_sample"], plan["gpu"], None, None, base_model_id=model["id"], nodes=plan["nodes"],
                                 chat={"task": task.id, "chat": task.chat_id})
        except ValueError as exc:
            await task.finish_step(s, status="error", text=f"Couldn't start: {exc}")
            await task.answer("Something about the setup didn't work. Fix it and ask me again.")
            return
        finally:
            _drop(data)
        started_run = True
        await task.finish_step(s, text="Training started")
        await task.emit("lab_run", id=f"lr-{run['id']}", parent=cloud_bee, run=run["id"])
        eval_bee = await task.emit("bee_created", bee=EVAL, text="will test it when training ends")
        run["chat"]["eval_parent"] = eval_bee
        lab._save_run(run)
        await task.answer(f"Adding the new data to {model['name']}. The result is saved as a new version, so the old one stays too.")
    finally:
        status.clear("data", "gpu") if started_run else status.clear("data", "gpu", "cloud", "eval")


def _drop(data: dict) -> None:
    for u in data.get("uploads", []):
        lab.drop_upload(u)
    for f in data.get("files", []):
        shutil.rmtree(Path(f).parent, ignore_errors=True)

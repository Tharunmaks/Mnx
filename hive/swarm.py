"""Creating a model in chat with several Bees.

"create a 50M parameter model for stories" →
  Queen       plans the work and hands it out
  Architect   designs the transformer and says what it takes to train
  Data Bee    gathers the text (Hugging Face dataset, your files, web pages via the Browser Bee)
  Cloud Bee   picks the hardware (this server, an SSH server, or rents a GPU) and trains
  Eval Bee    reports perplexity and sample text when the run finishes (on the live card)

"add data to my model" → Data Bee gathers more text, Cloud Bee continues training it.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path

from . import architect, system
from .cloud_chat import BEE as CLOUD, _gpu_targets, add_gpu_server, ask
from .core import Task
from .lab import MODELS, RECIPES, UPLOADS, lab

QUEEN, ARCHITECT, DATA, EVAL, BROWSER = "Queen", "Architect", "Data", "Eval", "Browser"
MAX_URLS = 15

# Public datasets by purpose (name, config, text column). Streamed, so size doesn't matter.
LANG = {"tamil": "ta", "hindi": "hi", "malayalam": "ml", "telugu": "te", "kannada": "kn", "bengali": "bn", "marathi": "mr",
        "french": "fr", "spanish": "es", "german": "de", "japanese": "ja", "chinese": "zh", "arabic": "ar", "portuguese": "pt"}


def suggest_dataset(purpose: str) -> tuple[str, str, str, str]:
    p = (purpose or "").lower()
    for word, code in LANG.items():
        if word in p:
            return "wikimedia/wikipedia", f"20231101.{code}", "text", f"{word.title()} Wikipedia"
    if re.search(r"stor(y|ies)|tale|kid|child|bedtime", p):
        return "roneneldan/TinyStories", "", "text", "TinyStories (simple children's stories)"
    if re.search(r"code|python|program|coder|software", p):
        return "codeparrot/codeparrot-clean-valid", "", "content", "Python code (codeparrot)"
    if re.search(r"chat|assistant|helper|conversation|answer", p):
        return "HuggingFaceH4/no_robots", "", "messages", "Human-written chats (no_robots)"
    if re.search(r"poem|poetry|song|lyric", p):
        return "merve/poetry", "", "content", "Poetry"
    return "wikimedia/wikipedia", "20231101.en", "text", "English Wikipedia"


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


def _largest_that_fits(target: dict) -> tuple[str, float] | None:
    for label, n in reversed(architect.SIZES):
        if architect.estimate(architect.design(n), target=target)["here"]["fits"]:
            return label, n
    return None


# ---------- data gathering (Data Bee) ----------
async def gather_data(task: Task, parent: str, purpose: str, arch_params: float | None) -> dict | None:
    """Ask what to learn from, collect it, and report how much there is.

    Returns {"uploads", "use_sample", "hf": {...}, "label", "chars", "files"} or None on cancel.
    """
    ds, cfg, col, ds_label = suggest_dataset(purpose)
    fields = [
        {"name": "source", "label": "Where should the text come from?", "input": "select", "value": "hf",
         "options": [{"value": "hf", "label": f"A public dataset · suggested: {ds_label}"},
                     {"value": "upload", "label": "My own file (.txt, .md, .jsonl, .csv)"},
                     {"value": "web", "label": "Web pages (the Browser Bee reads them)"},
                     {"value": "sample", "label": "Demo stories (built in, ~400 KB)"}]},
        {"name": "hf_dataset", "label": "Hugging Face dataset", "value": ds, "show_if": {"source": "hf"}},
        {"name": "hf_config", "label": "Config (optional)", "value": cfg, "show_if": {"source": "hf"}},
        {"name": "max_examples", "label": "Rows to use", "input": "number", "value": "20000", "show_if": {"source": "hf"}},
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
        out["hf"] = {"hf_dataset": name, "hf_config": (v.get("hf_config") or "").strip(), "max_examples": rows,
                     "text_column": col if name == ds else ""}
        out["label"] = f"{name}{' (' + out['hf']['hf_config'] + ')' if out['hf']['hf_config'] else ''}, up to {rows:,} rows"
        s = await task.step("fetching", f"Will stream {out['label']} from Hugging Face during training", bee=DATA, parent=parent)
        await task.finish_step(s)
        out["chars"] = rows * 2500  # rough: a Wikipedia-style row
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
            s = await task.step("reading", f"Browser Bee is reading {url[:70]}", bee=DATA, parent=parent)
            try:
                page = await browser.do("goto", url=url)
                text = (await browser.do("read"))["text"]
                if len(text.strip()) < 200:
                    raise RuntimeError("almost no text on the page")
                (folder / f"page_{i:02d}.txt").write_text(f"# {page.get('title') or url}\n\n{text}")
                out["chars"] += len(text)
                got += 1
                await task.finish_step(s, text=f"Read {page.get('title') or url} ({len(text) / 1000:.0f}K characters)", bee=DATA)
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
    s = await task.step("done", f"Data ready: {out['label']}. {note}", bee=DATA, parent=parent)
    return out


# ---------- hardware (Cloud Bee) ----------
async def pick_target(task: Task, parent: str, arch: dict) -> dict | None:
    targets = await asyncio.to_thread(_gpu_targets)
    best = next((t for t in targets if t["ok"]), targets[0])
    options = [{"value": t["id"], "label": t["label"]} for t in targets if t["ok"]]
    options.append({"value": "__add__", "label": "➕ Connect a GPU server (SSH)…"})
    if "RUNPOD_API_KEY" in lab.secret_names():
        options.append({"value": "__rent__", "label": "☁ Rent a GPU on RunPod…"})
    ref = architect.estimate(arch)["reference"]
    reply = await ask(task, "clarify", parent=parent, title="Cloud Bee: where should it train?",
                      text=f"For a {arch['params_text']} model a good setup is {ref['count']}× {ref['gpu']} "
                           f"(about {ref['time']}, ~{ref['cost']} to rent).",
                      fields=[{"name": "where", "label": "Train on", "input": "select", "value": best["id"], "options": options}],
                      actions=[{"id": "ok", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "ok":
        return None
    choice = reply["values"].get("where") or best["id"]
    if choice == "__add__":
        choice = await add_gpu_server(task, parent)
    elif choice == "__rent__":
        choice = await rent_gpu(task, parent, arch)
    if not choice:
        return None
    targets = await asyncio.to_thread(_gpu_targets)
    return next((t for t in targets if t["id"] == choice), best)


async def rent_gpu(task: Task, parent: str, arch: dict) -> str | None:
    from . import providers
    key = lab._secrets.get("RUNPOD_API_KEY")
    s = await task.step("searching", "Asking RunPod what GPUs are available", bee=CLOUD, parent=parent)
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
        await task.answer("No single RunPod machine (up to 8 GPUs) can hold a model this big. Pick a smaller size.")
        return None
    reply = await ask(task, "clarify", parent=parent, title="Rent a GPU", fields=[
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
    reply = await ask(task, "confirm", parent=parent, title="Confirm the rental", service=CLOUD,
                      details=[["Machine", f"{count}× {chosen['name']}"], ["Price", f"${chosen['price'] * int(count):.2f} per hour"],
                               ["Estimated", f"${cost:.2f} for {hours:g} hours"]],
                      text="This spends real money on your RunPod account.",
                      actions=[{"id": "go", "label": "Rent now"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "go":
        return None
    s = await task.step("running", f"Renting {count}× {chosen['name']}", bee=CLOUD, parent=parent)
    try:
        pod = await lab.rent_runpod(gpu_id, int(count))
        await task.finish_step(s, text=f"Rented pod {pod['id']} · waiting for it to boot")
        s = await task.step("waiting", "Waiting for the machine to accept SSH (usually 1–3 minutes)", bee=CLOUD, parent=parent)
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


# ---------- create a model ----------
async def create_model(task: Task, size: float | None, purpose: str) -> None:
    plan = await task.step("planning", "Plan: Architect designs it → Data Bee gathers text → Cloud Bee trains → Eval Bee checks",
                           bee=QUEEN)
    await task.finish_step(plan)

    if not size:
        reply = await ask(task, "clarify", title="How big should the model be?", fields=[
            {"name": "size", "label": "Size", "input": "select", "value": "10M",
             "options": [{"value": l, "label": f"{l} parameters"} for l, _ in architect.SIZES]}],
            text="Tiny models (1M–50M) train on a CPU in minutes and learn simple patterns. 1B+ needs GPUs. "
                 "Above ~30B you need many GPUs; the Architect will tell you how many.",
            actions=[{"id": "ok", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "ok":
            await task.answer("Okay, nothing started.")
            return
        size = architect.parse_size(reply["values"].get("size") or "10M") or 1e7

    # Architect
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
        [f"On {best['label'].split(' · ')[0]}", (f"fits · about {here['time']}" if here["fits"]
                                                 else f"doesn't fit: needs {here['needed']}, has {here['memory']}")],
        ["Good setup", f"{est['reference']['count']}× {est['reference']['gpu']} · {est['reference']['time']} · ~{est['reference']['cost']}"],
    ]
    await task.emit("result", parent=arch_bee, service=ARCHITECT, title=f"Blueprint: {arch['params_text']} model", badge="Designed",
                    details=details, text=("A Qwen2-style transformer: RMSNorm, SwiGLU, rotary attention"
                                           + (", grouped-query attention" if arch["kv_heads"] != arch["heads"] else "") + "."))

    # Data
    data_bee = await task.emit("bee_created", bee=DATA, text="is gathering text")
    data = await gather_data(task, data_bee, purpose, arch["params"])
    if data is None:
        return

    # Cloud
    cloud_bee = await task.emit("bee_created", bee=CLOUD, text="is on it")
    target = await pick_target(task, cloud_bee, arch)
    if target is None:
        _drop(data)
        await task.answer("Okay, nothing started.")
        return
    tinfo = _target_dict(target)
    est = architect.estimate(arch, target=tinfo)
    here = est["here"]
    warnings = []
    if not here["fits"]:
        alt = _largest_that_fits(tinfo)
        s = await task.step("error", f"{target['label'].split(' · ')[0]} can't hold a {arch['params_text']} model",
                            bee=CLOUD, parent=cloud_bee, detail=f"needs {here['needed']}, has {here['memory']}")
        actions = [{"id": "blueprint", "label": "Save the blueprint only"}]
        if alt:
            actions.insert(0, {"id": "smaller", "label": f"Train a {alt[0]} model instead"})
        actions.append({"id": "cancel", "label": "Cancel", "style": "danger"})
        reply = await ask(task, "confirm", parent=cloud_bee, title="Too big for this hardware", service=CLOUD,
                          text=f"Training a {arch['params_text']} model needs about {est['train_memory_gpu']} of GPU memory. "
                               f"A realistic setup is {est['reference']['count']}× {est['reference']['gpu']} for {est['reference']['time']} "
                               f"(~{est['reference']['cost']} to rent). Connect such a machine, or go smaller.",
                          details=[["Needs", here["needed"]], ["Available here", here["memory"]]], actions=actions)
        if reply["action"] == "smaller" and alt:
            arch = architect.design(alt[1])
            est = architect.estimate(arch, target=tinfo)
            here = est["here"]
            await task.step("done", f"Architect redesigned it as a {arch['params_text']} model", bee=ARCHITECT, parent=cloud_bee)
        elif reply["action"] == "blueprint":
            bp = lab.save_blueprint(f"{arch['params_text']} {purpose or 'model'} blueprint", arch, est, purpose)
            _drop(data)
            await task.emit("lab_model", id=f"lm-{bp['id']}", parent=cloud_bee, model=bp["id"])
            await task.answer("Saved the blueprint. When you have the hardware, connect it and say "
                              f"“train the {arch['params_text']} blueprint”.")
            return
        else:
            _drop(data)
            await task.answer("Okay, nothing started.")
            return
    if here["device"] == "CPU" and here["seconds"] > 6 * 3600:
        warnings.append(f"On CPU this would take about {here['time']}. A GPU server would be much faster.")
    if here["seconds"] > 30 * 86400:
        warnings.append(f"Even here it's about {here['time']}; consider more GPUs.")
    tokens = min(est["tokens"], int(3 * data["chars"] / 4))
    here = architect.estimate(arch, tokens=tokens, target=tinfo)["here"]  # time for the tokens we'll actually use
    warnings = [w for w in warnings if not w.startswith("On CPU")]
    if here["device"] == "CPU" and here["seconds"] > 6 * 3600:
        warnings.append(f"On CPU this would take about {here['time']}. A GPU server would be much faster.")
    details = [["Model", f"{arch['params_text']} parameters, {arch['layers']} layers"], ["Data", data["label"]],
               ["Training tokens", architect.fmt_params(tokens)], ["Train on", target["label"]],
               ["Estimated time", here["time"]], ["Memory needed", here["needed"]]]
    reply = await ask(task, "confirm", parent=cloud_bee, title="Start creating the model?", service=CLOUD, details=details,
                      text=" ".join(warnings) or "Looks good. Training keeps going on the server even if you close the app.",
                      actions=[{"id": "start", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "start":
        _drop(data)
        await task.answer("Okay, nothing started.")
        return
    params = {"size": "custom", "custom_params": arch["params"], "tokenizer": "own", "tokens": tokens, "epochs_max": 3,
              "seq_len": 0, "learning_rate": 0, "init_from": "", "max_examples": 20000, "hf_dataset": "", "hf_config": "",
              "sample_prompts": "Once upon a time|The", **data["hf"]}
    name = f"{arch['params_text']} {purpose.strip() or 'model'}"[:70]
    s = await task.step("running", f"Starting on {target['label'].split(' · ')[0]}", bee=CLOUD, parent=cloud_bee)
    try:
        files = [lab.take_upload(u) for u in data["uploads"]] + list(data["files"])
        run = lab.create_run("llm-pretrain", name, params, target["id"], files, data["use_sample"], target["gpu"], None, None)
    except ValueError as exc:
        await task.finish_step(s, status="error", text=f"Couldn't start: {exc}")
        await task.answer("Something about the setup didn't work. Fix it and ask me again.")
        return
    finally:
        _drop(data)
    await task.finish_step(s, text=f"Training started on {target['label'].split(' · ')[0]}")
    await task.emit("lab_run", id=f"lr-{run['id']}", parent=cloud_bee, run=run["id"])
    eval_bee = await task.emit("bee_created", bee=EVAL, text="will check it when training ends")
    await task.step("waiting", "Eval Bee: perplexity and sample text will appear on the card above", bee=EVAL, parent=eval_bee,
                    status="done")
    await task.answer(f"Your {arch['params_text']} model is being created. The card updates live, and training keeps going "
                      "if you close the app. When it's done, press Run it and talk to it right here.")


def find_blueprint(query: str) -> dict | None:
    q = [w for w in re.sub(r"blueprint|model|the|my", " ", (query or "").lower()).split() if w]
    bps = [m for m in lab.models.values() if m.get("kind") == "blueprint"]
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
    plan = await task.step("planning", f"Plan: Data Bee gathers more text → Cloud Bee keeps training {model['name']} → Eval Bee checks",
                           bee=QUEEN)
    await task.finish_step(plan)
    arch = model.get("arch") or (model.get("result") or {})
    n_params = (model.get("result") or {}).get("params") or (arch.get("params") if isinstance(arch, dict) else None)

    data_bee = await task.emit("bee_created", bee=DATA, text="is gathering text")
    data = await gather_data(task, data_bee, purpose, n_params)
    if data is None:
        return
    cloud_bee = await task.emit("bee_created", bee=CLOUD, text="is on it")
    target = await pick_target(task, cloud_bee, {"params": n_params or 1e7, "params_text": architect.fmt_params(n_params or 1e7)})
    if target is None:
        _drop(data)
        await task.answer("Okay, nothing started.")
        return
    tokens = int(min(3 * data["chars"] / 4, (n_params or 1e7) * 5))
    reply = await ask(task, "confirm", parent=cloud_bee, title=f"Keep training {model['name']}?", service=CLOUD,
                      details=[["Model", model["name"]], ["New data", data["label"]], ["Training tokens", architect.fmt_params(tokens)],
                               ["Train on", target["label"]]],
                      text="The model keeps everything it knows and learns the new text on top (continued training, lower learning rate).",
                      actions=[{"id": "start", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "start":
        _drop(data)
        await task.answer("Okay, nothing started.")
        return
    params = {"size": "custom", "custom_params": 0, "tokenizer": "own", "tokens": tokens, "epochs_max": 2, "seq_len": 0,
              "learning_rate": 0, "max_examples": 20000, "hf_dataset": "", "hf_config": "", "sample_prompts": "Once upon a time|The",
              **data["hf"]}
    s = await task.step("running", f"Continuing {model['name']} on {target['label'].split(' · ')[0]}", bee=CLOUD, parent=cloud_bee)
    try:
        files = [lab.take_upload(u) for u in data["uploads"]] + list(data["files"])
        run = lab.create_run("llm-pretrain", f"{model['name']} + {data['label'].split(' (')[0]}"[:70], params, target["id"], files,
                             data["use_sample"], target["gpu"], None, None, base_model_id=model["id"])
    except ValueError as exc:
        await task.finish_step(s, status="error", text=f"Couldn't start: {exc}")
        await task.answer("Something about the setup didn't work. Fix it and ask me again.")
        return
    finally:
        _drop(data)
    await task.finish_step(s, text="Training started")
    await task.emit("lab_run", id=f"lr-{run['id']}", parent=cloud_bee, run=run["id"])
    await task.answer(f"Adding the new data to {model['name']}. The result is saved as a new version, so the old one stays too.")


def _drop(data: dict) -> None:
    for u in data.get("uploads", []):
        lab.drop_upload(u)
    for f in data.get("files", []):
        shutil.rmtree(Path(f).parent, ignore_errors=True)

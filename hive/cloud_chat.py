"""The Cloud Bee in chat: "train Qwen 2.5 Coder" → find it on Hugging Face, pick a GPU,
ask for data, confirm, train, then run it and talk to it, all inside the conversation.

Until the Mnx brain is connected, requests are understood with fixed patterns (no AI):
  train / fine-tune / teach <model> [on <dataset>]   start a training
  status of my training / how is training going      show live runs
  stop training                                      stop the running training
  run my model / chat with <model>                   start a trained model and talk to it
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from . import architect, hf, system
from .core import Task
from .lab import lab

BEE = "Cloud"
ASK_TIMEOUT = 30 * 60  # stop waiting for an answer after 30 minutes

TRAIN = re.compile(r"\b(train|retrain|fine[\s-]?tune|finetune|teach|customi[sz]e)\b", re.I)
STATUS = re.compile(r"\b(status|progress|how(?:'s| is| are)|check)\b.*\b(train\w*|run|runs|fine[\s-]?tun\w*)\b", re.I)
STOP = re.compile(r"\b(stop|cancel|kill|abort)\b.*\b(train\w*|run|fine[\s-]?tun\w*)\b", re.I)
CREATE = re.compile(r"\b(create|build|make|design|generate|invent)\b.*\b(model|llm|ai|transformer|brain|network)\b"
                    r"|\bfrom scratch\b|\bnew (?:ai|model|llm)\b", re.I)
ADD_DATA = re.compile(r"\b(add|feed|give|load|put|upload)\b.*\b(data|dataset|knowledge|text|examples|docs|documents|pages)\b"
                      r"|\b(continue|keep|resume) training\b|\b(train|teach)\b.*\bmore\b", re.I)
PURPOSE = re.compile(r"\b(?:for|about|that (?:writes|knows|speaks|does|can)|which|to write|on)\s+(.+)$", re.I)
RENT_STOP = re.compile(r"\b(stop|end|cancel|kill)\b.*\b(rent\w*|pod|gpu machine|cloud machine)\b", re.I)
ON_PHONE = re.compile(r"\b(?:on|to|in|using)\s+(?:my\s+|the\s+)?(?:phone|mobile|android|termux)\b", re.I)
ON_LPU = re.compile(r"\b(?:on|to|in|using|with)\s+(?:my\s+|the\s+|an?\s+)?(?:(\d+)\s+)?(?:virtual\s+)?"
                    r"(?:lpus?(?:\s+chips?)?|groq(?:\s+chips?)?|virtual\s+(?:chips?|silicon)|chip\s+simulator)\b", re.I)
ACT = r"\b(run|train|teach|start|test|chat|generate|serve|simulate|fine[\s-]?tune)\b"
RUN = re.compile(r"\b(run|start|deploy|serve|launch|chat with|talk to)\b.*\b(model|mnx|qwen|llama|gemma|phi|smollm|mistral|deepseek)\w*\b", re.I)
SIZE_IN_TEXT = re.compile(r"\b(\d+(?:\.\d+)?)\s*([bm])\b", re.I)
HF_ID = re.compile(r"\b([A-Za-z0-9][\w.-]*/[\w.-]+)\b")
FILLER = re.compile(r"\b(a|an|the|my|model|models|llm|ai|please|for me|can you|could you|i want to|i want|want to|"
                    r"let'?s|hugging\s*face|huggingface|from|import|and|it|new|own|version|of)\b", re.I)


@dataclass
class Intent:
    kind: str
    model_query: str = ""
    size_b: float | None = None
    dataset: str = ""
    params: float | None = None  # exact parameter count for "create"
    purpose: str = ""
    test_prompts: int | None = None
    engine: str = ""  # "lpu": the phone runs the streamed layers on the virtual chip
    steps: int | None = None


def parse(text: str) -> Intent | None:
    t = " ".join(text.strip().split())
    from . import bee_chat, budget
    bi = bee_chat.parse(t)
    if bi:
        return Intent("bee", dataset=t, purpose=bi["kind"])
    if (budget.parse_money(t) and (budget.ASKS_GPU.search(t) or re.search(r"\b(train|pretrain|fine[\s-]?tune|finetune|what can i|i have|i've got|my budget)\b", t, re.I))) \
            or (re.search(r"\b(which|what|best|cheapest)\s+(?:cloud\s+)?gpus?\b", t, re.I) and not RENT_STOP.search(t)):
        return Intent("gpu_budget", dataset=t)
    pm = ON_PHONE.search(t)
    lm = ON_LPU.search(t)
    if pm and re.search(ACT, t, re.I):
        mode = "train" if re.search(r"\b(train|teach|fine[\s-]?tune)\b", t, re.I) else "run"
        rest = t[:pm.start()]
        sm = re.search(r"(\d+)\s*steps?", t, re.I)
        pr = re.search(r"(?:prompt|say|with|starting with)\s*[:\"“]\s*(.+?)[\"”]?$", t[pm.end():], re.I)
        rest = re.sub(r"^.*?\b(?:run|train|teach|start|test|chat with|generate with|serve|fine[\s-]?tune)\b", "", rest, flags=re.I)
        rest = re.sub(r"\d+\s*steps?", "", rest, flags=re.I)
        if lm and lm.start() < pm.start():
            rest = rest[:lm.start()]
        query = " ".join(FILLER.sub(" ", ON_LPU.sub(" ", rest)).split())
        return Intent("on_phone", model_query=query, purpose=mode, dataset=(pr.group(1).strip() if pr else ""),
                      size_b=float(sm.group(1)) if sm else None, engine="lpu" if lm else "")
    if lm and re.search(ACT, t, re.I):
        mode = "train" if re.search(r"\b(train|teach|fine[\s-]?tune)\b", t, re.I) else "run"
        cm = re.search(r"(\d+)\s*(?:lpu\s+|virtual\s+)?chips?\b", t, re.I)
        chips = int(lm.group(1) or (cm.group(1) if cm else 1))
        sm = re.search(r"(\d+)\s*steps?\b", t, re.I)
        pr = re.search(r"(?:prompt|say|with|starting with)\s*[:\"“]\s*(.+?)[\"”]?$", t[lm.end():], re.I)
        rest = re.sub(r"^.*?\b(?:run|train|teach|start|test|chat with|generate with|serve|simulate|fine[\s-]?tune)\b", "", t[:lm.start()], flags=re.I)
        rest = re.sub(r"\d+\s*(?:lpu\s+|virtual\s+)?chips?\b|\d+\s*steps?\b|\bfor\b", "", rest, flags=re.I)
        query = " ".join(FILLER.sub(" ", rest).split())
        return Intent("on_lpu", model_query=query, purpose=mode, dataset=(pr.group(1).strip() if pr else ""), size_b=float(chips),
                      steps=int(sm.group(1)) if sm else None)
    if RENT_STOP.search(t):
        return Intent("rent_stop")
    if STOP.search(t):
        return Intent("stop")
    if ADD_DATA.search(t) and not CREATE.search(t):
        mq = re.search(r"\b(?:to|into)\s+(?:my\s+)?(.+?)(?:\s+(?:model|llm|ai))?(?:\s*[.!?]|$|\s+(?:about|with|from)\b)", t, re.I)
        pm = PURPOSE.search(t)
        return Intent("add_data", model_query=(mq.group(1) if mq else "").strip(), purpose=(pm.group(1) if pm else "").strip())
    n_params = architect.parse_size(t) if re.search(r"param|\bparams\b|\b\d+(?:\.\d+)?\s*[kmbt]\b", t, re.I) else None
    if CREATE.search(t) and (n_params or re.search(r"from scratch|new (?:ai|model|llm)|\b(create|build|design|invent)\b", t, re.I)):
        tm = re.search(r"test\w*\s+(?:it\s+)?(?:with|on)\s+(\d+(?:[.,]\d+)?)\s*(k|m|thousand|million)?\s*prompts?", t, re.I)
        n_test = None
        if tm:
            n_test = float(tm.group(1).replace(",", ".")) * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get((tm.group(2) or "").lower(), 1)
            t = t[:tm.start()] + t[tm.end():]
        pm = PURPOSE.search(t)
        purpose = (pm.group(1) if pm else "").strip()
        purpose = re.sub(r"^(?:a|an|the)\s+", "", purpose)
        purpose = re.sub(r"\s*(?:,|and)?\s*$", "", purpose)
        return Intent("create", params=n_params, purpose=purpose[:120], test_prompts=int(n_test) if n_test else None)
    if STATUS.search(t) and not TRAIN.match(t):
        return Intent("status")
    m = TRAIN.search(t)
    if m:
        rest = t[m.end():]
        dataset = ""
        dm = re.search(r"\b(?:on|with|using|from)\s+(?:the\s+|my\s+)?(?:dataset\s+)?([\w.-]+/[\w.-]+)", rest, re.I)
        if dm:
            dataset = dm.group(1)
            rest = rest[:dm.start()]
        else:
            rest = re.split(r"\b(?:on|with|using|to|so|for)\b", rest, maxsplit=1, flags=re.I)[0]
        ids = HF_ID.findall(rest)
        size = None
        sm = SIZE_IN_TEXT.search(rest)
        if sm:
            size = float(sm.group(1)) / (1000 if sm.group(2).lower() == "m" else 1)
        if ids:
            query = ids[0]
        else:
            query = SIZE_IN_TEXT.sub(" ", rest)
            query = FILLER.sub(" ", query)
            query = re.sub(r"[^\w.\s-]", " ", query)
            query = " ".join(query.split())
        return Intent("train", query, size, dataset)
    if RUN.search(t):
        name = re.sub(r"^.*?\b(?:run|start|deploy|serve|launch|chat with|talk to)\b", "", t, flags=re.I)
        return Intent("run", " ".join(FILLER.sub(" ", name).split()))
    return None


async def handle(task: Task) -> bool:
    """Handle the message if it's a Cloud Bee request. Returns False when it isn't one."""
    intent = parse(task.text)
    if not intent:
        return False
    try:
        if intent.kind == "train" and re.search(r"blueprint", task.text, re.I):
            from . import swarm
            bp = swarm.find_blueprint(intent.model_query)
            if not bp:
                await task.answer("I don't have a blueprint by that name. Say “create a 1B model” to design one.")
            else:
                await swarm.create_model(task, bp["arch"]["params"], bp.get("purpose") or "", intent.test_prompts)
        elif intent.kind == "train":
            await train(task, intent)
        elif intent.kind == "status":
            await status(task)
        elif intent.kind == "stop":
            await stop(task)
        elif intent.kind == "run":
            await run_model(task, intent)
        elif intent.kind == "create":
            from . import swarm
            await swarm.create_model(task, intent.params, intent.purpose, intent.test_prompts)
        elif intent.kind == "add_data":
            from . import swarm
            await swarm.add_data(task, intent.model_query, intent.purpose)
        elif intent.kind == "rent_stop":
            await stop_renting(task)
        elif intent.kind == "on_phone":
            from . import phone_lab
            await phone_lab.on_phone(task, intent.purpose, intent.model_query, prompt=intent.dataset,
                                     steps=int(intent.size_b) if intent.size_b else 10, engine=intent.engine)
        elif intent.kind == "bee":
            from . import bee_chat
            await bee_chat.handle(task, intent.dataset)
        elif intent.kind == "gpu_budget":
            from . import budget
            await budget.on_budget(task, intent.dataset)
        elif intent.kind == "on_lpu":
            from . import lpu_lab
            await lpu_lab.on_lpu(task, intent.model_query, prompt=intent.dataset, chips=int(intent.size_b or 1), mode=intent.purpose,
                                 steps=intent.steps or 10)
    except asyncio.TimeoutError:
        await task.answer("I waited a long time for an answer, so I stopped here. Ask again whenever you're ready.")
    return True


async def ask(task: Task, card: str, **kw) -> dict:
    return await asyncio.wait_for(task.ask(card, **kw), ASK_TIMEOUT)


# ---------- train ----------
def _gpu_targets() -> list[dict]:
    """Every place training can run, with its GPU and memory, best first."""
    stats = system.stats()
    out = []
    for t in lab.target_views():
        if t["kind"] == "local":
            gpus = stats["gpus"] if lab.gpu_local else []
            mem_gb = round(sum(g["memory_mb"] for g in gpus) / 1024, 1) if gpus else round((stats["memory"]["available"] or 0) / 1024**3, 1)
            label = (f"This server · GPU {gpus[0]['name']} ({mem_gb} GB)" if gpus
                     else f"This server · CPU only ({stats['cpus']} cores, {mem_gb} GB RAM free)")
            out.append({"id": "local", "label": label, "gpu": bool(gpus), "mem_gb": mem_gb, "ok": True})
        else:
            gpus = (t.get("info") or {}).get("gpus") or []
            mem_mb = 0
            for g in gpus:  # "NVIDIA A100-SXM4-80GB, 81920 MiB"
                m = re.search(r"(\d+)\s*MiB", g)
                mem_mb += int(m.group(1)) if m else 0
            mem_gb = round(mem_mb / 1024, 1) if mem_mb else None
            if gpus:
                label = f"{t['name']} · GPU {gpus[0].split(',')[0]}" + (f" ({mem_gb} GB)" if mem_gb else "")
            else:
                ram = (t.get("info") or {}).get("ram_mb")
                label = f"{t['name']} · CPU only" + (f" ({round(int(ram) / 1024)} GB RAM)" if ram and str(ram).isdigit() else "")
                mem_gb = round(int(ram) / 1024, 1) if ram and str(ram).isdigit() else None
            out.append({"id": t["id"], "label": label + ("" if t.get("status") == "ok" else " · not reachable"),
                        "gpu": bool(gpus), "mem_gb": mem_gb, "ok": t.get("status") == "ok"})
    out.sort(key=lambda x: (not x["ok"], not x["gpu"], -(x["mem_gb"] or 0)))
    return out


def _pick_model(models: list[dict], target: dict, want: float | None) -> dict:
    if want:
        return min(models, key=lambda m: abs((m["size_b"] or 1e9) - want))
    budget = target["mem_gb"] or 0
    fitting = [m for m in models if m["size_b"] and (hf.memory_needed_gb(m["size_b"], target["gpu"]) or 0) <= budget]
    if not target["gpu"]:  # CPU: keep it small enough to finish today
        fitting = [m for m in fitting if m["size_b"] <= 1.5] or fitting
    return (fitting[-1] if fitting else min(models, key=lambda m: m["size_b"] or 1e9))


def _model_label(m: dict) -> str:
    size = f"{m['size_b']:g}B" if m.get("size_b") else "size unknown"
    extra = " · needs licence + HF_TOKEN" if m.get("gated") else ""
    downloads = f" · {m['downloads']:,} downloads" if m.get("downloads") else ""
    return f"{m['id']} ({size}{downloads}{extra})"


async def train(task: Task, intent: Intent) -> None:
    bee = await task.emit("bee_created", bee=BEE, text="is on it")
    if not intent.model_query:
        await task.answer("Which model should I train? For example: “train Qwen 2.5 Coder” or “fine-tune Qwen/Qwen2.5-1.5B-Instruct”.")
        return

    s = await task.step("searching", f"Looking for “{intent.model_query}” on Hugging Face", bee=BEE, parent=bee)
    models, source = await hf.find_models(intent.model_query)
    if not models:
        await task.finish_step(s, status="error", text=f"No trainable model matched “{intent.model_query}”")
        await task.answer("I couldn't find that model. Try its exact Hugging Face id, like Qwen/Qwen2.5-Coder-1.5B-Instruct.")
        return
    where_from = {"hub": "on Hugging Face", "catalog": "in my list of popular models (Hugging Face search wasn't reachable)",
                  "exact": "by its exact id"}[source]
    await task.finish_step(s, text=f"Found {len(models)} model{'s' if len(models) > 1 else ''} {where_from}")

    s = await task.step("fetching", "Checking GPUs and servers", bee=BEE, parent=bee)
    targets = await asyncio.to_thread(_gpu_targets)
    best = next((t for t in targets if t["ok"]), targets[0])
    gpu_note = (f"GPU ready: {best['label']}" if best["gpu"]
                else "No GPU connected yet: I can train small models on CPU, or you can connect a GPU server")
    await task.finish_step(s, text=gpu_note)

    default = _pick_model(models, best, intent.size_b)
    dataset = intent.dataset or hf.suggested_dataset(default["id"])
    fields = [
        {"name": "model", "label": "Model", "input": "select", "value": default["id"],
         "options": [{"value": m["id"], "label": _model_label(m)} for m in models]},
        {"name": "where", "label": "Train on", "input": "select", "value": best["id"],
         "options": [{"value": t["id"], "label": t["label"]} for t in targets if t["ok"]]
                    + [{"value": "__add__", "label": "➕ Connect a GPU server…"}]},
        {"name": "data", "label": "Training data", "input": "select", "value": "hf",
         "options": [{"value": "hf", "label": "A Hugging Face dataset"},
                     {"value": "upload", "label": "Upload my own .jsonl file"},
                     {"value": "sample", "label": "Tiny demo data (8 chats about Mnx)"}]},
        {"name": "hf_dataset", "label": "Hugging Face dataset", "value": dataset, "show_if": {"data": "hf"}},
        {"name": "max_examples", "label": "Use up to N examples", "input": "number", "value": "2000", "show_if": {"data": "hf"}},
        {"name": "file", "label": "Your .jsonl file", "input": "file", "accept": ".jsonl", "show_if": {"data": "upload"}},
        {"name": "epochs", "label": "Epochs (passes over the data)", "input": "number", "value": "1"},
    ]
    reply = await ask(task, "clarify", parent=bee, title="Set up the training", fields=fields,
                      text="I picked a size that fits the hardware. Change anything you like.",
                      actions=[{"id": "next", "label": "Continue"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "next":
        await task.answer("Okay, cancelled. Nothing was started.")
        return
    v = reply["values"]
    model = next((m for m in models if m["id"] == v.get("model")), default)

    target_id = v.get("where") or best["id"]
    if target_id == "__add__":
        target_id = await add_gpu_server(task, bee)
        if not target_id:
            return
        targets = await asyncio.to_thread(_gpu_targets)
    target = next((t for t in targets if t["id"] == target_id), best)

    # data
    uploads, use_sample, hf_dataset = [], False, ""
    data = v.get("data") or "hf"
    if data == "upload":
        if not v.get("file"):
            await task.answer("You chose to upload a file but didn't pick one. Ask again and choose your .jsonl file.")
            return
        uploads = [v["file"]]
        data_label = "your uploaded file"
    elif data == "sample":
        use_sample, data_label = True, "the tiny demo data"
    else:
        hf_dataset = (v.get("hf_dataset") or "").strip()
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", hf_dataset):
            await task.answer("That dataset name doesn't look like a Hugging Face id (owner/name). Ask again with a valid one.")
            return
        data_label = f"{hf_dataset} (up to {v.get('max_examples') or 2000} examples)"

    size = model.get("size_b")
    need = hf.memory_needed_gb(size, target["gpu"])
    warnings = []
    if not target["gpu"] and (size or 99) > 1.5:
        warnings.append("This runs on CPU, so a model this size will be very slow (likely days). A 0.5B–1.5B model, or a GPU server, is better.")
    if need and target["mem_gb"] and need > target["mem_gb"]:
        warnings.append(f"It needs about {need} GB but {target['label'].split(' · ')[0]} has {target['mem_gb']} GB, so it may run out of memory.")
    if model.get("gated"):
        warnings.append("This model is gated: accept its licence on Hugging Face and save HF_TOKEN in Cloud Bee → Servers → Secrets.")
    big = (size or 0) >= 3
    params = {
        "base_model": model["id"], "epochs": max(1, min(int(float(v.get("epochs") or 1)), 50)),
        "hf_dataset": hf_dataset, "max_examples": max(10, int(float(v.get("max_examples") or 2000))),
        "batch_size": 1 if big else 2, "grad_accum": 8 if big else 4,
        "max_length": 1024, "lora_r": 16, "merge": True,
    }
    details = [["Model", model["id"]], ["Size", f"{size:g}B parameters" if size else "unknown"],
               ["Train on", target["label"]], ["Data", data_label], ["Epochs", str(params["epochs"])],
               ["Method", "LoRA fine-tune, merged into one model"]]
    if need:
        details.append(["Memory needed", f"about {need} GB"])
    reply = await ask(task, "confirm", parent=bee, title="Start training?", service=BEE, details=details,
                      text=" ".join(warnings) or "Everything looks good. Training keeps going on the server even if you close the app.",
                      actions=[{"id": "start", "label": "Start training"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "start":
        for u in uploads:
            lab.drop_upload(u)
        await task.answer("Okay, I didn't start anything.")
        return

    s = await task.step("running", f"Starting the training on {target['label'].split(' · ')[0]}", bee=BEE, parent=bee)
    try:
        files = [lab.take_upload(u) for u in uploads]
        short = model["id"].split("/")[-1]
        run = lab.create_run("llm-finetune", f"{short} · {data_label.split(' (')[0]}", params, target["id"], files,
                             use_sample, target["gpu"], None, None)
    except ValueError as exc:
        await task.finish_step(s, status="error", text=f"Couldn't start: {exc}")
        await task.answer("Something about the setup didn't work. Fix it and ask me again.")
        return
    finally:
        for u in uploads:
            lab.drop_upload(u)
    await task.finish_step(s, text=f"Training started on {target['label'].split(' · ')[0]}")
    await task.emit("lab_run", id=f"lr-{run['id']}", parent=bee, run=run["id"])
    await task.answer("It's training now. The card above updates live, and the training keeps going if you close the app. "
                      "When it's done, press Run it and you can chat with your new model right here.")


async def add_gpu_server(task: Task, parent: str) -> str | None:
    key = await lab.hive_key()
    cmd = f"mkdir -p ~/.ssh && echo '{key}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    reply = await ask(task, "clarify", parent=parent, title="Connect a GPU server",
                      text=f"Any GPU machine you can SSH into works (RunPod, Vast.ai, Lambda, AWS, your PC). First run this on it:\n{cmd}",
                      fields=[{"name": "name", "label": "Name", "value": "GPU server"},
                              {"name": "host", "label": "Host or IP", "required": True},
                              {"name": "port", "label": "SSH port", "input": "number", "value": "22"},
                              {"name": "user", "label": "User", "value": "root"}],
                      actions=[{"id": "connect", "label": "Connect"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "connect":
        await task.answer("Okay. You can also connect GPU servers later in Cloud Bee → Servers.")
        return None
    v = reply["values"]
    s = await task.step("opening", f"Connecting to {v.get('host')}", bee=BEE, parent=parent)
    try:
        t = lab.add_target(v.get("name") or "GPU server", (v.get("host") or "").strip(), int(float(v.get("port") or 22)),
                           (v.get("user") or "root").strip(), "mnx-runs", None)
    except ValueError as exc:
        await task.finish_step(s, status="error", text=str(exc))
        await task.answer("That server address didn't look right. Ask me again to retry.")
        return None
    t = await lab.test_target(t["id"])
    if t["status"] != "ok":
        await task.finish_step(s, status="error", text="Couldn't log in", detail=(t.get("error") or "")[-160:])
        await task.answer("I couldn't log in to that server. Check the key command ran there and the host is right, "
                          "then press Test in Cloud Bee → Servers.")
        return None
    gpus = (t.get("info") or {}).get("gpus") or []
    await task.finish_step(s, text=f"Connected to {t['name']}" + (f" · GPU {gpus[0]}" if gpus else " · no GPU found"))
    return t["id"]


# ---------- status / stop / run ----------
async def status(task: Task) -> None:
    bee = await task.emit("bee_created", bee=BEE, text="checked")
    runs = sorted(lab.runs.values(), key=lambda r: r["created"], reverse=True)
    active = [r for r in runs if r["status"] in ("queued", "preparing", "running")]
    if not runs:
        await task.answer("There's no training yet. Try “train Qwen 2.5 Coder”.")
        return
    for r in (active or runs[:1]):
        await task.emit("lab_run", id=f"lr-{r['id']}", parent=bee, run=r["id"])
    await task.answer(f"{len(active)} training{'s' if len(active) != 1 else ''} running." if active
                      else "Nothing is training right now; here's the latest run.")


async def stop(task: Task) -> None:
    bee = await task.emit("bee_created", bee=BEE, text="checked")
    active = [r for r in lab.runs.values() if r["status"] in ("queued", "preparing", "running")]
    if not active:
        await task.answer("Nothing is training right now.")
        return
    run = max(active, key=lambda r: r["created"])
    last = run["metrics"][-1] if run["metrics"] else {}
    progress = ", ".join(f"{k} {v}" for k, v in last.items() if k in ("epoch", "step", "loss")) or run.get("stage") or run["status"]
    reply = await ask(task, "confirm", parent=bee, title="Stop this training?", service=BEE, default="stop",
                      details=[["Run", run["name"]], ["Progress", progress]],
                      actions=[{"id": "stop", "label": "Stop it", "style": "danger"}, {"id": "keep", "label": "Keep training"}])
    if reply["action"] == "stop":
        await lab.stop_run(run["id"])
        await task.step("done", f"Stopped {run['name']}", bee=BEE, parent=bee)
        await task.emit("lab_run", id=f"lr-{run['id']}", parent=bee, run=run["id"])
        await task.answer("Stopped.")
    else:
        await task.answer("Okay, it keeps training.")


async def run_model(task: Task, intent: Intent) -> None:
    bee = await task.emit("bee_created", bee=BEE, text="is on it")
    models = sorted(lab.models.values(), key=lambda m: m["created"], reverse=True)
    q = intent.model_query.lower()
    pick = [m for m in models if q and all(w in m["name"].lower() for w in q.split())] or models
    pick = [m for m in pick if m.get("servable")]
    if not pick:
        await task.answer("You don't have a model I can run yet. Train one first, e.g. “train Qwen 2.5 Coder”.")
        return
    m = pick[0]
    if m["id"] not in lab.deployments:
        s = await task.step("running", f"Starting {m['name']}", bee=BEE, parent=bee)
        try:
            await lab.deploy(m["id"], lab.gpu_local)
        except (ValueError, RuntimeError) as exc:
            await task.finish_step(s, status="error", text=str(exc))
            await task.answer("I couldn't start it.")
            return
        await task.finish_step(s, text=f"Starting {m['name']} as an API")
    await task.emit("lab_model", id=f"lm-{m['id']}", parent=bee, model=m["id"])
    await task.answer("Your model is starting below. When it says ready, type in the box on the card to talk to it.")


async def stop_renting(task: Task) -> None:
    bee = await task.emit("bee_created", bee=BEE, text="checked")
    if not lab.pods:
        await task.answer("You aren't renting any cloud machines right now.")
        return
    for pid, pod in list(lab.pods.items()):
        s = await task.step("running", f"Stopping {pod['name']} ({pod.get('count')}× {pod.get('gpu_type')})", bee=BEE, parent=bee)
        try:
            await lab.stop_pod(pid)
            await task.finish_step(s, text=f"Stopped {pod['name']}; billing has ended")
        except Exception as exc:
            await task.finish_step(s, status="error", text=f"Couldn't stop {pod['name']}: {exc}"[:200])
    await task.answer("Done. Double-check on runpod.io that nothing is still running.")

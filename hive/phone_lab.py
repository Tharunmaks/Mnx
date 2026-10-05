"""Run or train a model on your phone, layer by layer, from the chat.

"run my stories model on my phone"   → Phone Bee checks the phone, the Hive splits the model into layers,
                                       the phone streams them one at a time and the answer comes back here
"train my stories model on my phone" → same, but every layer is trained on the phone and sent back;
                                       the Hive merges them so Run it uses the new weights

The phone never holds more than one layer: a 500B model is 126 layers of ~7.9 GB (bf16), so the
plan card says honestly what fits and how long a token or a training step takes.
"""

from __future__ import annotations

import asyncio
import re

from . import architect, layers, phone, status
from .cloud_chat import ask
from .core import Task
from .lab import MODELS, lab

PHONE = "Phone"
POLL = 3


def _model_for(query: str) -> dict | None:
    """The trained model the words name; None when they name only a blueprint (or nothing matches and a blueprint does)."""
    from .swarm import _find_model, find_blueprint
    q = (query or "").lower().split()
    trained = [m for m in lab.models.values() if m.get("kind") == "llm" and (MODELS / m["id"] / "files" / "model").is_dir()]
    if q and not any(all(w in m["name"].lower() for w in q) for m in trained):
        bp = find_blueprint(query)
        if bp and all(w in bp["name"].lower() for w in q):
            return None
    return _find_model(query)


def _dims(model: dict) -> tuple[float, int, int]:
    arch = model.get("arch") or {}
    res = model.get("result") or {}
    params = res.get("params") or arch.get("params") or 0
    n_layers, hidden = arch.get("layers"), arch.get("hidden")
    cfg = MODELS / model["id"] / "files" / "model" / "config.json"
    if (not n_layers or not hidden) and cfg.exists():
        import json
        c = json.loads(cfg.read_text())
        n_layers, hidden = c.get("num_hidden_layers", n_layers), c.get("hidden_size", hidden)
    return params, int(n_layers or 1), int(hidden or 1)


async def on_phone(task: Task, mode: str, query: str, prompt: str = "", steps: int = 10, engine: str = "") -> None:
    status.set_busy("phone", "cloud")
    try:
        bee = await task.emit("bee_created", bee=PHONE, text="is checking your phone")
        try:
            p = phone.get()
        except LookupError:
            await task.answer("No phone is connected. Start the drone in Termux on your phone (Phone page shows the command), then ask again.")
            return
        if "stream_start" not in p.capabilities:
            await task.answer(f"{p.info.get('name') or 'Your phone'} is connected but can't run models yet. In Termux run "
                              "`pip install torch transformers safetensors`, restart the drone, and ask again.")
            return
        model = _model_for(query)
        if not model:
            from .swarm import find_blueprint
            bp = find_blueprint(query)
            if bp:
                await task.answer(f"{bp['name']} is only a blueprint so far: it has no weights to stream. Create it first "
                                  f"(“train the {architect.fmt_params(bp['arch']['params'])} blueprint”), then I can run it on your phone layer by layer.")
            else:
                await task.answer("I don't have a trained model by that name. Create one first (“create a 10M model for stories”).")
            return
        params, n_layers, hidden = _dims(model)
        ram = float(p.info.get("ram_gb") or 8)
        plan = layers.plan(params, n_layers, hidden, ram_gb=ram, storage_free_gb=p.info.get("storage_free_gb"),
                           cores=int(p.info.get("cores") or 8))
        s = await task.step("thinking", f"Planning {model['name']} on {p.info.get('name') or 'your phone'} ({ram:.0f} GB): "
                                        f"{n_layers} layers of {plan['layer']}", bee=PHONE, parent=bee)
        await task.finish_step(s)
        details = [["Model", f"{model['name']} · {architect.fmt_params(params)} parameters, {n_layers} layers"],
                   ["Phone", f"{p.info.get('name') or p.id} · {ram:.0f} GB RAM"],
                   ["Memory on the phone", f"{plan['memory_needed']} (one layer at a time)"],
                   ["Weights streamed from", plan["source"]],
                   ["Per generated token", plan["per_token"]] if mode == "run" else ["Per training step", f"{plan['per_step']} ({plan['train_tokens']:,} tokens)"]]
        if engine == "lpu":
            details.append(["Engine", "the virtual LPU on the phone: every streamed layer is compiled and simulated cycle by cycle "
                                      "(slower than the phone's own CPU; the chip's cycle counts come back with the result)"])
        if mode == "run":
            details.append(["Prompt", prompt or "Once upon a time"])
        else:
            details.append(["Steps", f"{steps} on the Hive's demo text (every layer trained on the phone, then merged back)"])
        if not plan["fits"]:
            await task.step("error", f"One layer ({plan['layer']}) is bigger than the phone's memory", bee=PHONE, parent=bee,
                            detail="a phone with more RAM, or a smaller model, is needed")
            reply = await ask(task, "confirm", parent=bee, title="It won't fit, even one layer at a time", service=PHONE,
                              details=details, ask_anyway=True,
                              text=layers.plan_text(plan, model["name"]) + " Try anyway (it may run out of memory), or cancel.",
                              actions=[{"id": "go", "label": "Try anyway"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        else:
            reply = await ask(task, "confirm", parent=bee, title=f"{'Run' if mode == 'run' else 'Train'} {model['name']} on the phone, layer by layer?",
                              service=PHONE, details=details, text=layers.plan_text(plan, model["name"]),
                              actions=[{"id": "go", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "go":
            await task.answer("Okay, nothing started.")
            return
        s = await task.step("opening", f"Splitting {model['name']} into {n_layers} layer files", bee=PHONE, parent=bee)
        model_dir = MODELS / model["id"] / "files" / "model"
        try:
            manifest = await asyncio.to_thread(layers.split, model_dir, MODELS / model["id"] / "layers")
        except ValueError as exc:
            await task.finish_step(s, status="error", text=str(exc))
            await task.answer("This model has no weights I can split. Train it first.")
            return
        await task.finish_step(s, text=f"{manifest['layers']} layers + shared embeddings, {architect.fmt_bytes(manifest['bytes'])} in total")
        args = {"mode": mode, "model": model["id"], "prompt": prompt or "Once upon a time", "max_new": 40, "steps": steps, "engine": engine or "torch"}
        if engine == "lpu" and mode == "train":
            args.update(lr=3e-2, seq_len=64)  # the chip's plain SGD on 64-token windows
        s = await task.step("running", f"Starting on the phone: {'generating' if mode == 'run' else 'training'} layer by layer", bee=PHONE, parent=bee)
        try:
            job = await p.call("stream_start", args, timeout=120)
        except (RuntimeError, asyncio.TimeoutError) as exc:
            await task.finish_step(s, status="error", text=str(exc)[:200])
            await task.answer("The phone couldn't start the runner. Check Termux has torch installed.")
            return
        await task.finish_step(s, text=f"Job {job['job']} is running on the phone")
        prog = await task.step("running", "Waiting for the first layer", bee=PHONE, parent=bee)
        last = None
        while True:
            await asyncio.sleep(POLL)
            try:
                st = await p.call("stream_status", {"job": job["job"]}, timeout=30)
            except (RuntimeError, asyncio.TimeoutError) as exc:
                await task.finish_step(prog, status="error", text=f"Lost the phone: {exc}"[:200])
                await task.answer("The phone stopped answering. The job may still be running there; check the Phone page.")
                return
            m = st.get("metric") or {}
            if m and m != last:
                last = m
                if mode == "run" and m.get("cycles"):
                    text = (f"Token {m.get('token')} · {m.get('cycles'):,} chip cycles ({m.get('chip_us')} µs on the LPU) · simulated in {m.get('seconds')} s · "
                            f"{m.get('fetched_mb')} MB fetched so far")
                elif mode == "run":
                    text = f"Token {m.get('token')} · {m.get('layers')} layers streamed in {m.get('seconds')} s · {m.get('fetched_mb')} MB fetched so far"
                elif m.get("cycles"):
                    text = f"Step {m.get('step')}/{steps} · loss {m.get('loss')} · {m.get('cycles'):,} chip cycles ({m.get('chip_us')} µs on the LPU) · simulated in {m.get('seconds')} s"
                else:
                    text = f"Step {m.get('step')}/{steps} · loss {m.get('loss')} · {m.get('seconds')} s per step · {m.get('sent_mb')} MB of trained layers sent back"
                await task.emit("running", text, bee=PHONE, id=prog)
            if not st.get("running"):
                break
        res = st.get("result")
        if st.get("exit_code") not in (0, None) or not res:
            await task.finish_step(prog, status="error", text="The runner stopped early", detail=(st.get("tail") or "")[-300:])
            await task.answer("Something went wrong on the phone:\n" + (st.get("tail") or "no output")[-600:])
            return
        if mode == "run":
            await task.finish_step(prog, text=f"Generated {len(res.get('text', '').split())} words in {res.get('seconds')} s "
                                              f"with {res.get('fetched_mb')} MB streamed")
            rows = [["Prompt", res.get("prompt", "")], ["Layers streamed per token", str(res.get("layers"))],
                    ["Time", f"{res.get('seconds')} s"], ["Data fetched", f"{res.get('fetched_mb')} MB"]]
            if res.get("cycles_per_token"):
                rows += [["On the virtual LPU", f"{res['cycles_per_token']:,} cycles per token → {res.get('chip_tokens_per_second', 0):,.0f} tokens/s on the chip"]]
            await task.emit("result", parent=bee, service=PHONE, title=f"{model['name']} ran on your phone, layer by layer"
                            + (" on the virtual LPU" if res.get("cycles_per_token") else ""), badge="Done", details=rows, text=res.get("text", ""))
            await task.answer(res.get("text") or "(the model wrote nothing)")
        else:
            await task.finish_step(prog, text=f"Trained {steps} steps on the phone · final loss {res.get('loss'):.3f} · "
                                              f"{res.get('sent_mb')} MB of layers sent back and merged")
            await task.emit("lab_model", id=f"lm-{model['id']}", parent=bee, model=model["id"])
            await task.answer(f"{model['name']} learned on your phone one layer at a time: {steps} steps, loss {res.get('loss'):.3f}. "
                              "The Hive merged the trained layers back, so Run it now uses them.")
    finally:
        status.clear("phone", "cloud")

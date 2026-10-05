"""Run or train a model on a virtual LPU from the chat: a cycle-accurate Groq-style chip, or a mesh of them, simulated in software.

"run my stories model on the lpu"                 → one virtual chip
"run my stories model on 8 lpu chips"             → a mesh: layers spread over chips, stages pipelined over the links
"train my stories model on the lpu for 20 steps"  → every step is one compiled SGD step on the chips; the trained
                                                    weights are merged back into the Hive's model, like the phone's

The LPU Bee compiles the model (exact cycles, SRAM addresses, resident or streamed weights), shows the plan, then
simulates it with the real weights and reports the chip's numbers next to the answer.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

from . import architect, layers, lpu, status
from .cloud_chat import ask
from .core import Task
from .lab import MODELS, RECIPES, lab
from .phone_lab import _dims, _model_for

LPU = "LPU"
POLL = 1.0
MAX_SIM_CHIPS = 16  # plans go to any size; the software simulation stops here (each chip is a 230 MB SRAM in RAM)
TRAIN_SEQ = 64      # tokens per training step on the chip
MAX_STEPS = 1000
LR = 3e-2  # plain SGD (no optimizer state on the chip): tiny models need a large step to move
TRAIN_SPEC = lpu.ChipSpec(word_bytes=4)  # training keeps 32-bit master weights in SRAM, else small SGD steps round away


def _layout(p: dict) -> str:
    s = f"{p['chips']} chip{'s' if p['chips'] != 1 else ''}"
    if p["tp"] > 1:
        s += f": every layer spread over {p['tp']}" + (f", {p['stages']} pipeline stages" if p["stages"] > 1 else "")
    elif p["stages"] > 1:
        s += f" as a {p['stages']}-stage pipeline"
    return s


def _util(st: dict) -> str:
    return ", ".join(f"{u} {v:.0f}%" for u, v in st["utilisation"].items() if not u.startswith("link"))


async def on_lpu(task: Task, query: str, prompt: str = "", chips: int = 1, max_new: int = 40, mode: str = "run", steps: int = 10) -> None:
    status.set_busy("gpu", "cloud")
    try:
        bee = await task.emit("bee_created", bee=LPU, text="is compiling for the virtual chip")
        train = mode != "run"
        model = _model_for(query)
        if not model:
            from .swarm import find_blueprint
            bp = find_blueprint(query)
            if bp:
                await task.answer(f"{bp['name']} is only a blueprint so far: it has no weights to {'train' if train else 'run'}. Create it first "
                                  f"(“train the {architect.fmt_params(bp['arch']['params'])} blueprint”), then I can compile it for the LPU.")
            else:
                await task.answer("I don't have a trained model by that name. Create one first (“create a 10M model for stories”).")
            return
        model_dir = MODELS / model["id"] / "files" / "model"
        if not (model_dir / "config.json").exists() or not (model_dir / "tokenizer.json").exists():
            await task.answer("This model has no weights or tokenizer I can compile. Train it first.")
            return
        cfg = json.loads((model_dir / "config.json").read_text())
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        params, n_layers, _ = _dims(model)
        chips = max(1, int(chips))
        steps = max(1, min(int(steps), MAX_STEPS))
        if train:
            seq = TRAIN_SEQ
            text_ids = tok.encode((RECIPES / "llm-pretrain" / "sample.txt").read_text(), add_special_tokens=False).ids
            if len(text_ids) < seq + 1:
                await task.answer("The Hive's demo text is too short to make a training window.")
                return
            ids: list = []
        else:
            prompt = prompt or "Once upon a time"
            ids = tok.encode(prompt, add_special_tokens=False).ids or [int(cfg.get("bos_token_id") or 0)]
            seq = len(ids) + max_new
        s = await task.step("thinking", f"Compiling {model['name']} for {chips} virtual chip{'s' if chips != 1 else ''}"
                                        + (f": one training step of {seq} tokens" if train else ""), bee=LPU, parent=bee)
        spec = TRAIN_SPEC if train else lpu.ChipSpec()
        plan = await asyncio.to_thread(lpu.plan, cfg, params, seq, chips, spec, 1.5e9, 3e10, 3e-5, train, LR)
        unit = "step" if train else "token"
        await task.finish_step(s, text=(f"{plan['cycles_per_token']:,} cycles per {unit} on {_layout(plan)}" if plan["fits"]
                                        else "it doesn't fit: " + plan["shard"] + " per chip is more than the SRAM"))
        details = [["Model", f"{model['name']} · {architect.fmt_params(params)} parameters, {n_layers} layers"],
                   ["Chip", f"{spec.name} · {plan['sram_per_chip']} SRAM ({plan['word_bits']}-bit words"
                            + (": master weights stay exact, so SRAM holds half as much" if train else "") + f"), {plan['clock_mhz']:.0f} MHz, "
                            f"{spec.mxm_units} matrix + {spec.vxm_units} vector units, no caches, no DRAM"],
                   ["Chips", _layout(plan)],
                   ["Weights", f"{plan['weights']} · {'resident in SRAM, loaded once' if plan['resident'] else 'streamed from the host every ' + unit + ' (' + plan['source'] + ')'}"]]
        if plan["fits"]:
            details += [[f"Per {unit} on the chip", f"{plan['cycles_per_token']:,} cycles = {plan['per_token']} → {plan['chip_tokens_per_second']:,.0f} tokens/s"
                                                   + ("" if plan["exact"] else " (one exact layer, multiplied out)")],
                        ["Simulating here", f"{'about ' if plan['sim_seconds_per_token'] >= 1 else ''}{plan['sim_per_token']} per {unit} "
                                            "(every multiply really runs, plus the tick loop)"]]
        if train:
            details += [["Steps", f"{steps} × {seq} tokens of the Hive's demo text, plain SGD (lr {LR:g}), forward + backward + update on the chips"],
                        ["Afterwards", "the trained weights are merged back into the model, so Run it uses them"]]
        else:
            details += [["Prompt", prompt], ["Tokens to generate", str(max_new)]]
        if not plan["fits"]:
            await task.step("error", f"One chip's share of a layer ({plan['shard']}) is bigger than its SRAM", bee=LPU, parent=bee,
                            detail="an LPU has no other memory; a layer must be spread over more chips")
            if plan["min_chips"] and plan["min_chips"] <= MAX_SIM_CHIPS:
                reply = await ask(task, "confirm", parent=bee, title="It doesn't fit on that many chips", service=LPU, details=details, ask_anyway=True,
                                  text=lpu.plan_text(plan, model["name"]),
                                  actions=[{"id": "more", "label": f"Use {plan['min_chips']} chips"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
                if reply["action"] != "more":
                    await task.answer("Okay, nothing started.")
                    return
                return await on_lpu(task, query, prompt, plan["min_chips"], max_new, mode, steps)
            await task.answer(lpu.plan_text(plan, model["name"]) + (f" That is more chips than I simulate here ({MAX_SIM_CHIPS})." if plan["min_chips"] else ""))
            return
        if plan["chips"] > MAX_SIM_CHIPS:
            reply = await ask(task, "confirm", parent=bee, title=f"Plan for {plan['chips']} chips; simulate on {MAX_SIM_CHIPS}?", service=LPU,
                              details=details, ask_anyway=True,
                              text=lpu.plan_text(plan, model["name"]) + f" I only simulate up to {MAX_SIM_CHIPS} chips here (each is a {plan['sram_per_chip']} SRAM in memory).",
                              actions=[{"id": "go", "label": f"Simulate on {MAX_SIM_CHIPS} chips"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
            if reply["action"] != "go":
                await task.answer("Okay, nothing started. The plan above is still exact for that many chips.")
                return
            return await on_lpu(task, query, prompt, MAX_SIM_CHIPS, max_new, mode, steps)
        title = (f"Train {model['name']} on {_layout(plan)} for {steps} steps, cycle by cycle?" if train
                 else f"Run {model['name']} on {_layout(plan)}, cycle by cycle?")
        reply = await ask(task, "confirm", parent=bee, title=title, service=LPU, details=details, text=lpu.plan_text(plan, model["name"]),
                          actions=[{"id": "go", "label": "Start"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "go":
            await task.answer("Okay, nothing started.")
            return
        s = await task.step("opening", f"Splitting {model['name']} into {n_layers} layer files", bee=LPU, parent=bee)
        try:
            manifest = await asyncio.to_thread(layers.split, model_dir, MODELS / model["id"] / "layers")
        except ValueError as exc:
            await task.finish_step(s, status="error", text=str(exc))
            await task.answer("This model has no weights I can split. Train it first.")
            return
        await task.finish_step(s, text=f"{manifest['layers']} layers + shared embeddings, {architect.fmt_bytes(manifest['bytes'])}")
        layers_dir = MODELS / model["id"] / "layers"

        def load_layer(i: int) -> dict:
            return lpu.strip_layer_prefix(lpu.read_safetensors(str(layers_dir / f"layer_{i:04d}.safetensors")), i)

        def shared() -> dict:
            return lpu.read_safetensors(str(layers_dir / "shared.safetensors"))

        prog = await task.step("running", "Compiling the schedule and loading the chips", bee=LPU, parent=bee)
        state: dict = {"n": 0, "stats": None, "done": False, "error": None, "out": [], "model": None, "loss": None, "written": 0}

        def run() -> None:
            try:
                m = lpu.Model(cfg, load_layer, shared, seq=seq, spec=spec, chips=chips, train=train, lr=LR)
                state["model"] = m
                t0 = time.time()
                if train:
                    for k in range(steps):
                        start = (k * seq) % (len(text_ids) - seq - 1)
                        state["loss"] = m.train_step(text_ids[start:start + seq], text_ids[start + 1:start + seq + 1])
                        state["n"], state["stats"] = k + 1, m.step_stats()
                    m.pull_weights()
                    for i in range(n_layers):  # the trained weights go back into the layer files, in the model's own dtype
                        orig = lpu.read_safetensors(str(layers_dir / f"layer_{i:04d}.safetensors"))
                        prefix = next(k for k in orig if ".layers." in k).split(".layers.")[0] + f".layers.{i}."
                        state["written"] += lpu.write_safetensors(str(layers_dir / f"layer_{i:04d}.safetensors"),
                                                                  {prefix + k: v.reshape(orig[prefix + k].shape) for k, v in m.weights[i].items()}, manifest["dtype"])
                    orig = lpu.read_safetensors(str(layers_dir / "shared.safetensors"))
                    state["written"] += lpu.write_safetensors(str(layers_dir / "shared.safetensors"),
                                                              {k: m.shared().get(k, v).reshape(v.shape) for k, v in orig.items()}, manifest["dtype"])
                    state["merged"] = lab.merge_layers(model["id"], steps, state["loss"])
                else:
                    def on_token(n, _tok, st):
                        state["n"], state["stats"] = n, st

                    state["out"] = m.generate(ids, max_new, temperature=0.7, eos=cfg.get("eos_token_id"), on_token=on_token)
                    state["stats"] = m.token_stats()
                state["seconds"] = time.time() - t0
            except Exception as exc:  # MemoryError / ScheduleError land in the chat, not in the log
                state["error"] = exc
            finally:
                state["done"] = True

        threading.Thread(target=run, daemon=True).start()
        last = 0
        total = steps if train else max_new
        while not state["done"]:
            await asyncio.sleep(POLL)
            if state["n"] != last and state["stats"]:
                last, st = state["n"], state["stats"]
                if train:
                    text = (f"Step {last}/{total} · loss {state['loss']:.3f} · {st['cycles_per_step']:,} cycles per step · "
                            f"{st['sim_seconds_per_step']:.2f} s to simulate each")
                else:
                    text = (f"Token {last}/{total} · {st['cycles_per_token']:,} cycles each · {st['sim_seconds_per_token']:.2f} s to simulate · "
                            f"{st['instructions_per_token']} instructions on {st['chips']} chip{'s' if st['chips'] != 1 else ''}")
                await task.emit("running", text, bee=LPU, id=prog)
        if state["error"] is not None:
            exc = state["error"]
            await task.finish_step(prog, status="error", text=f"{type(exc).__name__}: {exc}"[:300])
            await task.answer(("The compiler's plan and the chip disagreed, which is a bug in the compiler: " if isinstance(exc, lpu.ScheduleError)
                               else "The model doesn't fit the chips after all: " if isinstance(exc, MemoryError) else "The simulation failed: ") + str(exc)[:400])
            return
        st, m = state["stats"], state["model"]
        if train:
            losses = st["losses"]
            await task.finish_step(prog, text=f"{steps} steps in {state['seconds']:.1f} s of simulation · loss {losses[0]:.3f} → {losses[-1]:.3f} · "
                                              f"{st['cycles_per_step']:,} cycles per step = {lpu._fmt_time(st['chip_seconds_per_step'])} on the chip")
            curve = " ".join(f"{x:.2f}" for x in (losses if len(losses) <= 12 else losses[:6] + losses[-6:]))
            await task.emit("lab_model", id=f"lm-{model['id']}", parent=bee, model=model["id"])
            await task.emit("result", parent=bee, service=LPU, title=f"{model['name']} trained on {_layout(plan)}, cycle by cycle", badge="Done",
                            details=[["Steps", f"{steps} × {seq} tokens · loss {losses[0]:.3f} → {losses[-1]:.3f}" + (" (first and last six)" if len(losses) > 12 else "")],
                                     ["Loss", curve],
                                     ["Cycles per step", f"{st['cycles_per_step']:,} ({st['instructions_per_step']} instructions: forward, backward, update)"],
                                     ["On the chip", f"{lpu._fmt_time(st['chip_seconds_per_step'])} per step → {st['tokens_per_second']:,.0f} training tokens/s at {plan['clock_mhz']:.0f} MHz"],
                                     ["Weights", "resident in SRAM, updated in place" if st["resident"] else f"streamed: {st['host_mb_per_step']:.1f} MB over the host link per step"],
                                     ["SRAM used per chip", f"{architect.fmt_bytes(m.plan.sram_peak_bytes)} of {plan['sram_per_chip']} (peak, by the compiler's static allocation)"],
                                     ["Chip links", f"{st['link_mb_per_step'] * 1000:.0f} KB per step" if st["link_mb_per_step"] else "none (one chip)"],
                                     ["Unit utilisation", _util(st)],
                                     ["Simulation", f"{st['sim_seconds_per_step']:.2f} s per step here"],
                                     ["Merged", f"{architect.fmt_bytes(state['written'])} of trained layers written back; the model now has "
                                                f"{state['merged']['trained_steps']} layer-streamed steps"]],
                            text=f"Loss went from {losses[0]:.3f} to {losses[-1]:.3f}.")
            await task.answer(f"{model['name']} learned on the virtual LPU, one compiled step at a time: {steps} steps, loss {losses[0]:.3f} → {losses[-1]:.3f}. "
                              f"Each step took {st['cycles_per_step']:,} cycles ({lpu._fmt_time(st['chip_seconds_per_step'])}) on the chip. "
                              "The trained weights are merged back, so Run it now uses them.")
        else:
            text = tok.decode(state["out"])
            await task.finish_step(prog, text=f"{len(state['out'])} tokens in {state['seconds']:.1f} s of simulation · "
                                              f"{st['cycles_per_token']:,} cycles per token = {lpu._fmt_time(st['chip_seconds_per_token'])} on the chip")
            await task.emit("result", parent=bee, service=LPU, title=f"{model['name']} ran on {_layout(plan)}, cycle by cycle", badge="Done",
                            details=[["Prompt", prompt], ["Cycles per token", f"{st['cycles_per_token']:,} ({st['instructions_per_token']} instructions)"],
                                     ["On the chip", f"{lpu._fmt_time(st['chip_seconds_per_token'])} per token → {st['chip_tokens_per_second']:,.0f} tokens/s at {plan['clock_mhz']:.0f} MHz"],
                                     ["Weights", "resident in SRAM" if st["resident"] else f"streamed: {st['host_mb_per_token']:.1f} MB over the host link per token"],
                                     ["SRAM used per chip", f"{architect.fmt_bytes(m.plan.sram_peak_bytes)} of {plan['sram_per_chip']} (peak, by the compiler's static allocation)"],
                                     ["Chip links", f"{st['link_mb_per_token'] * 1000:.0f} KB per token" if st["link_mb_per_token"] else "none (one chip)"],
                                     ["Unit utilisation", _util(st)],
                                     ["Simulation", f"{st['sim_seconds_per_token']:.2f} s per token here ({st['ticks_per_token']:.0f} tick-loop events per token)"]],
                            text=text)
            await task.answer(text or "(the model wrote nothing)")
    finally:
        status.clear("gpu", "cloud")

"""The Queen: reads each message and sends it to what can do it: a call, a new Bee, your plan, the GPU, Expert Craft,
a website to build, an app or site to use, your Bees (Peace and Rush), training, and for everything else the model,
which answers with web research when the question needs today's facts (hive/respond.py).
"""

from __future__ import annotations

from . import apps, bots, brain, calls, cloud_chat, expert, games, gpu_pool, models3d, modes, plans, respond, sites
from .gpu_chat import gpu_chat
from .bees import registry
from .core import Task


async def handle(task: Task) -> None:
    await _handle(task)
    if not task.answered:  # a flow that stopped at a card you cancelled: say so instead of going quiet
        await task.answer("Okay, I stopped there; nothing was started.")


async def _handle(task: Task) -> None:
    peace = task.bee and registry.mine(task.bee, task.user)
    if peace:
        task.speaker = peace["name"]  # Peace mode: this Bee answers in its own chat
    received = await task.step("planning", "Reading your message", bee="Queen")
    await task.finish_step(received)

    # "call me", "call me in 10 minutes and tell me …": a voice call with the Fish Audio voice
    if await calls.handle_chat(task):
        return
    if bots.parse(task.text):  # "make a bee called … that …"
        await bots.create_flow(task, bots.parse(task.text))
        return
    if plans.parse(task.text):  # "subscribe me to plus"
        await plans.handle(task, plans.parse(task.text))
        return
    if gpu_pool.parse(task.text):  # "run train.py on the gpu", "gpu status"
        await gpu_chat(task, gpu_pool.parse(task.text))
        return
    if expert.parse(task.text):  # Expert Craft training on a Bee's computer; "set up expert stream" (the owner)
        await expert.handle(task, expert.parse(task.text))
        return
    if models3d.parse(task.text):  # "make a 3d model of a car": a Bee designs it, builds the .glb, checks it; Mnx Studio edits it
        await models3d.handle(task, models3d.parse(task.text))
        return
    if games.parse(task.text):  # "create an open world game in 3d": a Bee writes it, play-tests it, fixes it, gives a private link
        await games.handle(task, games.parse(task.text))
        return
    if sites.parse(task.text):  # "build a website for my bakery": a Bee writes it, Chroma runs it, the files sit in its Vortex OS
        await sites.handle(task, sites.parse(task.text))
        return
    if apps.parse(task.text):  # "add KitKat in blinkit": a Bee uses the website (or the owner's phone) for you
        await apps.handle(task, apps.parse(task.text))
        return
    if task.bee and await modes.peace(task):
        return
    if task.mode == "rush" and await modes.rush(task):
        return

    # Requests a Bee understands on its own (no AI needed), e.g. "train Qwen 2.5 Coder".
    if await cloud_chat.handle(task):
        return

    if task.bee:
        await modes.peace_fallback(task)
        return
    if task.mode == "rush" and await modes.rush_plan(task):
        return

    if not brain.is_connected():
        await task.step(
            "error",
            "No model is running yet",
            bee="Queen",
            detail="the Hive's owner sets up the models",
        )
        await task.answer(
            "I got your message, but no model is running on this Hive yet, so I can't answer that. "
            "What already works without one: “make a bee called … that …”, “list my bees”, “what plan am I on”, "
            "“call me”" + (", and in Rush: “everyone run …”, “roll call”." if task.mode == "rush" else ".")
        )
        return

    # Everything else: the model answers, streaming, after looking things up on the web when they need today's facts.
    await respond.respond(task, brain.system_prompt(task.effort), history=getattr(task, "history", None))

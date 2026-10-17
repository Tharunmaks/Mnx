"""The Queen: receives a task, plans it, hands steps to Bees, merges the result.

Planning needs the Mnx brain, which isn't connected yet, so for now the Queen
reports that honestly instead of inventing an answer.
"""

from __future__ import annotations

from . import apps, bots, brain, calls, cloud_chat, gpu_pool, modes, plans
from .gpu_chat import gpu_chat
from .bees import registry
from .core import Task


async def handle(task: Task) -> None:
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

    # When the brain is live, this is where the Queen plans and runs Bees:
    #   plan = await brain.ask(planning_prompt(task.text, bees))
    #   for step in plan: await bee.run(task, step)
    #   await task.answer(merged_answer)
    await task.answer(await brain.ask(task.text, task.effort, getattr(task, "history", None)))

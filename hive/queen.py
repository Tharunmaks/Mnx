"""The Queen: receives a task, plans it, hands steps to Bees, merges the result.

Planning needs the Mnx brain, which isn't connected yet, so for now the Queen
reports that honestly instead of inventing an answer.
"""

from __future__ import annotations

from . import brain, cloud_chat
from .core import Task


async def handle(task: Task) -> None:
    received = await task.step("planning", "Reading your message", bee="Queen")
    await task.finish_step(received)

    # Requests a Bee understands on its own (no AI needed), e.g. "train Qwen 2.5 Coder".
    if await cloud_chat.handle(task):
        return

    if not brain.is_connected():
        await task.step(
            "error",
            "Mnx brain isn't connected yet",
            bee="Queen",
            detail="set MNX_BRAIN_URL and implement hive/brain.py",
        )
        await task.answer(
            "I got your message, but my brain (the Mnx model) isn't connected to the Hive yet, "
            "so I can't plan or answer that yet. What already works without it: training and running models, "
            "e.g. “train Qwen 2.5 Coder”, “how is my training going?”, “stop training”, “run my model”."
        )
        return

    # When the brain is live, this is where the Queen plans and runs Bees:
    #   plan = await brain.ask(planning_prompt(task.text, bees))
    #   for step in plan: await bee.run(task, step)
    #   await task.answer(merged_answer)
    await task.answer(await brain.ask(task.text, task.effort))

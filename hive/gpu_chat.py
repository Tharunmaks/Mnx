"""The GPU in the chat: "run train.py on the gpu" (from a Bee's own files) and "gpu status"."""

from __future__ import annotations

from .bees import registry
from .core import Task
from .gpu_pool import GpuError, pool


async def read_bee_file(b: dict, rel: str) -> bytes:
    from .cells import cells
    cell = cells.get(b["id"], b["name"], b["computer"])
    if cell.mode == "remote":
        return await cell.rread(rel, 25 * 1024 * 1024)
    p = cell.safe_path(rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    return p.read_bytes()


async def gpu_chat(task: Task, intent: dict) -> None:
    b = registry.get(task.bee or "") or registry.get("cloud")
    if intent["kind"] == "status":
        v = pool.view()
        mine = [j for j in v["jobs"] if not task.bee or j["bee"] == task.bee][:8]
        await task.emit("result", bee="GPU", service="GPU", title=f"GPU pool · {v['hours_week']:.1f} of {v['weekly_limit']:g} free hours used this week",
                        badge="GPU", details=[[j["id"], f"{j['status']} · {j['command'][:60]}" + (f" · {j['error']}" if j.get("error") else "")]
                                              for j in mine] or [["Jobs", "none yet"]])
        src = {"local": "this machine's GPU", "kaggle": "a free Kaggle GPU (16 GB VRAM)"}.get(v["source"] or "", "not set up yet (Free cloud page)")
        await task.answer(f"The Hive's GPU: {src}.")
        return
    files = {}
    for rel in intent.get("files") or []:
        try:
            files[rel] = await read_bee_file(b, rel)
        except (FileNotFoundError, ValueError, OSError):
            await task.answer(f"{rel} isn't in {b['name']}'s computer. Put it in its work folder first (Cell → Files).")
            return
    s = await task.step("running", f"Sending {intent['command'][:60]} to the GPU", bee="GPU",
                        detail=f"{len(files)} file{'s' if len(files) != 1 else ''} from {b['name']}'s computer")
    try:
        job = await pool.submit(b["id"], intent["command"], files)
    except GpuError as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(str(exc))
        return
    await task.finish_step(s, text=f"job {job['id']} running on {job['gpu']}")
    await task.answer(f"It's running on {job['gpu']} (job {job['id']}). When it finishes, its output files land in "
                      f"{b['name']}'s computer under gpu-out/{job['id']}/. Say “gpu status” to check.")

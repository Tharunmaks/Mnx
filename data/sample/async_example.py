import asyncio


async def fetch(delay, value):
    """Pretend to fetch value after delay seconds."""
    await asyncio.sleep(delay)
    return value


async def fetch_all(values):
    """Fetch many values concurrently."""
    tasks = [fetch(0.01, v) for v in values]
    return await asyncio.gather(*tasks)


def run():
    return asyncio.run(fetch_all([1, 2, 3]))

"""A decorator that retries a function up to n times when it raises an exception.

tags: retry decorator exceptions attempts
"""

import functools
import time


def retry(times: int = 3, delay: float = 0.0):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorate


# tests
tries = []
@retry(times=3)
def flaky():
    tries.append(1)
    if len(tries) < 3:
        raise ValueError
    return "ok"
assert flaky() == "ok" and len(tries) == 3

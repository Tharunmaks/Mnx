"""A decorator that prints how long a function takes to run.

tags: timer decorator timing measure time elapsed
"""

import functools
import time


def timer(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"{fn.__name__} took {time.perf_counter() - start:.4f}s")

    return wrapper


# tests
@timer
def add(a, b):
    return a + b
assert add(1, 2) == 3

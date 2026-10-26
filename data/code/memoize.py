"""A decorator that caches a function's results (memoization).

tags: memoize cache decorator memoization
"""

import functools


def memoize(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]

    return wrapper


# tests
calls = []
@memoize
def sq(x):
    calls.append(x)
    return x * x
assert sq(3) == 9 and sq(3) == 9 and calls == [3]

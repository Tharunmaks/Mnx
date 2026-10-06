def count_up(start=0, step=1):
    """Yield numbers forever starting at start."""
    n = start
    while True:
        yield n
        n += step


def take(iterable, n):
    """Return the first n items of an iterable as a list."""
    result = []
    for item in iterable:
        if len(result) >= n:
            break
        result.append(item)
    return result


def evens(limit):
    """Yield even numbers below limit."""
    for i in range(0, limit, 2):
        yield i

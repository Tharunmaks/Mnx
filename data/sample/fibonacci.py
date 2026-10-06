def fib(n):
    """Return the n-th Fibonacci number iteratively."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_recursive(n):
    """Return the n-th Fibonacci number recursively."""
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_sequence(count):
    """Return a list with the first count Fibonacci numbers."""
    result = []
    for i in range(count):
        result.append(fib(i))
    return result

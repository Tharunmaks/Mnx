"""Return the n-th Fibonacci number (fib(0) = 0, fib(1) = 1), iteratively in O(n).

tags: fibonacci number sequence nth
"""

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# tests
assert [fibonacci(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
assert fibonacci(90) == 2880067194370816120

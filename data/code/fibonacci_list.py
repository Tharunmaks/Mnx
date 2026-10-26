"""Return a list of the first n Fibonacci numbers.

tags: fibonacci sequence list first n series
"""

def fibonacci_list(n: int) -> list:
    out, a, b = [], 0, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


# tests
assert fibonacci_list(7) == [0, 1, 1, 2, 3, 5, 8]
assert fibonacci_list(0) == []

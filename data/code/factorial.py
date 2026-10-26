"""Return n! (the factorial of n) for a non-negative integer.

tags: factorial product n
"""

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# tests
assert factorial(0) == 1
assert factorial(5) == 120

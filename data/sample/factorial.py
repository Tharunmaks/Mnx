def factorial(n):
    """Return n! for a non-negative integer n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def factorial_recursive(n):
    """Return n! using recursion."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

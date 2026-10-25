"""Check whether an integer is prime using trial division up to its square root.

tags: prime number check primality
"""

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# tests
assert [n for n in range(20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
assert not is_prime(1) and is_prime(7919)

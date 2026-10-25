"""Return all primes up to and including n using the Sieve of Eratosthenes.

tags: sieve eratosthenes primes list up to n
"""

def primes_up_to(n: int) -> list:
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(range(i * i, n + 1, i))
    return [i for i, p in enumerate(sieve) if p]


# tests
assert primes_up_to(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
assert primes_up_to(1) == []

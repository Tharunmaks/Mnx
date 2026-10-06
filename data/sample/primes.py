def is_prime(n):
    """Return True if n is a prime number."""
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


def primes_up_to(limit):
    """Return a list of all primes less than or equal to limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)]


def sieve(limit):
    """Return primes up to limit using the sieve of Eratosthenes."""
    flags = [True] * (limit + 1)
    flags[0] = flags[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            for j in range(i * i, limit + 1, i):
                flags[j] = False
    return [i for i, ok in enumerate(flags) if ok]

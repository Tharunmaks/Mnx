"""Return the prime factors of a number, smallest first, with repeats.

tags: math, prime, factors, factorize
"""

def prime_factors(n):
    out = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out

# tests
assert prime_factors(12) == [2, 2, 3]
assert prime_factors(7) == [7]

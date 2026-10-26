"""Check whether a number is perfect (equal to the sum of its proper divisors).

tags: perfect number divisors sum
"""

def perfect_number(n: int) -> bool:
    if n < 2:
        return False
    total, i = 1, 2
    while i * i <= n:
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
        i += 1
    return total == n


# tests
assert perfect_number(28) and not perfect_number(12)

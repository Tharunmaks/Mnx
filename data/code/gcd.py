"""Greatest common divisor of two integers with Euclid's algorithm.

tags: gcd greatest common divisor hcf euclid
"""

def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


# tests
assert gcd(48, 18) == 6
assert gcd(0, 5) == 5

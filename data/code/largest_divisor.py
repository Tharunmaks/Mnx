"""Return the largest number smaller than n that divides n evenly.

tags: math, divisor, factor, largest, divide
"""

def largest_divisor(n):
    for d in range(n - 1, 0, -1):
        if n % d == 0:
            return d
    return 1

# tests
assert largest_divisor(15) == 5
assert largest_divisor(10) == 5

"""Add up the digits of a non-negative integer.

tags: math, digits, sum, add, number
"""

def digit_sum(n):
    return sum(int(c) for c in str(abs(n)))

# tests
assert digit_sum(123) == 6
assert digit_sum(0) == 0

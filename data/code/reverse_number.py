"""Reverse the digits of an integer.

tags: math, reverse, digits, number
"""

def reverse_number(n):
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])

# tests
assert reverse_number(123) == 321
assert reverse_number(-45) == -54

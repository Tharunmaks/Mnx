"""Return the decimal part of a positive number.

tags: math, decimal, truncate, fractional, part
"""

def truncate_decimal(number):
    return number % 1.0

# tests
assert truncate_decimal(3.5) == 0.5
assert abs(truncate_decimal(1.25) - 0.25) < 1e-9

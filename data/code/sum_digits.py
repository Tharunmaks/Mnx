"""Sum the decimal digits of an integer.

tags: sum digits number
"""

def sum_digits(n: int) -> int:
    return sum(int(d) for d in str(abs(n)))


# tests
assert sum_digits(12345) == 15
assert sum_digits(-90) == 9

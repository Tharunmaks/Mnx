"""Count the one bits in the binary form of a number.

tags: math, binary, bits, count, ones, population
"""

def count_bits(n):
    return bin(abs(n)).count("1")

# tests
assert count_bits(7) == 3
assert count_bits(0) == 0

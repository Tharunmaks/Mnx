"""Return the longest run of zeros between ones in a number binary form.

tags: math, binary, gap, zeros, longest, run
"""

def binary_gap(n):
    bits = bin(n)[2:].strip("0")
    return max((len(g) for g in bits.split("1")), default=0)

# tests
assert binary_gap(9) == 2
assert binary_gap(15) == 0

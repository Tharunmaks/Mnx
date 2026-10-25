"""Convert a binary string to an integer.

tags: binary to decimal convert base 2 parse
"""

def binary_to_decimal(bits: str) -> int:
    value = 0
    for b in bits:
        value = value * 2 + (1 if b == "1" else 0)
    return value


# tests
assert binary_to_decimal("1010") == 10

"""Convert a non-negative integer to its binary string.

tags: decimal to binary convert base 2
"""

def decimal_to_binary(n: int) -> str:
    if n == 0:
        return "0"
    bits = []
    while n:
        bits.append(str(n % 2))
        n //= 2
    return "".join(reversed(bits))


# tests
assert decimal_to_binary(10) == "1010"
assert decimal_to_binary(0) == "0"

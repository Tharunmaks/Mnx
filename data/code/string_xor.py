"""Exclusive-or two strings of bits, character by character.

tags: string, xor, binary, bits
"""

def string_xor(a, b):
    return "".join("0" if x == y else "1" for x, y in zip(a, b))

# tests
assert string_xor('010', '110') == '100'
assert string_xor('111', '111') == '000'

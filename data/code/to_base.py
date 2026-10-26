"""Convert a non-negative integer to a string in another base, up to base 9.

tags: math, base, convert, number, string, radix
"""

def to_base(x, base):
    if x == 0:
        return "0"
    out = ""
    while x:
        out = str(x % base) + out
        x //= base
    return out

# tests
assert to_base(8, 3) == '22'
assert to_base(0, 2) == '0'

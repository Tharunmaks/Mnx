def to_binary(n):
    """Return the binary representation of a non-negative integer."""
    if n == 0:
        return "0"
    bits = []
    while n > 0:
        bits.append(str(n % 2))
        n //= 2
    return "".join(reversed(bits))


def from_binary(s):
    """Parse a binary string into an integer."""
    value = 0
    for ch in s:
        value = value * 2 + (1 if ch == "1" else 0)
    return value


def count_bits(n):
    """Return the number of set bits in n."""
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

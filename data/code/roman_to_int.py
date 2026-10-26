"""Convert a Roman numeral to an integer.

tags: roman numeral to integer convert
"""

def roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        v = values[c]
        if i + 1 < len(s) and values[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


# tests
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("IX") == 9

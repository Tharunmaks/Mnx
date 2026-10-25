"""Return the value of a single Roman numeral letter.

tags: roman, numeral, letter, value, convert
"""

def roman_numeral_value(c):
    return {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}[c.upper()]

# tests
assert roman_numeral_value('X') == 10
assert roman_numeral_value('i') == 1

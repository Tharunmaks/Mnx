"""Convert an integer (1 to 3999) to a Roman numeral.

tags: integer to roman numeral convert
"""

def int_to_roman(n: int) -> str:
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


# tests
assert int_to_roman(1994) == "MCMXCIV"
assert int_to_roman(4) == "IV"

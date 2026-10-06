ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n):
    """Convert a positive integer to a Roman numeral."""
    result = []
    for value, symbol in ROMAN:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)

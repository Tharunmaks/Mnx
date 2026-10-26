"""Swap upper and lower case in a string.

tags: swap case upper lower string
"""

def swap_case(s: str) -> str:
    return "".join(c.lower() if c.isupper() else c.upper() for c in s)


# tests
assert swap_case("Hello") == "hELLO"

"""Reverse a string.

tags: reverse string backwards flip text
"""

def reverse_string(s: str) -> str:
    return s[::-1]


# tests
assert reverse_string("abc") == "cba"
assert reverse_string("") == ""

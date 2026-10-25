"""Count the vowels in a string.

tags: count vowels string
"""

def count_vowels(s: str) -> int:
    return sum(1 for c in s.lower() if c in "aeiou")


# tests
assert count_vowels("Hello World") == 3

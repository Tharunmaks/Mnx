"""Capitalize the first letter of every word in a string.

tags: capitalize words title case first letter
"""

def capitalize_words(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))


# tests
assert capitalize_words("hello big world") == "Hello Big World"

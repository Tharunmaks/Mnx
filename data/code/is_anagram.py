"""Check whether two strings are anagrams of each other (ignoring case and spaces).

tags: anagram two strings same letters
"""

def is_anagram(a: str, b: str) -> bool:
    norm = lambda s: sorted(s.replace(" ", "").lower())
    return norm(a) == norm(b)


# tests
assert is_anagram("Listen", "Silent")
assert not is_anagram("abc", "abd")

"""Count the number of words in a string.

tags: count words string number of words
"""

def count_words(s: str) -> int:
    return len(s.split())


# tests
assert count_words("the quick  brown fox") == 4

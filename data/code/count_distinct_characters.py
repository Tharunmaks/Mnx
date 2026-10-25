"""Count how many distinct characters a string has, ignoring case.

tags: string, distinct, unique, characters, count
"""

def count_distinct_characters(s):
    return len(set(s.lower()))

# tests
assert count_distinct_characters('Jerry') == 4
assert count_distinct_characters('') == 0

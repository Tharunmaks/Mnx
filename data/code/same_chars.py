"""Return True if two words are made of the same set of characters.

tags: string, characters, same, set, compare
"""

def same_chars(a, b):
    return set(a) == set(b)

# tests
assert same_chars('abcd', 'dddddddabc')
assert not same_chars('abcd', 'dddddddabce')

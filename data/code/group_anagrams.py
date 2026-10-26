"""Group words that are anagrams of each other.

tags: string, anagram, group, dictionary, sort
"""

def group_anagrams(words):
    out = {}
    for w in words:
        out.setdefault("".join(sorted(w)), []).append(w)
    return list(out.values())

# tests
assert group_anagrams(['eat', 'tea', 'tan']) == [['eat', 'tea'], ['tan']]
assert group_anagrams([]) == []

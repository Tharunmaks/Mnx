def is_anagram(a, b):
    """Return True if a and b are anagrams of each other."""
    return sorted(a.replace(" ", "").lower()) == sorted(b.replace(" ", "").lower())


def group_anagrams(words):
    """Group words that are anagrams of each other."""
    groups = {}
    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)
    return list(groups.values())

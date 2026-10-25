"""Return the length of the longest substring with no repeated character.

tags: string, longest, substring, repeat, unique, window
"""

def longest_substring_without_repeat(s):
    seen = {}
    best = start = 0
    for i, c in enumerate(s):
        if c in seen and seen[c] >= start:
            start = seen[c] + 1
        seen[c] = i
        best = max(best, i - start + 1)
    return best

# tests
assert longest_substring_without_repeat('abcabcbb') == 3
assert longest_substring_without_repeat('') == 0

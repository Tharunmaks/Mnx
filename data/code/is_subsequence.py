"""Return True if the first string is a subsequence of the second.

tags: string, subsequence, contains, order
"""

def is_subsequence(a, b):
    it = iter(b)
    return all(c in it for c in a)

# tests
assert is_subsequence('abc', 'ahbgdc')
assert not is_subsequence('axc', 'ahbgdc')

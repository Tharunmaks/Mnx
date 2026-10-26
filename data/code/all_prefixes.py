"""Return every prefix of a string, shortest first.

tags: string, prefix, prefixes, list
"""

def all_prefixes(s):
    return [s[:i + 1] for i in range(len(s))]

# tests
assert all_prefixes('abc') == ['a', 'ab', 'abc']
assert all_prefixes('') == []

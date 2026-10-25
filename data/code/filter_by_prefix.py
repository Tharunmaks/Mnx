"""Keep the strings that start with a given prefix.

tags: string, filter, prefix, startswith
"""

def filter_by_prefix(strings, prefix):
    return [s for s in strings if s.startswith(prefix)]

# tests
assert filter_by_prefix(['abc', 'bcd'], 'a') == ['abc']
assert filter_by_prefix([], 'a') == []

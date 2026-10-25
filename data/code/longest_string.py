"""Return the longest string in a list, or None if the list is empty.

tags: string, list, longest, maximum
"""

def longest_string(strings):
    if not strings:
        return None
    return max(strings, key=len)

# tests
assert longest_string(['a', 'bbb', 'cc']) == 'bbb'
assert longest_string([]) is None

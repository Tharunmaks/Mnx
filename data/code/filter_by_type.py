"""Keep only the values of a list that are integers.

tags: list, filter, integers, type
"""

def filter_by_type(values):
    return [v for v in values if isinstance(v, int) and not isinstance(v, bool)]

# tests
assert filter_by_type([1, 'a', 2]) == [1, 2]
assert filter_by_type([]) == []

"""Merge two dictionaries; values in the second win.

tags: dictionary, merge, combine, update
"""

def merge_dicts(a, b):
    out = dict(a)
    out.update(b)
    return out

# tests
assert merge_dicts({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
assert merge_dicts({'a': 1}, {'a': 2}) == {'a': 2}

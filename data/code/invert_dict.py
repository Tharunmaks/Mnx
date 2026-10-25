"""Swap the keys and values of a dictionary.

tags: dictionary, invert, swap, keys, values, reverse
"""

def invert_dict(d):
    return {v: k for k, v in d.items()}

# tests
assert invert_dict({'a': 1}) == {1: 'a'}
assert invert_dict({}) == {}

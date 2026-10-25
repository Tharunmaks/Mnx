"""Count how often each item appears in a list.

tags: list, count, occurrences, frequency, dictionary, tally
"""

def count_occurrences(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out

# tests
assert count_occurrences(['a', 'b', 'a']) == {'a': 2, 'b': 1}
assert count_occurrences([]) == {}

"""Group words by their length into a dictionary.

tags: string, group, dictionary, length, words
"""

def group_by_length(words):
    out = {}
    for w in words:
        out.setdefault(len(w), []).append(w)
    return out

# tests
assert group_by_length(['a', 'bb', 'cc']) == {1: ['a'], 2: ['bb', 'cc']}
assert group_by_length([]) == {}

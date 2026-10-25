"""Return the edit distance between two strings using one row of memory.

tags: string, edit, distance, levenshtein, dynamic
"""

def edit_distance_rows(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

# tests
assert edit_distance_rows('kitten', 'sitting') == 3
assert edit_distance_rows('', 'ab') == 2

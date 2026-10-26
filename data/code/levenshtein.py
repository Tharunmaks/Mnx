"""Edit (Levenshtein) distance between two strings with dynamic programming.

tags: levenshtein edit distance strings dynamic programming
"""

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# tests
assert levenshtein("kitten", "sitting") == 3
assert levenshtein("", "abc") == 3

"""Return the length of the longest increasing subsequence.

tags: list, longest, increasing, subsequence, dynamic, programming
"""

def longest_increasing_subsequence(xs):
    if not xs:
        return 0
    best = [1] * len(xs)
    for i in range(len(xs)):
        for j in range(i):
            if xs[j] < xs[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best)

# tests
assert longest_increasing_subsequence([3, 1, 2, 5]) == 3
assert longest_increasing_subsequence([]) == 0

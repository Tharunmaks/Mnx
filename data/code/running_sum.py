"""Return the running (cumulative) totals of a list.

tags: list, cumulative, running, prefix, sums
"""

def running_sum(xs):
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out

# tests
assert running_sum([1, 2, 3]) == [1, 3, 6]
assert running_sum([]) == []

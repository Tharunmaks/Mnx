"""Count how often a substring occurs in a string, overlaps included.

tags: string, count, substring, occurrences, times, overlap
"""

def how_many_times(string, substring):
    if not substring:
        return 0
    return sum(1 for i in range(len(string) - len(substring) + 1) if string[i:i + len(substring)] == substring)

# tests
assert how_many_times('aaaa', 'aa') == 3
assert how_many_times('abc', 'x') == 0

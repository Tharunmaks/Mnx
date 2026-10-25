"""Longest common prefix of a list of strings.

tags: longest common prefix strings
"""

def longest_common_prefix(words: list) -> str:
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


# tests
assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
assert longest_common_prefix(["a", "b"]) == ""

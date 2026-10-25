"""Run-length encode a string, e.g. 'aaabcc' -> 'a3b1c2'.

tags: run length encoding compress string
"""

def compress_string(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


# tests
assert compress_string("aaabcc") == "a3b1c2"
assert compress_string("") == ""

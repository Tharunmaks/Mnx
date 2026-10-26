"""Return True if the round brackets in a string are balanced.

tags: string, brackets, parentheses, balanced, matching
"""

def is_balanced_parens(s):
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0

# tests
assert is_balanced_parens('(())')
assert not is_balanced_parens('(()')

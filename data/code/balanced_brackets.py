"""Check whether brackets (), [] and {} in a string are balanced.

tags: balanced parentheses brackets valid stack
"""

def balanced_brackets(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


# tests
assert balanced_brackets("{[()()]}")
assert not balanced_brackets("([)]")
assert not balanced_brackets("((")

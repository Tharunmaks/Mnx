"""Return the shortest palindrome that begins with the given string.

tags: string, palindrome, shortest, suffix
"""

def make_palindrome(s):
    for i in range(len(s)):
        if s[i:] == s[i:][::-1]:
            return s + s[:i][::-1]
    return s

# tests
assert make_palindrome('cat') == 'catac'
assert make_palindrome('') == ''

"""Return the string with every vowel removed.

tags: string, remove, vowels, filter
"""

def remove_vowels(s):
    return "".join(c for c in s if c.lower() not in "aeiou")

# tests
assert remove_vowels('hello') == 'hll'
assert remove_vowels('') == ''
